import copy
import torch
from torch import nn

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        print('here')
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)
    



class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, future_predictor=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        # new
        # self._future = config.future
        # self._combine = config.combine
        # self._use_counterfactuals = config.use_counterfactuals
        # self._counterfactual_candidate = config.counterfactual_candidate

        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        
        # add self.future_predictor new
        # if self._future:
        #     self._future_predictor = future_predictor
        #     self._future_opt = tools.Optimizer(
        #         "future",
        #         self._future_predictor.parameters(),
        #         config.model_lr,
        #         config.opt_eps,
        #         config.grad_clip,
        #         config.weight_decay,
        #         opt=config.opt,
        #         use_amp=self._use_amp,
        #     )
            
        # if self._future:
        #     if config.dyn_discrete:
        #         feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter + config.future_dim
        #     else:
        #         feat_size = config.dyn_stoch + config.dyn_deter + config.future_dim
                
        # elif self._combine:
        #     if config.dyn_discrete:
        #         feat_size = 2*(config.dyn_stoch * config.dyn_discrete + config.dyn_deter)
        #     else:
        #         feat_size = 2*(config.dyn_stoch + config.dyn_deter)
            
        # else:
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
                    
        # print(f'feat_size:', feat_size)
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                
                # new
                # if self._future:
                #     self.saved_deter = imag_state["deter"]
                #     self.saved_stoch = imag_state["stoch"]
                
                # print('deter shape', self.saved_deter.shape)
                # print('stoch shape', self.saved_stoch.shape)
                
                ## todo
                # if self._counterfactual_candidate:
                #     # ===== Counterfactual branching at t=0 =====
                #     initial_state = {k: v[0].detach() for k, v in imag_state.items()}
                #     initial_feat = imag_feat[0].detach()
                    
                #     (a_best, a_worst, a_sample, R_a_best, R_a_worst, R_a_sample) = self.select_counterfactual_actions(initial_feat, initial_state)
                    
                #     # Repeat start states twice (for best & worst)
                #     start_double = {k: v.repeat_interleave(2, dim=0) for k, v in start.items()}

                #     # Concatenate best and worst actions for forced rollout
                #     forced_actions = torch.cat([a_best, a_worst], dim=0)
                    
                #     imag_feat_best, imag_state_best, imag_action_best = self._imagine(
                #         start, self.actor, self._config.imag_horizon, first_action=a_best
                #     )
                #     imag_feat_worst, imag_state_worst, imag_action_worst = self._imagine(
                #         start, self.actor, self._config.imag_horizon, first_action=a_worst
                #     )
                    
                #     imag_feat = torch.cat([imag_feat_best, imag_feat_worst], dim=1)
                #     imag_state = {k: torch.cat([imag_state_best[k], imag_state_worst[k]], dim=1) for k in imag_state_best}
                #     imag_action = torch.cat([imag_action_best, imag_action_worst], dim=1)

                #     loss_regret = R_a_best - R_a_sample
                #     loss_impact = -(R_a_sample - R_a_worst)

                #     metrics["loss_regret"] = loss_regret.item()
                #     metrics["loss_impact"] = loss_impact.item()
                #     metrics["counterfactual_R_best"] = R_a_best.item()
                #     metrics["counterfactual_R_sample"] = R_a_sample.item()
                #     metrics["counterfactual_R_worst"] = R_a_worst.item()
                    
                # # Enhance with counterfactual reasoning if enabled
                # elif self._use_counterfactuals:
                #     cf_metrics = self._train_with_counterfactuals(start, objective)
                #     metrics.update(cf_metrics)
                    
                    
                    
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                
           
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                
                # if self._counterfactual_candidate:
                #     # print("previous_actor_loss:", actor_loss)
                #     # contrastive learning
                #     actor_loss = actor_loss - self._config.regret_scale * loss_regret - self._config.impact_scale * loss_impact
                #     actor_loss = actor_loss.squeeze()
                    
                # print("actor_loss:", actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, first_action=None):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        # if self._future:
        #     init_state = (start, None, None)

        # elif self._combine:
        #     initial_feat = dynamics.get_feat(start).detach()
        #     state_key = start.keys()
        #     init_state = (
        #         {
        #             **{f'{k}': v for k, v in start.items()},
        #             'moving_avg': initial_feat,
        #             'count': torch.tensor(1, device=start['deter'].device)
        #         },
        #         None,
        #         None
        #     )
        # else:
        init_state = (start, None, None)

        def step(prev, _):
            # if self._future:
            #     state, _, _ = prev
            #     feat = dynamics.get_feat(state).detach()
            #     future_h_t_pred = self._future_predictor(state["deter"], state["stoch"]).detach()
            #     policy_input = torch.cat([feat, future_h_t_pred], dim=-1)
            #     action = policy(policy_input).sample()
            #     succ = dynamics.img_step(state, action)
            #     return succ, policy_input, action

            # elif self._combine:
            #     prev_state, _, _ = prev
            #     state = {k: v for k, v in prev_state.items() if k in state_key}
            #     moving_avg = prev_state['moving_avg']
            #     count = prev_state['count']

                # mode = "ema"
                # if mode == 'avg':
                #     feat = dynamics.get_feat(state).detach()
                #     avg_feat = feat if count == 1 else moving_avg
                #     policy_input = torch.cat([feat, avg_feat], dim=-1)
                    
                # elif mode == 'ema':
                #     feat = dynamics.get_feat(state).detach()
                #     alpha = 0.99 if count > 1 else 0.0
                #     new_moving_avg = alpha * moving_avg + (1 - alpha) * feat
                #     policy_input = torch.cat([feat, new_moving_avg], dim=-1)

                # action = policy(policy_input).sample()
                # succ = dynamics.img_step(state, action)

                # new_moving_avg = (moving_avg * count + feat) / (count + 1)
                # new_count = count + 1

                # new_state = {
                #     **{f'{k}': v for k, v in succ.items()},
                #     'moving_avg': new_moving_avg,
                #     'count': new_count
                # }
                # return new_state, policy_input, action
                
            # else:
            state, _, _ = prev
            feat = dynamics.get_feat(state).detach()
            action = policy(feat).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        # Run static_scan
        # if self._counterfactual_candidate:
        #     if first_action is not None:
        #         # First step manually
        #         state = dynamics.img_step(init_state[0], first_action)
        #         feat = dynamics.get_feat(state).detach()

        #         new_init_state = (state, feat, first_action)

        #         # Continue static_scan from t=1
        #         succ_out, feats, actions = tools.static_scan(step, [torch.arange(horizon - 1)], new_init_state)

        #         # Prepend first step results:
        #         succ_out = {k: torch.cat([state[k].unsqueeze(0), v], dim=0) for k, v in succ_out.items()}
        #         feats = torch.cat([feat.unsqueeze(0), feats], dim=0)
        #         actions = torch.cat([first_action.unsqueeze(0), actions], dim=0)
                
        #     else:
        #         succ_out, feats, actions = tools.static_scan(step, [torch.arange(horizon)], init_state)
        # else:
        
        succ_out, feats, actions = tools.static_scan(step, [torch.arange(horizon)], init_state)

        # Extract correct states
        # if self._combine:
        #     states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ_out.items() if k in state_key}
        # else:
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ_out.items()}

        return feats, states, actions
    

    
    def _train_with_counterfactuals(self, start, objective):
        """Additional training using counterfactual reasoning."""
        if self._config.cf_importance:
            self.cf_importance = self._config.cf_importance
        else:
            self.cf_importance = 0.5
            
        metrics = {}
        
        # Sample a few starting states
        batch_size = next(iter(start.values())).shape[0]
        num_samples = min(batch_size, 4)  # Use at most 4 counterfactual samples
        indices = torch.randperm(batch_size)[:num_samples]
        
        # Extract selected states
        sampled_states = {k: v[indices] for k, v in start.items()}
        
        # For each state, imagine a counterfactual trajectory
        cf_losses = []
        
        for i in range(num_samples):
            state_i = {k: v[i:i+1] for k, v in sampled_states.items()}
            
            # Get factual actions from current policy
            feat = self._world_model.dynamics.get_feat(state_i)
            action_dist = self.actor(feat)
            factual_action = action_dist.sample()
            
            # Imagine factual next state and outcome
            factual_next = self._world_model.dynamics.img_step(state_i, factual_action)
            factual_feat = self._world_model.dynamics.get_feat(factual_next)
            factual_reward = objective(factual_feat, factual_next, factual_action)
            
            # Generate counterfactual action (opposite direction)
            cf_action = -factual_action
            
            # Imagine counterfactual outcome
            cf_next = self._world_model.dynamics.img_step(state_i, cf_action)
            cf_feat = self._world_model.dynamics.get_feat(cf_next)
            cf_reward = objective(cf_feat, cf_next, cf_action)
            
            # If counterfactual is better, update policy to increase probability of that action
            reward_diff = cf_reward - factual_reward
            print("reward_diff", reward_diff)
            print("cf action", action_dist.log_prob(cf_action))
            
            if reward_diff > 0:
                # Learn from the counterfactual - increase probability of better action
                cf_loss = -action_dist.log_prob(cf_action) * reward_diff
                cf_losses.append(cf_loss)
                
        # If we found beneficial counterfactuals, optimize for them
        if cf_losses:
            cf_loss = torch.mean(torch.cat(cf_losses))
            
            with tools.RequiresGrad(self.actor):
                opt_info = self._actor_opt(
                    cf_loss * self.cf_importance, 
                    self.actor.parameters()
                )
                metrics.update({f"cf_{k}": v for k, v in opt_info.items()})
            
            metrics["cf_loss"] = float(cf_loss.detach().cpu().numpy())
            metrics["cf_count"] = len(cf_losses)
        
        return metrics


    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
       
       
    def select_counterfactual_actions(self, feat, latent_state):
        num_candidates = self._config.num_action_candidates
        imagination_horizon = self._config.counterfactual_horizon

        # At the very beginning (latent_state = initial state)
        start_state = {k: v.unsqueeze(0) for k, v in latent_state.items()}
        
        # Sample multiple candidate actions from the same distribution
        actor = self.actor(feat)
        candidate_actions = actor.sample((num_candidates,))  # [N, action_dim]    
        candidate_rewards = []
        for i in range(num_candidates):
            branch_start = {k: v.unsqueeze(0) for k, v in latent_state.items()}
            branch_feats, _, _ = self._imagine(
                start_state,
                self.actor,
                imagination_horizon,
                first_action=candidate_actions[i]
            )
            branch_reward = self._world_model.heads["reward"](branch_feats).mode()
            
            branch_reward_total = branch_reward.mean(dim=(0, 1)) 
            candidate_rewards.append(branch_reward_total)
            
        candidate_rewards = torch.stack(candidate_rewards)
        candidate_rewards = candidate_rewards.squeeze(-1)

        # Get best, worst, and a random sampled action (not best or worst)
        best_idx = torch.argmax(candidate_rewards)
        worst_idx = torch.argmin(candidate_rewards)
        

        # Sample another fresh action from actor as a_sample (current policy action)
        a_sample = actor.sample()  # Just one from the dist

        a_best = candidate_actions[best_idx]
        a_worst = candidate_actions[worst_idx]

        R_a_best = candidate_rewards[best_idx]
        R_a_worst = candidate_rewards[worst_idx]
        
        # Evaluate return from a_sample branch
        branch_start = {k: v.unsqueeze(0) for k, v in latent_state.items()}
        sample_feats, _, _ = self._imagine(
            branch_start, 
            self.actor, 
            imagination_horizon,
            first_action=a_sample
        )
        R_a_sample = self._world_model.heads["reward"](sample_feats).mean()
        sample_reward_total = R_a_sample.mean(dim=(0, 1)) 

        return a_best, a_worst, a_sample, R_a_best, R_a_worst, sample_reward_total
       
# new     
# class FutureHiddenPredictor(nn.Module):
#     def __init__(self, config, future_horizon=10):
#         super().__init__()
#         self.future_horizon = future_horizon
#         self._device = config.device  # Store device

#         if config.dyn_discrete:
#             feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
#         else:
#             feat_size = config.dyn_stoch + config.dyn_deter

#         self.fc = nn.Sequential(
#             nn.Linear(feat_size, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1024),  # Predict future hidden state
#             nn.ReLU(),
#             nn.Linear(1024, self.deter_dim)  # Predict future hidden state
#         )

#         # Move the model to the specified device
#         self.to(self._device)

#     def forward(self, h_t, s_t):
#         if h_t is None:
#             raise ValueError("h_t is None!")
#         if s_t is None:
#             raise ValueError("s_t is None!")

#         # Move inputs to the correct device
#         h_t = h_t.to(self._device)
#         s_t = s_t.to(self._device)

#         # Concatenate current hidden state (h_t, s_t)
#         s_t = s_t.reshape(s_t.shape[0], -1)
#         input_features = torch.cat([h_t, s_t], dim=-1)

#         # Predict future hidden state
#         return self.fc(input_features).to(self._device)


class FutureHiddenPredictor(nn.Module):
    def __init__(self, config, num_layers=6, nhead=8):
        super().__init__()
        self.device = config.device  # Store device

        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        self.deter_dim = config.dyn_deter
        self.stoch_dim = config.dyn_stoch * config.dyn_stoch
        self.action_dim = config.num_actions

        transformer_dim = 512  # Define Transformer hidden size

        # Embedding layer to project input features
        self.embedding = nn.Linear(feat_size, transformer_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=nhead, 
            dim_feedforward=1024, 
            activation="relu", 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection layer
        self.fc_out = nn.Linear(transformer_dim, self.deter_dim)

    def forward(self, h_t, s_t):
        if h_t is None or s_t is None:
            raise ValueError("h_t or s_t is None!")

        s_t = s_t.reshape(s_t.shape[0], -1)  # Flatten stochastic state
        input_features = torch.cat([h_t, s_t], dim=-1)  # Concatenate states

        # Pass through embedding layer
        embedded_features = self.embedding(input_features)  # Shape: (batch_size, feature_dim)

        # Transformer expects (batch_size, sequence_length, feature_dim)
        # Since we don't have a temporal sequence, we reshape it as a sequence of length 1
        embedded_features = embedded_features.unsqueeze(1)  # Shape: (batch_size, 1, transformer_dim)

        # Transformer Encoder (single step)
        transformed_features = self.transformer(embedded_features)

        # Get the output (since sequence length is 1, we extract the first element)
        output = self.fc_out(transformed_features[:, 0, :])

        return output.to(self.device)


