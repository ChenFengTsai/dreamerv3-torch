import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class CausalVariable(nn.Module):
    """Represents a node in the Structural Causal Model."""
    def __init__(self, name, shape=None, parents=None, device="cuda"):
        super(CausalVariable, self).__init__()
        self.name = name
        self.shape = shape
        self.parents = parents or []
        self.device = device
        self._intervened = False
        self._intervention_value = None
    
    def forward(self, parent_values=None, **kwargs):
        """Computes the value of this variable based on parents.
        If intervened upon, returns the intervention value instead.
        """
        if self._intervened:
            return self._intervention_value
        else:
            return self._compute_from_parents(parent_values, **kwargs)
    
    def _compute_from_parents(self, parent_values, **kwargs):
        """To be implemented by subclasses."""
        raise NotImplementedError("Structural equation not implemented")
    
    def do(self, value):
        """Perform a do-operation (intervention) on this variable."""
        self._intervened = True
        self._intervention_value = value
        return self
    
    def undo(self):
        """Remove intervention."""
        self._intervened = False
        self._intervention_value = None
        return self
    
    def is_intervened(self):
        return self._intervened


class SCM(nn.Module):
    """Structural Causal Model representing the causal relationships."""
    def __init__(self):
        super(SCM, self).__init__()
        self.variables = nn.ModuleDict()
        self._graph = {}  # adjacency representation of causal graph
    
    def add_variable(self, variable):
        """Add a causal variable to the model."""
        self.variables[variable.name] = variable
        self._graph[variable.name] = [p for p in variable.parents]
        return self
    
    def get_variable(self, name):
        """Get a causal variable by name."""
        return self.variables[name]
    
    def get_parents(self, name):
        """Get parents of a variable."""
        return self._graph.get(name, [])
    
    def do(self, name, value):
        """Perform intervention on a variable."""
        if name in self.variables:
            self.variables[name].do(value)
        return self
    
    def undo(self, name=None):
        """Remove intervention from a variable or all variables."""
        if name is None:
            # Remove all interventions
            for var in self.variables.values():
                var.undo()
        elif name in self.variables:
            self.variables[name].undo()
        return self
    
    def get_intervention_status(self):
        """Get intervention status of all variables."""
        return {name: var.is_intervened() for name, var in self.variables.items()}


class DeterVariable(CausalVariable):
    """Deterministic component of the RSSM state."""
    def __init__(self, rssm, device="cuda"):
        super(DeterVariable, self).__init__("deter", parents=["prev_deter", "prev_stoch", "action"], device=device)
        self.rssm = rssm
        
    def _compute_from_parents(self, parent_values, is_first=None, **kwargs):
        prev_deter = parent_values.get("prev_deter")
        prev_stoch = parent_values.get("prev_stoch")
        action = parent_values.get("action")
        
        # Handle initialization or reset
        if prev_deter is None or prev_stoch is None or (is_first is not None and torch.any(is_first)):
            batch_size = action.shape[0] if action is not None else is_first.shape[0]
            init_state = self.rssm.initial(batch_size)
            
            if prev_deter is None and prev_stoch is None:
                # Initial state for all
                prev_deter = init_state["deter"]
                prev_stoch = init_state["stoch"]
            elif is_first is not None and torch.any(is_first):
                # Partial reset where is_first=True
                mask = is_first[:, None].to(prev_deter.device)
                while len(mask.shape) < len(prev_deter.shape):
                    mask = mask[..., None]
                prev_deter = prev_deter * (1 - mask) + init_state["deter"] * mask
                
                mask = is_first[:, None].to(prev_stoch.device)
                while len(mask.shape) < len(prev_stoch.shape):
                    mask = mask[..., None]
                prev_stoch = prev_stoch * (1 - mask) + init_state["stoch"] * mask
        
        # For initial step without action
        if action is None:
            batch_size = prev_stoch.shape[0]
            action = torch.zeros((batch_size, self.rssm._num_actions), device=prev_stoch.device)
            
        # Process stochastic state
        if self.rssm._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self.rssm._stoch * self.rssm._discrete]
            prev_stoch = prev_stoch.reshape(shape)
            
        # Combine stochastic state and action
        x = torch.cat([prev_stoch, action], -1)
        x = self.rssm._img_in_layers(x)
        
        # Update deterministic state through GRU
        x, deter = self.rssm._cell(x, [prev_deter])
        deter = deter[0]  # GRU returns a list
        
        return deter


class StochVariable(CausalVariable):
    """Stochastic component of the RSSM state."""
    def __init__(self, rssm, device="cuda"):
        super(StochVariable, self).__init__("stoch", parents=["deter", "embed"], device=device)
        self.rssm = rssm
        
    def _compute_from_parents(self, parent_values, **kwargs):
        deter = parent_values["deter"]
        embed = parent_values.get("embed")  # Optional for imagination
        
        if embed is not None:
            # Posterior computation (with observation)
            x = torch.cat([deter, embed], -1)
            x = self.rssm._obs_out_layers(x)
            stats = self.rssm._suff_stats_layer("obs", x)
        else:
            # Prior computation (without observation)
            x = self.rssm._img_out_layers(deter)
            stats = self.rssm._suff_stats_layer("ims", x)
            
        dist = self.rssm.get_dist(stats)
        stoch = dist.sample()
        
        # Return sample and distribution parameters
        if self.rssm._discrete:
            return {"stoch": stoch, "logit": stats["logit"]}
        else:
            return {"stoch": stoch, "mean": stats["mean"], "std": stats["std"]}


class SCMRSSM(nn.Module):
    """SCM-based implementation of the RSSM dynamics model."""
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super(SCMRSSM, self).__init__()
        
        # Create a standard RSSM for its components and functions
        self._rssm = networks.RSSM(
            stoch, deter, hidden, rec_depth, discrete, act, norm, 
            mean_act, std_act, min_std, unimix_ratio, initial, 
            num_actions, embed, device
        )
        
        # Store parameters
        self._stoch = stoch
        self._deter = deter
        self._discrete = discrete
        self._device = device
        self._num_actions = num_actions
        
        # Create the SCM
        self.scm = SCM()
        
        # Add causal variables
        self.scm.add_variable(DeterVariable(self._rssm, device))
        self.scm.add_variable(StochVariable(self._rssm, device))
        
    def forward(self, *args, **kwargs):
        """Forward pass, same signature as original RSSM"""
        raise NotImplementedError("Use specific methods instead of forward")
    
    def initial(self, batch_size):
        """Return initial state, same as original RSSM"""
        return self._rssm.initial(batch_size)
    
    def observe(self, embed, action, is_first, state=None):
        """Process a sequence of embedded observations and actions.
        Returns posterior and prior states.
        """
        # Extract dimensions
        batch_size, seq_len = embed.shape[0], embed.shape[1]
        
        # Storage for results
        post_states = []
        prior_states = []
        
        # Process sequence step by step
        prev_post = state
        for t in range(seq_len):
            # Current timestep data
            curr_embed = embed[:, t]
            curr_action = action[:, t] if action is not None else None
            curr_is_first = is_first[:, t] if is_first is not None else None
            
            # Compute posterior state
            post, prior = self.obs_step(
                prev_post, 
                curr_action, 
                curr_embed, 
                curr_is_first
            )
            
            # Store states
            post_states.append(post)
            prior_states.append(prior)
            
            # Update for next step
            prev_post = post
        
        # Combine into batch
        post_out = {}
        prior_out = {}
        
        for key in post_states[0].keys():
            post_out[key] = torch.stack([s[key] for s in post_states], dim=1)
            prior_out[key] = torch.stack([s[key] for s in prior_states], dim=1)
        
        return post_out, prior_out
    
    def imagine(self, action, state):
        """Imagine a single step using the prior (without observation)."""
        return self.img_step(state, action)
    
    def imagine_with_action(self, action_sequence, initial_state):
        """Imagine a sequence of states given a sequence of actions."""
        batch_size, seq_len = action_sequence.shape[0], action_sequence.shape[1]
        
        # Initialize with given state
        state = initial_state
        states = []
        
        # Imagine forward
        for t in range(seq_len):
            # Get current action
            curr_action = action_sequence[:, t]
            
            # Imagine next state
            state = self.img_step(state, curr_action)
            states.append(state)
        
        # Convert list of states to tensor format
        states_out = {}
        for key in states[0].keys():
            states_out[key] = torch.stack([s[key] for s in states], dim=1)
            
        return states_out
    
    def obs_step(self, prev_state, action, embed, is_first, sample=True):
        """Single step of observation update, using the SCM."""
        deter_var = self.scm.get_variable("deter")
        stoch_var = self.scm.get_variable("stoch")
        
        # Handle very first step
        if prev_state is None:
            batch_size = embed.shape[0]
            prev_state = self.initial(batch_size)
        
        # Get deterministic state
        deter = deter_var.forward({
            "prev_deter": prev_state["deter"] if prev_state else None,
            "prev_stoch": prev_state["stoch"] if prev_state else None,
            "action": action
        }, is_first=is_first)
        
        # Get prior stochastic state (without embedding)
        prior_out = stoch_var.forward({
            "deter": deter,
            "embed": None  # No embedding for prior
        })
        
        prior = {
            "deter": deter,
            "stoch": prior_out["stoch"]
        }
        
        if self._discrete:
            prior["logit"] = prior_out["logit"]
        else:
            prior["mean"] = prior_out["mean"]
            prior["std"] = prior_out["std"]
        
        # Get posterior stochastic state (with embedding)
        post_out = stoch_var.forward({
            "deter": deter,
            "embed": embed
        })
        
        posterior = {
            "deter": deter,
            "stoch": post_out["stoch"]
        }
        
        if self._discrete:
            posterior["logit"] = post_out["logit"]
        else:
            posterior["mean"] = post_out["mean"]
            posterior["std"] = post_out["std"]
        
        return posterior, prior
    
    def img_step(self, prev_state, action, sample=True):
        """Single step of imagination (prior only), using the SCM."""
        deter_var = self.scm.get_variable("deter")
        stoch_var = self.scm.get_variable("stoch")
        
        # Get deterministic state
        deter = deter_var.forward({
            "prev_deter": prev_state["deter"],
            "prev_stoch": prev_state["stoch"],
            "action": action
        })
        
        # Get prior stochastic state (without embedding)
        prior_out = stoch_var.forward({
            "deter": deter,
            "embed": None  # No embedding for prior
        })
        
        prior = {
            "deter": deter,
            "stoch": prior_out["stoch"]
        }
        
        if self._discrete:
            prior["logit"] = prior_out["logit"]
        else:
            prior["mean"] = prior_out["mean"]
            prior["std"] = prior_out["std"]
        
        return prior
    
    def get_feat(self, state):
        """Extract features from state, matching original RSSM."""
        return self._rssm.get_feat(state)
    
    def get_dist(self, state):
        """Get distribution from state, matching original RSSM."""
        return self._rssm.get_dist(state)
    
    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        """Compute KL divergence loss, matching original RSSM."""
        return self._rssm.kl_loss(post, prior, free, dyn_scale, rep_scale)
    
    def intervene(self, variable, value):
        """Perform intervention on a variable in the SCM."""
        self.scm.do(variable, value)
        return self
    
    def remove_intervention(self, variable=None):
        """Remove intervention from a variable (or all variables)."""
        self.scm.undo(variable)
        return self


class WorldModelWithSCM(nn.Module):
    """Dreamer's world model using SCM-based dynamics."""
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModelWithSCM, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        
        # Create encoder
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        
        # Create SCM-based RSSM
        self.dynamics = SCMRSSM(
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
        
        # Create heads (same as original)
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
        
        # Setup optimizer (same as original)
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
        
        # Loss scales (same as original)
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        
        print(
            f"World Model with SCM has {sum(param.numel() for param in self.parameters())} variables."
        )
    
    # The rest of the methods are identical to the original WorldModel
    
    def _train(self, data):
        """Train the world model on batched data."""
        data = self.preprocess(data)
        
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # Compute embeddings and latent states
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                
                # Compute KL divergence loss
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                
                # Compute prediction losses
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
                
                # Scale losses
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                
                # Combine all losses
                model_loss = sum(scaled.values()) + kl_loss
                
            # Optimization step
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())
        
        # Collect metrics
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
            
        # Context for policy learning
        context = dict(
            embed=embed,
            feat=self.dynamics.get_feat(post),
            kl=kl_value,
            postent=self.dynamics.get_dist(post).entropy(),
        )
        
        # Detach post for return
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics
    
    def preprocess(self, obs):
        """Preprocess observations."""
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = obs["discount"].unsqueeze(-1)
        assert "is_first" in obs
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs
    
    def video_pred(self, data):
        """Generate video predictions."""
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        
        # Combine observed and predicted frames
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)
    
    # Additional methods for causal interventions
    
    def intervene(self, variable, value):
        """Perform causal intervention on a dynamics variable."""
        self.dynamics.intervene(variable, value)
        return self
    
    def remove_intervention(self, variable=None):
        """Remove intervention."""
        self.dynamics.remove_intervention(variable)
        return self
    
    def counterfactual_imagine(self, initial_state, actions, interventions=None):
        """Imagine a trajectory with counterfactual interventions.
        
        Args:
            initial_state: Starting state
            actions: Action sequence
            interventions: Dict of {variable: value} for interventions
            
        Returns:
            List of states from counterfactual rollout
        """
        # Apply interventions
        if interventions:
            for var, value in interventions.items():
                self.intervene(var, value)
        
        # Run imagination
        states = [initial_state]
        current_state = initial_state
        
        for t in range(len(actions)):
            next_state = self.dynamics.img_step(current_state, actions[t])
            states.append(next_state)
            current_state = next_state
        
        # Remove interventions
        if interventions:
            self.remove_intervention()
            
        return states