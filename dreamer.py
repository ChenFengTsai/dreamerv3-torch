import argparse
import functools
import os
import pathlib
import sys
from torch.utils.tensorboard import SummaryWriter

os.environ["MUJOCO_GL"] = "osmesa"
# os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models

# causal module
import scm_world_model
import causal_VAE

import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd
torch.autograd.set_detect_anomaly(True)

to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        # new
        # self._future = config.future
        # self._combine = config.combine
        
        # self._counterfactual_candidate = config.counterfactual_candidate
        # self._best_candidate = config.best_candidate
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._use_amp = True if config.precision == 16 else False
        if config.causal_world_model and config.causal_mode == "SCM":
            self._wm = scm_world_model.WorldModelWithSCM(obs_space, act_space, self._step, config)
            
        elif config.causal_world_model and config.causal_mode == "causalVAE":
            self._wm = causal_VAE.CausalVAE_WorldModel(obs_space, act_space, self._step, config)

        else:
            self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        
        ### new 
        # if self._future:
        #     self.future_predictor = models.FutureHiddenPredictor(config, config.future_horizon)
        #     # add in self.future_predictor
        #     self._task_behavior = models.ImagBehavior(config, self._wm, self.future_predictor)
        # else:
        self._task_behavior = models.ImagBehavior(config, self._wm)
            
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    
    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        
        # Preprocess input observation
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)

        # Standard latent state update from world model
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])

        # If eval_state_mean is enabled, use mean of stochastic latent space
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]

        # Extract feature representation from the current latent state
        feat = self._wm.dynamics.get_feat(latent)
        # print(f"original feat shape: {feat.shape}")
        
        # new
        # if self._future:
        #     future_hidden = self._predict_future_state(latent)  # Predict future state
        #     feat = torch.cat([feat, future_hidden.detach()], dim=-1)  # Concatenate both representations
            
        # elif self._combine:
        #     # Run an imagined rollout from the current latent state
        #     horizon = self._config.imag_horizon
        #     start = {k: v.unsqueeze(0) for k, v in latent.items()}  # Add time dimension
        #     imag_feat, _, _ = self._task_behavior._imagine(start, self._task_behavior.actor, horizon)

        #     # Aggregate the imagined features over the horizon (mean pooling)
        #     imag_feat_mean = imag_feat.mean(dim=0)  # Shape: [batch_size, feature_dim]

        #     # Concatenate the aggregated imagined features to the current feature
        #     size = int(imag_feat_mean.shape[1]/2)
        #     feat = torch.cat([feat, imag_feat_mean[:, size:].detach()], dim=-1)
            
        # print(f"combined feat shape: {feat.shape}")
    
        # Choose action based on training or exploration mode
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            if self._best_candidate:
                a_best, a_worst, action, R_a_best, R_a_worst, R_a_sample = self._task_behavior.select_counterfactual_actions(feat, latent)
                action = a_best
            else:
                actor = self._task_behavior.actor(feat)
                action = actor.sample()

        # Compute log probability of the action
        logprob = actor.log_prob(action)

        # Detach latent and action for stability
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()

        # Convert action to one-hot if required
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )

        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        
        return policy_output, state



    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
                
        ### New Training for FutureHiddenPredictor ###
        # Retrieve stored hidden states from `_task_behavior._train()`
        # if self._future:
        #     with tools.RequiresGrad(self.future_predictor):
        #         with torch.cuda.amp.autocast(self._use_amp):
        #             # Retrieve stored hidden states (current and future)
        #             first_h_t = self._task_behavior.saved_deter[0].detach()  # first
        #             first_s_t = self._task_behavior.saved_stoch[0].detach()  # first
        #             # Target: The deterministic hidden state **10 steps into the future**
        #             h_t_future = self._task_behavior.saved_deter[-1].detach()
        #             # print("h_t_future shape:", h_t_future.shape)
            
        #             # Predict the future hidden states from the current step
        #             h_t_future_pred = self.future_predictor(first_h_t, first_s_t)
        #             # print("h_t_future_pred shape:", h_t_future_pred.shape)

        #             # Loss: Predicting future hidden state from real imagined states
        #             future_loss = torch.nn.functional.mse_loss(h_t_future_pred, h_t_future)

        #     # Log statistics
        #     metrics["future_loss"] = to_np(future_loss)

        #     # Apply optimization step using self._future_opt
        #     with tools.RequiresGrad(self):
        #         metrics.update(self._task_behavior._future_opt(future_loss, self.future_predictor.parameters()))


        # Store metrics
        # for name, value in metrics.items():
        #     if not name in self._metrics.keys():
        #         self._metrics[name] = [value]
        #     else:
        #         self._metrics[name].append(value)
    

    ### New
    # def _predict_future_state(self, latent):
    #     """
    #     Predicts a future latent state using a learned function.
    #     """
    #     h_t = latent["deter"]  # Deterministic hidden state
    #     s_t = latent["stoch"]  # Stochastic latent state

    #     # Predict future latent state
    #     future_latent_pred = self.future_predictor(h_t, s_t)

    #     return future_latent_pred

    


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id, modify=(config.modify_env, config.gravity_scale)
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        import envs.metaworld_env as metaworld

        env = metaworld.MetaWorldEnv(
            task, action_repeat=config.action_repeat, size=config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env




def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    
    # Prevent creating new tensor summaries during eval_only
    ### Logger ###
    if config.eval_only:
        if config.action_perturb:
            eval_logdir = logdir / f"eval_only_log_action_perturb_{config.action_noise_scale}"
        elif config.modify_env:
            eval_logdir = logdir / f"eval_only_log_gravity_{config.gravity_scale}"
            print(config.gravity_scale)
        else:
            eval_logdir = logdir / "eval_only_log"
        logger = tools.Logger(eval_logdir, config.action_repeat * step)
    else:
        logger = tools.Logger(logdir, config.action_repeat * step)


    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    
    # setup 
    # if config.future:
    #     config.future_horizon = 10
    #     config.future_dim = config.dyn_deter

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    
    
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    
    if config.action_perturb:
        def add_action_noise(action):
            noise = torch.randn_like(action) * config.action_noise_scale
            # print("Noise scale:", self._config.action_noise_scale)
            perturbed_action = action + noise
            perturbed_action = torch.clamp(perturbed_action, -1.0, 1.0)
            return perturbed_action

        # Patch Dreamer policy for eval noise inside eval_only block
        original_policy = agent._policy

        def noisy_policy(obs, state, training=False):
            policy_output, state = original_policy(obs, state, training)
            policy_output["action"] = add_action_noise(policy_output["action"])
            # print(policy_output["action"])
            return policy_output, state

        agent._policy = noisy_policy
        
    agent.requires_grad_(requires_grad=False)
    
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
        
    
    if config.eval_only:
        print("Running evaluation only mode...")
        # print(eval_eps)
        eval_policy = functools.partial(agent, training=False)
        tools.simulate(
            eval_policy,
            eval_envs,
            eval_eps,
            config.evaldir,
            logger,
            is_eval=True,
            episodes=config.eval_episode_num,
        )
        # if config.video_pred_log:
        #     video_pred = agent._wm.video_pred(next(eval_dataset))
        #     logger.video("eval_openl", to_np(video_pred))

        print("Evaluation complete.")
        for env in eval_envs:
            try:
                env.close()
            except Exception:
                pass
        return

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass
        

    
    # Continue with normal training loop...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))

