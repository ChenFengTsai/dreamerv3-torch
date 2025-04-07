import argparse
import functools
import pathlib
import sys
import numpy as np
import torch
from torch import distributions as torchd

import ruamel.yaml as yaml


sys.path.append(str(pathlib.Path(__file__).parent))

import tools
import dreamer
from parallel import Parallel, Damy

to_np = lambda x: x.detach().cpu().numpy()

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

# Assume we're using a similar config structure as in the original Dreamer code
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
    make = lambda mode, id: dreamer.make_env(config, mode, id)
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

    
    print("Setting up SCM World Model.")
    
    # Create Dreamer agent (same as original)
    agent = dreamer.Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    )
    
    
    
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
    
    # Example of causal intervention during inference
    def inference_with_intervention(obs, state=None):
        # Preprocess observation
        obs = agent._wm.preprocess(obs)
        embed = agent._wm.encoder(obs)
        
        # Update latent state
        if state is None:
            latent = action = None
        else:
            latent, action = state

        latent, _ = agent._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        
        # Example intervention: modify deterministic state component
        agent._wm.intervene("deter", latent["deter"] * 1.5)  # Amplify deterministic state
        
        # Generate action with intervened state
        feat = agent._wm.dynamics.get_feat(latent)
        action_dist = agent._task_behavior.actor(feat)
        action = action_dist.sample()
        
        # Remove intervention
        agent._wm.remove_intervention("deter")
        
        return action, latent
    
    # Example of counterfactual reasoning
    def analyze_counterfactual(obs_sequence, action_sequence):
        # Get factual trajectory
        obs = agent._wm.preprocess(obs_sequence)
        embed = agent._wm.encoder(obs)
        factual_states, _ = agent._wm.dynamics.observe(embed, action_sequence, obs["is_first"])
        
        # Define counterfactual intervention
        initial_state = {k: v[:, 0] for k, v in factual_states.items()}
        
        # Alternative 1: Different actions
        cf_actions = action_sequence.clone()
        cf_actions[:, 5:10] = -cf_actions[:, 5:10]  # Invert actions for steps 5-10
        
        cf_trajectory1 = agent._wm.counterfactual_imagine(
            initial_state, 
            cf_actions
        )
        
        # Alternative 2: Intervention on state variable
        # Amplify stochastic state at step 4
        cf_interventions = {
            "stoch": {"timestep": 4, "value": lambda s: s * 2.0}
        }
        
        cf_trajectory2 = agent._wm.counterfactual_imagine(
            initial_state, 
            action_sequence,
            interventions=cf_interventions
        )
        
        # Compare outcomes
        factual_rewards = [agent._wm.heads["reward"](agent._wm.dynamics.get_feat(s)).mode() 
                        for s in factual_states]
        
        cf_rewards1 = [agent._wm.heads["reward"](agent._wm.dynamics.get_feat(s)).mode() 
                    for s in cf_trajectory1]
                    
        cf_rewards2 = [agent._wm.heads["reward"](agent._wm.dynamics.get_feat(s)).mode() 
                    for s in cf_trajectory2]
        
        return {
            "factual_rewards": factual_rewards,
            "cf_rewards1": cf_rewards1,
            "cf_rewards2": cf_rewards2
        }
        
        
        
        
    
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