import argparse
import os
import pathlib
import sys
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import defaultdict

# Add the project root to path
sys.path.append(str(pathlib.Path(__file__).parent))

import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

# Import your model and environment
import models
import causal_VAE
from dreamer import make_env, count_steps


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate generalization of trained model')
    parser.add_argument('--logdir', type=str, required=True, 
                        help='Path to trained model directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run evaluation on')
    parser.add_argument('--task', type=str, default='dmc_reacher_easy',
                        help='Task to evaluate on')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for evaluation')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes to evaluate per condition')
    parser.add_argument('--action_repeat', type=int, default=2,
                        help='Action repeat for the environment')
    parser.add_argument('--size', nargs=2, type=int, default=(64, 64),
                        help='Image size for observations')
    parser.add_argument('--time_limit', type=int, default=1000,
                        help='Time limit for each episode')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config file from training (optional)')
    return parser.parse_args()


def load_config(logdir, config_path=None):
    """Load configuration from the log directory or provided path"""
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Try to find config in the log directory
    config_path = os.path.join(logdir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # If no config file found, return default configuration
    print("No config file found. Using default configuration.")
    return {
        'task': 'dmc_reacher_easy',
        'action_repeat': 2,
        'size': [64, 64],
        'time_limit': 1000,
        'causal_world_model': True,
        'causal_mode': 'causalVAE',
        'causal_factors': 7,
        'grayscale': False,
        'dyn_hidden': 512,
        'dyn_deter': 512,
        'dyn_stoch': 32,
        'dyn_discrete': 32,
        'dyn_rec_depth': 1,
        'encoder': {
            'mlp_keys': '$^', 
            'cnn_keys': 'image', 
            'cnn_depth': 32,
            'act': 'SiLU',
            'norm': True
        },
        'decoder': {
            'mlp_keys': '$^', 
            'cnn_keys': 'image', 
            'cnn_depth': 32,
            'act': 'SiLU',
            'norm': True
        },
        'reward_EMA': True,
        'precision': 32,
        'discount': 0.997,
        'compile': False,
    }


def setup_environments(args, config, test_conditions):
    """Set up environments for each test condition"""
    envs_by_condition = {}
    
    for condition_name, params in test_conditions.items():
        # Create evaluation directory for this condition
        eval_dir = pathlib.Path(args.logdir) / f"eval_generalization_{condition_name}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environments with the specific parameters
        env_list = []
        for i in range(args.eval_episodes):
            # Extract condition-specific parameters
            arm_length_scale = params.get('arm_length_scale', None)
            joint_damping_scale = params.get('joint_damping_scale', None)
            arm_mass_scale = params.get('arm_mass_scale', None)
            
            # Set up environment
            env = make_env_with_params(
                config, 
                "eval", 
                i, 
                arm_length_scale=arm_length_scale,
                joint_damping_scale=joint_damping_scale,
                arm_mass_scale=arm_mass_scale
            )
            env = Damy(env)  # Wrap environment
            env_list.append(env)
        
        envs_by_condition[condition_name] = {
            'envs': env_list,
            'eval_dir': eval_dir,
        }
    
    return envs_by_condition


def make_env_with_params(config, mode, id, 
                         arm_length_scale=None, 
                         joint_damping_scale=None,
                         arm_mass_scale=None):
    """Create environment with specific physical parameters"""
    import envs.dmc as dmc
    
    # Determine if we need to modify the environment
    modify = any(param is not None for param in 
                [arm_length_scale, joint_damping_scale, arm_mass_scale])
    
    # Create environment with parameters
    params = {
        'arm_length_scale': arm_length_scale,
        'joint_damping_scale': joint_damping_scale,
        'arm_mass_scale': arm_mass_scale
    }
    
    env = dmc.DeepMindControl(
        config['task'].split('_', 1)[1],  # Extract task name (e.g., "reacher_easy")
        config['action_repeat'],
        config['size'],
        seed=config['seed'] + id,
        modify=(modify, params)
    )
    
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, config['time_limit'])
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    
    return env


def load_agent(args, config, obs_space, act_space):
    """Load the trained agent"""
    # Create logger
    log_dir = pathlib.Path(args.logdir)
    step = count_steps(log_dir / "train_eps")
    logger = tools.Logger(log_dir / "eval_generalization", config['action_repeat'] * step)
    
    # Create datasets (empty for evaluation)
    eval_eps = tools.load_episodes(log_dir / "eval_eps", limit=1)
    dataset = make_dataset(eval_eps, config)
    
    # Create agent based on causal mode
    if config['causal_world_model'] and config['causal_mode'] == 'causalVAE':
        agent = causal_VAE.CausalVAE_Dreamer(
            obs_space,
            act_space,
            config,
            logger,
            dataset
        ).to(args.device)
    else:
        agent = models.Dreamer(
            obs_space,
            act_space,
            config,
            logger,
            dataset
        ).to(args.device)
    
    # Load trained weights
    checkpoint = torch.load(log_dir / "latest.pt", map_location=args.device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
    agent.requires_grad_(requires_grad=False)
    agent.eval()
    
    return agent, logger


def make_dataset(episodes, config):
    """Create a dataset from episodes for the agent"""
    generator = tools.sample_episodes(episodes, config['batch_length'])
    dataset = tools.from_generator(generator, config['batch_size'])
    return dataset


def evaluate_condition(agent, envs, eval_dir, logger, config):
    """Evaluate the agent on a specific condition"""
    # Set up evaluation episodes
    eval_eps = {}
    
    # Create evaluation policy (no training)
    eval_policy = lambda obs, reset, state=None: agent(obs, reset, state, training=False)
    
    # Run simulation
    state = tools.simulate(
        eval_policy,
        envs,
        eval_eps,
        eval_dir,
        logger,
        is_eval=True,
        episodes=len(envs)
    )
    
    # Collect evaluation results
    results = {
        'rewards': [],
        'lengths': []
    }
    
    # Extract metrics from episodes
    for ep_id, episode in eval_eps.items():
        rewards = episode['reward']
        total_reward = float(np.sum(rewards))
        episode_length = len(rewards) - 1
        
        results['rewards'].append(total_reward)
        results['lengths'].append(episode_length)
    
    # Calculate statistics
    results['mean_reward'] = float(np.mean(results['rewards']))
    results['std_reward'] = float(np.std(results['rewards']))
    results['mean_length'] = float(np.mean(results['lengths']))
    results['std_length'] = float(np.std(results['lengths']))
    
    return results


def run_evaluations(agent, envs_by_condition, logger, config):
    """Run evaluations for all conditions"""
    results = {}
    
    for condition_name, condition_data in envs_by_condition.items():
        print(f"Evaluating condition: {condition_name}")
        
        # Evaluate the agent on this condition
        condition_results = evaluate_condition(
            agent,
            condition_data['envs'],
            condition_data['eval_dir'],
            logger,
            config
        )
        
        # Store results
        results[condition_name] = condition_results
        
        # Log results
        logger.scalar(f"generalization/{condition_name}/mean_reward", condition_results['mean_reward'])
        logger.scalar(f"generalization/{condition_name}/mean_length", condition_results['mean_length'])
        logger.write()
        
        print(f"  Mean reward: {condition_results['mean_reward']:.2f} ± {condition_results['std_reward']:.2f}")
        print(f"  Mean length: {condition_results['mean_length']:.2f} ± {condition_results['std_length']:.2f}")
    
    return results


def plot_results(results, output_dir):
    """Create visualizations of the evaluation results"""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract condition names and parameters for organizing plots
    conditions = list(results.keys())
    param_types = defaultdict(list)
    
    # Group conditions by parameter type
    for condition in conditions:
        if 'arm_length' in condition:
            param_types['arm_length'].append(condition)
        elif 'joint_damping' in condition:
            param_types['joint_damping'].append(condition)
        elif 'arm_mass' in condition:
            param_types['arm_mass'].append(condition)
        else:
            param_types['other'].append(condition)
    
    # Include baseline in all parameter groups
    if 'baseline' in conditions:
        for param_type in param_types:
            if 'baseline' not in param_types[param_type]:
                param_types[param_type].append('baseline')
    
    # Create plots for each parameter type
    for param_type, param_conditions in param_types.items():
        # Skip if only one condition (baseline)
        if len(param_conditions) <= 1:
            continue
            
        # Sort conditions
        param_conditions.sort()
        if 'baseline' in param_conditions:
            # Move baseline to front
            param_conditions.remove('baseline')
            param_conditions.insert(0, 'baseline')
        
        # Extract values for plotting
        mean_rewards = [results[cond]['mean_reward'] for cond in param_conditions]
        std_rewards = [results[cond]['std_reward'] for cond in param_conditions]
        mean_lengths = [results[cond]['mean_length'] for cond in param_conditions]
        std_lengths = [results[cond]['std_length'] for cond in param_conditions]
        
        # Create reward plot
        plt.figure(figsize=(10, 6))
        plt.bar(param_conditions, mean_rewards, yerr=std_rewards, capsize=5)
        plt.xlabel('Condition')
        plt.ylabel('Mean Reward')
        plt.title(f'Generalization Performance: {param_type.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f"generalization_rewards_{param_type}.png")
        plt.close()
        
        # Create episode length plot
        plt.figure(figsize=(10, 6))
        plt.bar(param_conditions, mean_lengths, yerr=std_lengths, capsize=5)
        plt.xlabel('Condition')
        plt.ylabel('Mean Episode Length')
        plt.title(f'Episode Length: {param_type.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f"generalization_lengths_{param_type}.png")
        plt.close()
    
    # Create a combined visualization
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(conditions))
    width = 0.35
    
    # Normalize rewards and lengths for combined visualization
    max_reward = max(results[cond]['mean_reward'] for cond in conditions)
    max_length = max(results[cond]['mean_length'] for cond in conditions)
    
    norm_rewards = [results[cond]['mean_reward'] / max_reward for cond in conditions]
    norm_lengths = [results[cond]['mean_length'] / max_length for cond in conditions]
    
    plt.bar(x - width/2, norm_rewards, width, label='Normalized Reward')
    plt.bar(x + width/2, norm_lengths, width, label='Normalized Length')
    
    plt.xlabel('Condition')
    plt.ylabel('Normalized Value')
    plt.title('Generalization Performance Across All Conditions')
    plt.xticks(x, conditions, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "generalization_combined.png")
    plt.close()
    
    # Save results as JSON
    with open(output_dir / "generalization_results.json", "w") as f:
        json.dump(results, f, indent=2)


def define_test_conditions():
    """Define different test conditions for generalization evaluation"""
    conditions = {
        'baseline': {},  # No modifications
        
        # Arm length variations
        'arm_length_0.8': {'arm_length_scale': 0.8},
        'arm_length_1.2': {'arm_length_scale': 1.2},
        'arm_length_1.5': {'arm_length_scale': 1.5},
        
        # Joint damping variations
        'joint_damping_0.5': {'joint_damping_scale': 0.5},
        'joint_damping_2.0': {'joint_damping_scale': 2.0},
        'joint_damping_5.0': {'joint_damping_scale': 5.0},
        
        # Arm mass variations
        'arm_mass_0.5': {'arm_mass_scale': 0.5},
        'arm_mass_2.0': {'arm_mass_scale': 2.0},
        'arm_mass_5.0': {'arm_mass_scale': 5.0},
    }
    
    return conditions


def main():
    args = parse_args()
    
    # Set random seed
    tools.set_seed_everywhere(args.seed)
    
    # Load configuration
    config = load_config(args.logdir, args.config_path)
    config['seed'] = args.seed
    config['device'] = args.device
    
    # Override config with command line arguments
    config['task'] = args.task
    config['batch_size'] = args.batch_size
    config['action_repeat'] = args.action_repeat
    config['size'] = args.size
    config['time_limit'] = args.time_limit
    
    # Define test conditions
    test_conditions = define_test_conditions()
    
    # Set up environments for each condition
    envs_by_condition = setup_environments(args, config, test_conditions)
    
    # Get observation and action spaces from the first environment
    first_env = next(iter(envs_by_condition.values()))['envs'][0]
    obs_space = first_env.observation_space
    act_space = first_env.action_space
    
    # Load agent
    agent, logger = load_agent(args, config, obs_space, act_space)
    
    # Run evaluations
    results = run_evaluations(agent, envs_by_condition, logger, config)
    
    # Plot and save results
    plot_dir = pathlib.Path(args.logdir) / "eval_generalization" / "plots"
    plot_results(results, plot_dir)
    
    print("Generalization evaluation complete. Results saved to:", plot_dir)
    
    # Clean up
    for condition_data in envs_by_condition.values():
        for env in condition_data['envs']:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()