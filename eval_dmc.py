import argparse
import pathlib
import sys
import ruamel.yaml as yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

# Add the dreamer directory to the path
sys.path.append(str(pathlib.Path(__file__).parent))

# Make sure to apply the patch to dmc.py before importing dreamer
def patch_dmc():
    dmc_path = pathlib.Path(__file__).parent / "envs" / "dmc.py"
    if not dmc_path.exists():
        print(f"Warning: Could not find dmc.py at {dmc_path}")
        return
        
    with open(dmc_path, 'r') as f:
        content = f.read()
    
    # Check if we need to patch the file
    if 'tuple(self._size)' not in content:
        # Patch the observation_space method
        content = content.replace(
            'spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)',
            'spaces["image"] = gym.spaces.Box(0, 255, tuple(self._size) + (3,), dtype=np.uint8)'
        )
        
        # Patch the __init__ method to ensure _size is stored as a tuple
        content = content.replace(
            'self._size = size',
            'self._size = tuple(size) if isinstance(size, list) else size'
        )
        
        # Save the patched file
        with open(dmc_path, 'w') as f:
            f.write(content)
        print(f"Patched {dmc_path} to fix the size issue")

# Apply the patch
patch_dmc()

# Now import dreamer and tools
import dreamer
import tools

def add_action_noise(self, action):
    """
    Add Gaussian noise to actions during evaluation.
    """
    noise = torch.randn_like(action) * self._config.test_noise_scale
    perturbed_action = action + noise
    
    # Clip actions if using normalized action space
    if hasattr(self._task_behavior.actor, "absmax") and self._task_behavior.actor.absmax is not None:
        perturbed_action = torch.clamp(perturbed_action, -1.0, 1.0)
    
    return perturbed_action

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True, 
                        help="Path to the trained model directory")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., dmc_walker_walk)")
    parser.add_argument("--noise_scale", type=float, default=0.1,
                        help="Scale of noise to add to actions")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to evaluate")
    parser.add_argument("--record_video", action="store_true",
                        help="Record videos of the evaluation episodes")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run evaluation on")
    parser.add_argument("--configs", nargs="+", 
                        help="Configuration presets to use from configs.yaml")
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    base_logdir = pathlib.Path(args.logdir)
    
    # Load configuration presets from configs.yaml
    configs_path = pathlib.Path(__file__).parent / "configs.yaml"
    with open(configs_path, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    # Recursive update function for configs
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value
    
    # Start with defaults
    config = {}
    recursive_update(config, all_configs["defaults"])
    
    # Add preset configs if specified
    if args.configs:
        for preset in args.configs:
            if preset in all_configs:
                recursive_update(config, all_configs[preset])
            else:
                print(f"Warning: Config preset '{preset}' not found in configs.yaml")
    
    # If there's a task-specific config (like dmc_vision for dmc tasks)
    task_type = args.task.split("_")[0]  # e.g., "dmc" from "dmc_walker_walk"
    if task_type in all_configs:
        recursive_update(config, all_configs[task_type])
    
    # Create evaluation directory
    eval_dir = base_logdir / f"eval_noise_{args.noise_scale}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard writer
    tb_dir = eval_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    # Create dirs for episodes
    train_eps_dir = eval_dir / "train_eps"
    train_eps_dir.mkdir(parents=True, exist_ok=True)
    eval_eps_dir = eval_dir / "eval_eps"
    eval_eps_dir.mkdir(parents=True, exist_ok=True)
    
    # Make sure size is a tuple, not a list
    if 'size' in config and isinstance(config['size'], list):
        config['size'] = tuple(config['size'])
    
    # Override specific settings for evaluation
    config.update({
        "task": args.task,
        "logdir": str(eval_dir),
        "traindir": train_eps_dir,
        "evaldir": eval_eps_dir,
        "test_noise_scale": args.noise_scale,
        "eval_episode_num": args.episodes,
        "device": args.device,
        "prefill": 0,       # No prefilling for evaluation
        "steps": 0,         # No training steps
        "expl_until": 0,    # No exploration
        "video_pred_log": args.record_video,  # Enable video prediction logging
    })
    
    # Check if there's a model checkpoint to load
    checkpoint_path = base_logdir / "latest.pt"
    if not checkpoint_path.exists():
        print(f"Warning: No checkpoint found at {checkpoint_path}")
    print(f"Loading checkpoint from {checkpoint_path}")

    
    # Save the config for reference
    with open(eval_dir / "eval_config.yaml", 'w') as f:
        yaml.dump({k: str(v) if isinstance(v, pathlib.Path) else v for k, v in config.items()}, f)
    
    # Create a namespace object from the dictionary
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    config_args = Args(**config)
    
    # Patch the Dreamer class to add the action noise method
    dreamer.Dreamer.add_action_noise = add_action_noise
    
    # Store original methods for patching
    original_policy = dreamer.Dreamer._policy
    
    def patched_policy(self, obs, state, training):
        policy_output, state = original_policy(self, obs, state, training)
        
        # Add noise to the action during evaluation
        if not training and hasattr(self._config, 'test_noise_scale') and self._config.test_noise_scale > 0:
            policy_output["action"] = self.add_action_noise(policy_output["action"])
            # We don't need to recompute logprob as it's not used during evaluation
        
        return policy_output, state
    
    # Modify the logger write function to also log to our tensorboard
    original_logger_write = tools.Logger.write
    
    def patched_logger_write(self, fps=False, step=False):
        # Call the original write method
        original_logger_write(self, fps, step)
        
        # Also log to our tensorboard
        for name, values in self._scalars.items():
            if isinstance(values, list):
                for value in values:
                    writer.add_scalar(name, value, self.step)
            else:
                writer.add_scalar(name, values, self.step)
        
        # Log videos if available
        for name, value in self._videos.items():
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            writer.add_video(name, value, self.step, fps=16)
    
    # Apply the patches
    dreamer.Dreamer._policy = patched_policy
    tools.Logger.write = patched_logger_write
    
    print(f"Evaluating model with noise scale {args.noise_scale}")
    print(f"Results will be saved to {eval_dir}")
    print(f"Tensorboard logs will be in {tb_dir}")
    
    # Run the evaluation
    dreamer.main(config_args)
    
    # Calculate and log overall statistics to tensorboard
    returns = []
    lengths = []
    for filename in eval_eps_dir.glob("*.npz"):
        try:
            episode = np.load(filename)
            returns.append(float(np.sum(episode["reward"])))
            lengths.append(len(episode["reward"]))
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    if returns:
        # Log overall statistics
        writer.add_scalar(f"noise_{args.noise_scale}/mean_return", np.mean(returns), 0)
        writer.add_scalar(f"noise_{args.noise_scale}/std_return", np.std(returns), 0)
        writer.add_scalar(f"noise_{args.noise_scale}/min_return", np.min(returns), 0)
        writer.add_scalar(f"noise_{args.noise_scale}/max_return", np.max(returns), 0)
        writer.add_scalar(f"noise_{args.noise_scale}/mean_length", np.mean(lengths), 0)
        
        # Log individual episode returns
        for i, ret in enumerate(returns):
            writer.add_scalar(f"noise_{args.noise_scale}/episode_returns", ret, i)
        
        # Print results
        print(f"\nResults with noise scale {args.noise_scale}:")
        print(f"Mean return: {np.mean(returns):.2f}")
        print(f"Std return: {np.std(returns):.2f}")
        print(f"Min return: {np.min(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}")
        print(f"Mean episode length: {np.mean(lengths):.2f}")
        
        # Save results to a text file
        with open(eval_dir / "results.txt", "w") as f:
            f.write(f"Results with noise scale {args.noise_scale}:\n")
            f.write(f"Mean return: {np.mean(returns):.2f}\n")
            f.write(f"Std return: {np.std(returns):.2f}\n")
            f.write(f"Min return: {np.min(returns):.2f}\n")
            f.write(f"Max return: {np.max(returns):.2f}\n")
            f.write(f"Mean episode length: {np.mean(lengths):.2f}\n")
            f.write("\nIndividual episode returns:\n")
            for i, ret in enumerate(returns):
                f.write(f"Episode {i+1}: {ret:.2f}\n")
    else:
        print("No episode data found. Check for errors in the evaluation.")
    
    # Close the writer
    writer.close()
    
    print(f"\nEvaluation complete!")
    print(f"To view tensorboard logs run: tensorboard --logdir={tb_dir}")

if __name__ == "__main__":
    main()
