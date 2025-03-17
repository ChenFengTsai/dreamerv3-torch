import gym
import numpy as np
import metaworld

class MetaWorldEnv:
    metadata = {}

    def __init__(self, action_repeat=2, size=(64, 64), seed=0, time_limit=500):
        env_name = "pick-place-v2"  # Specific task

        # Load MetaWorld task
        ml1 = metaworld.ML1(env_name)

        self._env = ml1.train_classes[env_name]()  # Create environment instance
        self._env.action_space.seed(seed)
        self._env.observation_space.seed(seed)

        # Set an initial task
        self._task = ml1.train_tasks[0]
        self._env.set_task(self._task)

        self._action_repeat = action_repeat
        self._size = size
        self.time_limit = time_limit
        self.reward_range = (-np.inf, np.inf)

    @property
    def observation_space(self):
        """Returns a Gym Dict space with state and image observations."""
        obs_dim = self._env.observation_space.shape[0]
        spaces = {
            "state": gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        """Returns the action space formatted for Gym."""
        return self._env.action_space

    def step(self, action):
        """Executes an action and returns formatted observations."""
        assert np.isfinite(action).all(), action
        total_reward = 0
        done = False
        info = {}

        for _ in range(self._action_repeat):
            obs, step_reward, done, step_info = self._env.step(action)
            total_reward += step_reward
            info.update(step_info)
            if done:
                break
        
        # Define is_terminal similar to DMC
        is_terminal = done and info.get("success", 0) == 0  # Failure if done but task not solved

        obs_dict = {
            "state": np.array(obs, dtype=np.float32),
            "image": self.render(),
            "is_terminal": is_terminal,  # Tracks unrecoverable failures
            "is_first": False,  # No explicit first-step tracking in Meta-World
        }
        return obs_dict, total_reward, done, info

    def reset(self):
        """Resets the environment and returns the initial observation."""
        obs = self._env.reset()
        obs_dict = {
            "state": np.array(obs, dtype=np.float32),
            "image": self.render(),
            "is_terminal": False,  # Reset always starts fresh
            "is_first": True,  # First step of episode
        }
        return obs_dict

    def render(self, mode="rgb_array"):
        """Returns an image observation of the current state."""
        return self._env.sim.render(width=self._size[0], height=self._size[1], camera_name="corner2")
