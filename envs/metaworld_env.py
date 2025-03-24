import gym
import numpy as np
import sys
import cv2

sys.path.append('/home/richtsai1103/CRL/Metaworld')
import metaworld


class MetaWorldEnv:
    metadata = {}

    def __init__(self, name, action_repeat=2, size=(64, 64), seed=0):
        # Strip metaworld_ prefix if present
        if name.startswith("metaworld_"):
            task_name = name[len("metaworld_") :]
        else:
            task_name = name

        # Load MetaWorld task suite
        ml1 = metaworld.ML1(task_name)

        # Create environment instance
        self._env = ml1.train_classes[task_name]()
        self._env.action_space.seed(seed)
        self._env.observation_space.seed(seed)

        # Find and set the corresponding task
        self._task = next(t for t in ml1.train_tasks if t.env_name == task_name)
        self._env.set_task(self._task)

        self._action_repeat = action_repeat
        self._size = size
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
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self._action_repeat):
            obs, step_reward, done, truncated, step_info  = self._env.step(action)
            if isinstance(obs, tuple):
                obs = obs[0]
            total_reward += step_reward
            info.update(step_info)
            if done or truncated:
                break
        
        # is_terminal = done and info.get("success", 0) == 0
        is_terminal = done

        obs_dict = {
            "state": np.array(obs, dtype=np.float32),
            "image": self.render(),
            "is_terminal": is_terminal,
            "is_first": False,
        }
        return obs_dict, total_reward, done, info


    def reset(self):
        """Resets the environment and returns the initial observation."""
        obs, _  = self._env.reset()

        obs_dict = {
            "state": np.array(obs, dtype=np.float32),
            "image": self.render(),
            "is_terminal": False,  # Reset always starts fresh
            "is_first": True,  # First step of episode
        }
        return obs_dict

    def render(self, mode="rgb_array"):
        img = self._env.mujoco_renderer.render(render_mode=mode)
        img = cv2.resize(img, self._size, interpolation=cv2.INTER_AREA)
        return img




    

