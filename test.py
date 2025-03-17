# import os
# os.environ["MUJOCO_GL"] = "osmesa"

# from dm_control import suite

# # Load the environment (e.g., humanoid stand task)
# env = suite.load(domain_name="humanoid", task_name="stand")
# time_step = env.reset()
# print(time_step.observation.keys())import metaworld

import metaworld
import gym

ml1 = metaworld.ML1('pick-place-v2')  # Load a single-task benchmark
env = ml1.train_classes['pick-place-v2']()  # Create the environment instance
print("Meta-World Environment Loaded Successfully!")


