# import gym
# import numpy as np
# import ruamel.yaml as yaml


# class DeepMindControl:
#     metadata = {}

#     def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0, modify=None):
#         domain, task = name.split("_", 1)
#         if domain == "cup":  # Only domain with multiple words.
#             domain = "ball_in_cup"
            
#         if isinstance(domain, str):
#             from dm_control import suite

#             self._env = suite.load(
#                 domain,
#                 task,
#                 task_kwargs={"random": seed},
#             )
            
#             # todo
#             # Change gravity
#             if modify[0]:
#                 self._env.physics.model.opt.gravity[:] = [0, 0, modify[1]]  # Stronger gravity

#             # # Scale all body masses
#             # mass_scale = 1.2  # Increase all masses by 20%
#             # self._env.physics.model.body_mass[:] *= mass_scale

#             # # Reduce friction for every geom (more slippery environment)
#             # for geom_id in range(self._env.physics.model.ngeom):
#             #     self._env.physics.model.geom_friction[geom_id] = [0.1, 0.01, 0.001]

#             # # Limit actuator strength (makes the humanoid weaker)
#             # self._env.physics.model.actuator_gainprm[:, 0] *= 0.8

#         else:
#             assert task is None
#             self._env = domain()
#         self._action_repeat = action_repeat
#         self._size = tuple(size) if isinstance(size, list) else size
#         if camera is None:
#             camera = dict(quadruped=2).get(domain, 0)
#         self._camera = camera
#         self.reward_range = [-np.inf, np.inf]
        
        
# import gym
# import numpy as np
# import ruamel.yaml as yaml
# import xml.etree.ElementTree as ET


# class DeepMindControl:
#     metadata = {}
#     def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0, modify=None):
#         # Store the full name for use in property extraction
#         self._name = name
        
#         domain, task = name.split("_", 1)
#         if domain == "cup":  # Only domain with multiple words.
#             domain = "ball_in_cup"
                
#         if isinstance(domain, str):
#             from dm_control import suite

#             self._env = suite.load(
#                 domain,
#                 task,
#                 task_kwargs={"random": seed},
#             )
                
#             # Apply arm length modification if specified
#             if modify and modify[0] and modify[1] is not None and domain == "reacher":
#                 arm_scale = modify[1]  # Scaling factor for arm lengths
#                 physics = self._env.physics
                
#                 # Find arm and hand geoms by their dimensions
#                 arm_id = None
#                 hand_id = None
#                 for i in range(len(physics.model.geom_size)):
#                     size = physics.model.geom_size[i]
#                     if size[1] == 0.06 and size[0] == 0.01:  # arm has y=0.06
#                         arm_id = i
#                     elif size[1] == 0.05 and size[0] == 0.01:  # hand has y=0.05
#                         hand_id = i
                
#                 # Directly modify the arm lengths in the model
#                 if arm_id is not None:
#                     # Store original lengths for reference
#                     self._original_arm_length = physics.model.geom_size[arm_id][1]
#                     # Modify the arm length by scaling it
#                     physics.model.geom_size[arm_id][1] = self._original_arm_length * arm_scale
                
#                 if hand_id is not None:
#                     # Store original lengths for reference
#                     self._original_hand_length = physics.model.geom_size[hand_id][1]
#                     # Modify the hand length by scaling it
#                     physics.model.geom_size[hand_id][1] = self._original_hand_length * arm_scale
                
#                 # Log the modification if successful
#                 if arm_id is not None and hand_id is not None:
#                     print(f"Modified Reacher arm lengths with scale factor {arm_scale}:")
#                     print(f"  Arm: {self._original_arm_length} -> {physics.model.geom_size[arm_id][1]}")
#                     print(f"  Hand: {self._original_hand_length} -> {physics.model.geom_size[hand_id][1]}")
#                 else:
#                     print("Warning: Could not find arm/hand geoms to modify lengths.")

#         else:
#             assert task is None
#             self._env = domain()
        
#         self._action_repeat = action_repeat
#         self._size = tuple(size) if isinstance(size, list) else size
#         if camera is None:
#             camera = dict(quadruped=2).get(domain, 0)
#         self._camera = camera
#         self.reward_range = [-np.inf, np.inf]
        
#         # Initialize cache for physical properties
#         self._finger_id = None
#         self._target_id = None
#         self._warned_about_arm_lengths = False
        
#     # def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0, modify=None):
#     #     domain, task = name.split("_", 1)
#     #     if domain == "cup":  # Only domain with multiple words.
#     #         domain = "ball_in_cup"
            
#     #     if isinstance(domain, str):
#     #         from dm_control import suite
#     #         from dm_control import mujoco

#     #         self._env = suite.load(
#     #             domain,
#     #             task,
#     #             task_kwargs={"random": seed},
#     #         )
            
#     #         # Apply arm length modification if specified
#     #         if modify and modify[0] and modify[1] is not None:
#     #             arm_scale = modify[1]  # Now using the first parameter for arm length
#     #             physics = self._env.physics
#     #             if domain in ["reacher", "jaco", "finger"]:
#     #                 # Get the XML string representation of the model
#     #                 xml_string = physics.model.get_xml()
                    
#     #                 # Parse XML
#     #                 root = ET.fromstring(xml_string)
                    
#     #                 # Find and modify arm dimensions
#     #                 if domain == "reacher":
#     #                     # Modify limb geoms in reacher
#     #                     for body in root.findall(".//body"):
#     #                         if "arm" in body.get("name", ""):
#     #                             for geom in body.findall("./geom"):
#     #                                 # Get fromto for limb geoms
#     #                                 fromto = geom.get("fromto")
#     #                                 if fromto:
#     #                                     values = [float(v) for v in fromto.split()]
#     #                                     # Calculate the direction vector
#     #                                     direction = [values[3] - values[0], values[4] - values[1], values[5] - values[2]]
#     #                                     # Scale the direction vector
#     #                                     new_end = [
#     #                                         values[0] + direction[0] * arm_scale,
#     #                                         values[1] + direction[1] * arm_scale,
#     #                                         values[2] + direction[2] * arm_scale
#     #                                     ]
#     #                                     # Set new fromto
#     #                                     new_fromto = f"{values[0]} {values[1]} {values[2]} {new_end[0]} {new_end[1]} {new_end[2]}"
#     #                                     geom.set("fromto", new_fromto)
                    
#     #                 # Add similar modification logic for other domains as needed
                    
#     #                 # Convert modified XML back to string
#     #                 new_xml = ET.tostring(root, encoding='unicode')
                    
#     #                 # Reload physics with modified XML
#     #                 new_physics = mujoco.Physics.from_xml_string(new_xml)
#     #                 self._env.physics = new_physics

#     #     else:
#     #         assert task is None
#     #         self._env = domain()
#     #     self._action_repeat = action_repeat
#     #     self._size = tuple(size) if isinstance(size, list) else size
#     #     if camera is None:
#     #         camera = dict(quadruped=2).get(domain, 0)
#     #     self._camera = camera
#     #     self.reward_range = [-np.inf, np.inf]

#     # Rest of the class remains the same

#     @property
#     def observation_space(self):
#         spaces = {}
#         for key, value in self._env.observation_spec().items():
#             if len(value.shape) == 0:
#                 shape = (1,)
#             else:
#                 shape = value.shape
#             spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
#         spaces["image"] = gym.spaces.Box(0, 255, tuple(self._size) + (3,), dtype=np.uint8)
#         return gym.spaces.Dict(spaces)

#     @property
#     def action_space(self):
#         spec = self._env.action_spec()
#         return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

#     # def step(self, action):
#     #     assert np.isfinite(action).all(), action
#     #     reward = 0
#     #     for _ in range(self._action_repeat):
#     #         time_step = self._env.step(action)
#     #         reward += time_step.reward or 0
#     #         if time_step.last():
#     #             break
#     #     obs = dict(time_step.observation)
#     #     obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
#     #     obs["image"] = self.render()
#     #     # There is no terminal state in DMC
#     #     obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
#     #     obs["is_first"] = time_step.first()
#     #     done = time_step.last()
#     #     info = {"discount": np.array(time_step.discount, np.float32)}
#     #     return obs, reward, done, info
    
#     def step(self, action):
#         assert np.isfinite(action).all(), action
#         reward = 0
#         for _ in range(self._action_repeat):
#             time_step = self._env.step(action)
#             reward += time_step.reward or 0
#             if time_step.last():
#                 break
#         obs = dict(time_step.observation)
#         obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
#         obs["image"] = self.render()
        
#         # Extract physical properties for causal model
#         domain, _ = self._name.split("_", 1) if hasattr(self, '_name') else ("unknown", "")
        
#         if domain == "reacher":
#             physics = self._env.physics
            
#             # Find geom IDs for finger (end effector) and target if not already cached
#             if not hasattr(self, '_finger_id') or not hasattr(self, '_target_id'):
#                 self._finger_id = None
#                 self._target_id = None
                
#                 # Search for the specific geoms based on their sizes
#                 for i in range(len(physics.model.geom_size)):
#                     size = physics.model.geom_size[i]
#                     # Finger has a specific size in the Reacher environment
#                     if size[1] == 0 and size[0] == 0.01:  # finger typically has y=0, x=0.01
#                         self._finger_id = i
#                     # Target also has a specific size
#                     elif size[0] == 0.05 and size[1] == 0:  # target typically has x=0.05, y=0
#                         self._target_id = i
            
#             # Add physical properties to observation
#             if self._finger_id is not None:
#                 obs['end_effector_pos'] = physics.data.geom_xpos[self._finger_id][:2].copy()
            
#             if self._target_id is not None:
#                 obs['target_pos'] = physics.data.geom_xpos[self._target_id][:2].copy()
            
#             # Find arm and hand by their dimensions
#             arm_id = None
#             hand_id = None
#             for i in range(len(physics.model.geom_size)):
#                 size = physics.model.geom_size[i]
#                 if size[1] == 0.06 and size[0] == 0.01:  # arm has y=0.06
#                     arm_id = i
#                 elif size[1] == 0.05 and size[0] == 0.01:  # hand has y=0.05
#                     hand_id = i

#             # Extract the arm lengths directly from the model
#             arm_lengths = np.zeros(2, dtype=np.float32)
#             if arm_id is not None:
#                 arm_lengths[0] = physics.model.geom_size[arm_id][1]  # y dimension for arm
#             if hand_id is not None:
#                 arm_lengths[1] = physics.model.geom_size[hand_id][1]  # y dimension for hand

#             # # If we couldn't find the geoms, use default values
#             # if arm_id is None or hand_id is None:
#             #     if not hasattr(self, '_warned_about_arm_lengths'):
#             #         print("Warning: Could not find arm and hand geoms by size. Using default values.")
#             #         self._warned_about_arm_lengths = True
#             #     arm_lengths = np.array([0.06, 0.05], dtype=np.float32)

#             # Store the arm lengths in the observation
#             obs['arm_lengths'] = arm_lengths
#         # print("domain", domain)
        
#         # There is no terminal state in DMC
#         obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
#         obs["is_first"] = time_step.first()
#         done = time_step.last()
#         info = {"discount": np.array(time_step.discount, np.float32)}
#         return obs, reward, done, info

#     # def reset(self):
#     #     time_step = self._env.reset()
#     #     obs = dict(time_step.observation)
#     #     obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
#     #     obs["image"] = self.render()
#     #     obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
#     #     obs["is_first"] = time_step.first()
#     #     return obs
    
#     def reset(self):
#         time_step = self._env.reset()
#         obs = dict(time_step.observation)
#         obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
#         obs["image"] = self.render()
        
#         # Extract physical properties for causal model
#         domain, _ = self._name.split("_", 1) if hasattr(self, '_name') else ("unknown", "")
        
#         if domain == "reacher":
#             physics = self._env.physics
            
#             # Find geom IDs for finger (end effector) and target if not already cached
#             if not hasattr(self, '_finger_id') or not hasattr(self, '_target_id'):
#                 self._finger_id = None
#                 self._target_id = None
                
#                 # Search for the specific geoms based on their sizes
#                 for i in range(len(physics.model.geom_size)):
#                     size = physics.model.geom_size[i]
#                     # Finger has a specific size in the Reacher environment
#                     if size[1] == 0 and size[0] == 0.01:  # finger typically has y=0, x=0.01
#                         self._finger_id = i
#                     # Target also has a specific size
#                     elif size[0] == 0.05 and size[1] == 0:  # target typically has x=0.05, y=0
#                         self._target_id = i
#                 print("_finger_id", self._finger_id)
#                 print("_target_id", self._target_id)
            
#             # Add physical properties to observation
#             if self._finger_id is not None:
#                 obs['end_effector_pos'] = physics.data.geom_xpos[self._finger_id][:2].copy()
            
#             if self._target_id is not None:
#                 obs['target_pos'] = physics.data.geom_xpos[self._target_id][:2].copy()
            
#             # Find arm and hand by their dimensions
#             arm_id = None
#             hand_id = None
#             for i in range(len(physics.model.geom_size)):
#                 size = physics.model.geom_size[i]
#                 if size[1] == 0.06 and size[0] == 0.01:  # arm has y=0.06
#                     arm_id = i
#                 elif size[1] == 0.05 and size[0] == 0.01:  # hand has y=0.05
#                     hand_id = i

#             # Extract the arm lengths directly from the model
#             arm_lengths = np.zeros(2, dtype=np.float32)
#             if arm_id is not None:
#                 arm_lengths[0] = physics.model.geom_size[arm_id][1]  # y dimension for arm
#             if hand_id is not None:
#                 arm_lengths[1] = physics.model.geom_size[hand_id][1]  # y dimension for hand

#             # If we couldn't find the geoms, use default values
#             # if arm_id is None or hand_id is None:
#             #     if not hasattr(self, '_warned_about_arm_lengths'):
#             #         print("Warning: Could not find arm and hand geoms by size. Using default values.")
#             #         self._warned_about_arm_lengths = True
#             #     arm_lengths = np.array([0.06, 0.05], dtype=np.float32)

#             # Store the arm lengths in the observation
#             obs['arm_lengths'] = arm_lengths
        
#         obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
#         obs["is_first"] = time_step.first()
#         return obs

#     def render(self, *args, **kwargs):
#         if kwargs.get("mode", "rgb_array") != "rgb_array":
#             raise ValueError("Only render mode 'rgb_array' is supported.")
#         return self._env.physics.render(*self._size, camera_id=self._camera)
import gym
import numpy as np
import ruamel.yaml as yaml


class DeepMindControl:
    metadata = {}
    
    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0, modify=None):
        # Store the full name for use in property extraction
        self._name = name
        
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
                
        if isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(
                domain,
                task,
                task_kwargs={"random": seed},
            )
                
            # Apply arm length modification if specified
            if modify and modify[0] and modify[1] is not None and domain == "reacher":
                arm_scale = modify[1]  # Scaling factor for arm lengths
                physics = self._env.physics
                
                # Find arm and hand geoms by their dimensions
                arm_id = None
                hand_id = None
                for i in range(len(physics.model.geom_size)):
                    size = physics.model.geom_size[i]
                    if size[1] == 0.06 and size[0] == 0.01:  # arm has y=0.06
                        arm_id = i
                    elif size[1] == 0.05 and size[0] == 0.01:  # hand has y=0.05
                        hand_id = i
                
                # Directly modify the arm lengths in the model
                if arm_id is not None:
                    # Store original lengths for reference
                    self._original_arm_length = physics.model.geom_size[arm_id][1]
                    # Modify the arm length by scaling it
                    physics.model.geom_size[arm_id][1] = self._original_arm_length * arm_scale
                
                if hand_id is not None:
                    # Store original lengths for reference
                    self._original_hand_length = physics.model.geom_size[hand_id][1]
                    # Modify the hand length by scaling it
                    physics.model.geom_size[hand_id][1] = self._original_hand_length * arm_scale
                
                # Log the modification if successful
                if arm_id is not None and hand_id is not None:
                    print(f"Modified Reacher arm lengths with scale factor {arm_scale}:")
                    print(f"  Arm: {self._original_arm_length} -> {physics.model.geom_size[arm_id][1]}")
                    print(f"  Hand: {self._original_hand_length} -> {physics.model.geom_size[hand_id][1]}")
                else:
                    print("Warning: Could not find arm/hand geoms to modify lengths.")

        else:
            assert task is None
            self._env = domain()
        
        self._action_repeat = action_repeat
        self._size = tuple(size) if isinstance(size, list) else size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]
        
        # Initialize cache for physical properties
        self._finger_id = None
        self._target_id = None
        
        # Initialize and cache the domain type for observation space handling
        self._domain_type = domain

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        
        # Add standard image observation
        spaces["image"] = gym.spaces.Box(0, 255, tuple(self._size) + (3,), dtype=np.uint8)
        
        # Add custom physical properties based on domain type
        if self._domain_type == "reacher":
            # Add end effector and target position spaces
            spaces["end_effector_pos"] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
            spaces["target_pos"] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
            spaces["arm_lengths"] = gym.spaces.Box(0, np.inf, (2,), dtype=np.float32)
        
        # Add standard flags for all environments
        spaces["is_terminal"] = gym.spaces.Box(0, 1, (1,), dtype=np.bool_)
        spaces["is_first"] = gym.spaces.Box(0, 1, (1,), dtype=np.bool_)
        
        return gym.spaces.Dict(spaces)
    
    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    
    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        
        # Extract physical properties for causal model based on domain type
        domain_type = self._domain_type
        
        if domain_type == "reacher":
            physics = self._env.physics
            
            # Find geom IDs for finger (end effector) and target if not already cached
            if self._finger_id is None or self._target_id is None:
                
                # Search for the specific geoms based on their sizes
                for i in range(len(physics.model.geom_size)):
                    size = physics.model.geom_size[i]
                    # Finger has a specific size in the Reacher environment
                    if size[1] == 0 and size[0] == 0.01:  # finger typically has y=0, x=0.01
                        self._finger_id = i
                    # Target also has a specific size
                    elif size[0] == 0.05 and size[1] == 0:  # target typically has x=0.05, y=0
                        self._target_id = i
            
            # Add physical properties to observation
            if self._finger_id is not None:
                obs['end_effector_pos'] = physics.data.geom_xpos[self._finger_id][:2].copy()
            else:
                # Provide a fallback value to ensure consistency
                obs['end_effector_pos'] = np.zeros(2, dtype=np.float32)
            
            if self._target_id is not None:
                obs['target_pos'] = physics.data.geom_xpos[self._target_id][:2].copy()
            else:
                # Provide a fallback value to ensure consistency
                obs['target_pos'] = np.zeros(2, dtype=np.float32)
            
            # Find arm and hand by their dimensions
            arm_id = None
            hand_id = None
            for i in range(len(physics.model.geom_size)):
                size = physics.model.geom_size[i]
                if size[1] == 0.06 and size[0] == 0.01:  # arm has y=0.06
                    arm_id = i
                elif size[1] == 0.05 and size[0] == 0.01:  # hand has y=0.05
                    hand_id = i

            # Extract the arm lengths directly from the model
            arm_lengths = np.zeros(2, dtype=np.float32)
            if arm_id is not None:
                arm_lengths[0] = physics.model.geom_size[arm_id][1]  # y dimension for arm
            if hand_id is not None:
                arm_lengths[1] = physics.model.geom_size[hand_id][1]  # y dimension for hand

            # Store the arm lengths in the observation
            obs['arm_lengths'] = arm_lengths
            # print("hand_id", hand_id)
            # print("arm_id", arm_id)
            # print("target_pos", self._target_id)
            # print("end_effector_pos", self._finger_id)
        
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()

        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info
    
    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        
        # Extract physical properties for causal model based on domain type
        domain_type = self._domain_type
        
        if domain_type == "reacher":
            physics = self._env.physics
            
            # Find geom IDs for finger (end effector) and target if not already cached
            if self._finger_id is None or self._target_id is None:
                
                # Search for the specific geoms based on their sizes
                for i in range(len(physics.model.geom_size)):
                    size = physics.model.geom_size[i]
                    # Finger has a specific size in the Reacher environment
                    if size[1] == 0 and size[0] == 0.01:  # finger typically has y=0, x=0.01
                        self._finger_id = i
                    # Target also has a specific size
                    elif size[0] == 0.05 and size[1] == 0:  # target typically has x=0.05, y=0
                        self._target_id = i
            
            # Add physical properties to observation
            if self._finger_id is not None:
                obs['end_effector_pos'] = physics.data.geom_xpos[self._finger_id][:2].copy()
            else:
                # Provide a fallback value to ensure consistency
                obs['end_effector_pos'] = np.zeros(2, dtype=np.float32)
            
            if self._target_id is not None:
                obs['target_pos'] = physics.data.geom_xpos[self._target_id][:2].copy()
            else:
                # Provide a fallback value to ensure consistency
                obs['target_pos'] = np.zeros(2, dtype=np.float32)
            
            # Find arm and hand by their dimensions
            arm_id = None
            hand_id = None
            for i in range(len(physics.model.geom_size)):
                size = physics.model.geom_size[i]
                if size[1] == 0.06 and size[0] == 0.01:  # arm has y=0.06
                    arm_id = i
                elif size[1] == 0.05 and size[0] == 0.01:  # hand has y=0.05
                    hand_id = i

            # Extract the arm lengths directly from the model
            arm_lengths = np.zeros(2, dtype=np.float32)
            if arm_id is not None:
                arm_lengths[0] = physics.model.geom_size[arm_id][1]  # y dimension for arm
            if hand_id is not None:
                arm_lengths[1] = physics.model.geom_size[hand_id][1]  # y dimension for hand

            # Store the arm lengths in the observation
            obs['arm_lengths'] = arm_lengths
        
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)