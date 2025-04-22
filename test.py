# import os
# os.environ["MUJOCO_GL"] = "osmesa"

# from dm_control import suite

# # Load the environment (e.g., humanoid stand task)
# env = suite.load(domain_name="humanoid", task_name="stand")
# time_step = env.reset()
# print(time_step.observation.keys())import metaworld

# import metaworld
# import gym

# ml1 = metaworld.ML1('pick-place-v2')  # Load a single-task benchmark
# env = ml1.train_classes['pick-place-v2']()  # Create the environment instance
# print("Meta-World Environment Loaded Successfully!")
import dm_control.suite as suite
import numpy as np

# Load the Reacher-easy environment
env = suite.load(domain_name="reacher", task_name="easy")
physics = env.physics

# Reset to get initial observation
timestep = env.reset()

# Find geom IDs for finger (end effector) and target
finger_id = None
target_id = None
for i in range(len(physics.model.geom_size)):
    size = physics.model.geom_size[i]
    if size[1] == 0 and size[0] == 0.01:  # finger has y=0
        finger_id = i
    elif size[0] == 0.05 and size[1] == 0:  # target
        target_id = i

# 1. Joint Angles -> End Effector Position
joint_angles = physics.data.qpos[:2]  # The two joint angles
end_effector_pos = physics.data.geom_xpos[finger_id][:2]  # Actual end effector position

print("1. Joint Angles -> End Effector Position:")
print(f"   Joint angles: {joint_angles}")
print(f"   End effector position: {end_effector_pos}")

# 2. Joint Velocities -> End Effector Position
# To see how joint velocities affect end effector position, we need to apply velocities and step
old_state = physics.get_state()  # Save current state

# Apply velocities
physics.data.qvel[0] = 0.5  # First joint velocity
physics.data.qvel[1] = 0.3  # Second joint velocity
print("\n2. Joint Velocities -> End Effector Position:")
print(f"   Applied joint velocities: {physics.data.qvel[:2]}")

# Step the simulation
physics.step()
new_end_effector_pos = physics.data.geom_xpos[finger_id][:2]
print(f"   End effector position after velocity step: {new_end_effector_pos}")
print(f"   Change in position: {new_end_effector_pos - end_effector_pos}")

# Restore original state
physics.set_state(old_state)

# 3. Target Position
target_position = physics.data.geom_xpos[target_id][:2]
print("\n3. Target Position:")
print(f"   Target position: {target_position}")

# 4. Arm Lengths
# Find arm and hand by their dimensions
arm_id = None
hand_id = None
for i in range(len(physics.model.geom_size)):
    size = physics.model.geom_size[i]
    if size[1] == 0.06 and size[0] == 0.01:  # arm has y=0.06
        arm_id = i
    elif size[1] == 0.05 and size[0] == 0.01:  # hand has y=0.05
        hand_id = i

arm_length1 = physics.model.geom_size[arm_id][1]  # y dimension for arm
arm_length2 = physics.model.geom_size[hand_id][1]  # y dimension for hand

print("\n4. Arm Lengths:")
print(f"   Arm segment 1 length: {arm_length1}")
print(f"   Arm segment 2 length: {arm_length2}")
print(f"   Total arm length: {arm_length1 + arm_length2}")

# To see how arm lengths relate to end effector position:
# We can measure the distance from root to end effector and compare to arm length
root_pos = physics.data.geom_xpos[5][:2]  # Root geom is typically index 5
distance_to_ee = np.linalg.norm(end_effector_pos - root_pos)
print(f"   Distance from root to end effector: {distance_to_ee}")
print(f"   Maximum theoretical reach (total arm length): {arm_length1 + arm_length2}")

# 5. Joint Damping
joint_damping = physics.model.dof_damping[:2]
print("\n5. Joint Damping:")
print(f"   Joint damping values: {joint_damping}")

# To show effect of damping on velocities:
old_state = physics.get_state()
physics.data.qvel[0] = 1.0
physics.data.qvel[1] = 1.0
print(f"   Initial joint velocities: {physics.data.qvel[:2]}")
# Step forward to see damping effect
physics.step()
print(f"   Joint velocities after one step: {physics.data.qvel[:2]}")
physics.set_state(old_state)

# 6. Arm Mass/Inertia
# The arm bodies are typically indices 1 and 2 (after the world body)
arm_masses = physics.model.body_mass[1:3]
arm_inertias = physics.model.body_inertia[1:3]
print("\n6. Arm Mass/Inertia:")
print(f"   Arm masses: {arm_masses}")
print(f"   Arm inertias: {arm_inertias}")

# To show effect of mass/inertia on joint velocities:
old_state = physics.get_state()
# Apply torque by updating ctrl (this simulates force application)
physics.data.ctrl[:] = np.array([0.5, 0.5])  # Apply equal control input
print(f"   Applied control: {physics.data.ctrl}")
physics.step()
print(f"   Resulting velocities: {physics.data.qvel[:2]}")
# The heavier/higher inertia joint will have lower velocity for same torque
physics.set_state(old_state)
# 4. Arm Lengths
# Fin


# # Function to explore attributes
# def explore_attributes(obj, prefix="", max_depth=3, current_depth=0):
#     if current_depth > max_depth:
#         return
    
#     # Get all attributes
#     if hasattr(obj, "__dict__"):
#         attrs = vars(obj)
#     else:
#         try:
#             attrs = {k: getattr(obj, k) for k in dir(obj) 
#                     if not k.startswith("_") and not callable(getattr(obj, k))}
#         except:
#             return
    
#     # Print attributes
#     for name, value in attrs.items():
#         value_repr = repr(value)
#         if len(value_repr) > 100:
#             if isinstance(value, np.ndarray):
#                 value_repr = f"ndarray(shape={value.shape}, dtype={value.dtype})"
#             else:
#                 value_repr = f"{type(value).__name__}(...)"
        
#         print(f"{prefix}{name}: {value_repr}")
        
#         # Recursively explore non-primitive types
#         if (not isinstance(value, (str, int, float, bool, np.ndarray)) and 
#             not name.startswith("_") and value is not None):
#             explore_attributes(value, prefix=f"{prefix}{name}.", 
#                               max_depth=max_depth, current_depth=current_depth+1)

# # Explore physics attributes
# print("=== Physics Attributes ===")
# explore_attributes(env.physics, prefix="physics.")

# # For the named attributes specifically, which have a different structure
# print("\n=== Named Attributes ===")
# if hasattr(env.physics, "named"):
#     named = env.physics.named
#     if hasattr(named, "model"):
#         model = named.model
#         # Print all available attributes
#         print("Available named.model attributes:")
#         for attr in dir(model):
#             if not attr.startswith("_") and not callable(getattr(model, attr)):
#                 print(f"  - {attr}")
                
#         # For accessing sizes, try printing all available properties
#         if hasattr(model, "geom_size"):
#             print("\nAvailable geom_size keys:")
#             print(model.geom_size)
#             for key in model.geom_size:
#                 print(key)
#                 value = model.geom_size[key]
#                 print(f"  - {key}: {value}")
