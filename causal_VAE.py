import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd
import numpy as np

import networks
# Import only necessary functions from utils
from utils import (
    conditional_sample_gaussian,
    condition_prior,
    kl_normal,
    log_bernoulli_with_logits,
    gaussian_parameters,
    vector_expand
)

to_np = lambda x: x.detach().cpu().numpy()
import tools


# class ReacherPhysicalProperties(nn.Module):
#     """
#     Module to extract and process physical properties from Reacher environment
#     to be used as labels for CausalVAE.
#     """
#     def __init__(self, config, device=None):
#         super(ReacherPhysicalProperties, self).__init__()
#         self.config = config
#         self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
#         # Define dimensions for each property
#         self.joint_angles_dim = 2  # 2 joints for standard Reacher
#         self.joint_velocities_dim = 2
#         self.end_effector_pos_dim = 2  # x, y coordinates
#         self.target_pos_dim = 2  # x, y coordinates
#         self.arm_lengths_dim = 2  # length of each arm segment
#         self.joint_damping_dim = 2  # damping of each joint
#         self.arm_mass_dim = 2  # mass/inertia of each arm
        
#         # Total dimension of all properties
#         self.total_dim = (self.joint_angles_dim + self.joint_velocities_dim + 
#                          self.end_effector_pos_dim + self.target_pos_dim + 
#                          self.arm_lengths_dim + self.joint_damping_dim + 
#                          self.arm_mass_dim)
        
#         # Projection layer to map properties to label space
#         self.label_dim = config.causal_factors if hasattr(config, 'causal_factors') else 7
#         self.projection = nn.Sequential(
#             nn.Linear(self.total_dim, 64),
#             nn.ELU(),
#             nn.Linear(64, self.label_dim)
#         )
        
#         # Normalization parameters (will be updated during runtime)
#         self.register_buffer('means', torch.zeros(self.total_dim, device=self.device))
#         self.register_buffer('stds', torch.ones(self.total_dim, device=self.device))
#         self.register_buffer('initialized', torch.tensor(0, device=self.device))

#     def extract_properties(self, obs):
#         """
#         Extract relevant physical properties from observation dictionary
        
#         Args:
#             obs: Dictionary containing observation from DMC Reacher
            
#         Returns:
#             Tensor containing concatenated physical properties
#         """
#         # Extract properties from observation
#         batch_size = next(iter(obs.values())).shape[0]
        
#         # Extract joint angles - based on reference, they are in qpos[:2]
#         if 'qpos' in obs:
#             joint_angles = obs['qpos'][..., :2]
#         elif 'joints' in obs:
#             joint_angles = obs['joints']
#         else:
#             joint_angles = torch.zeros((batch_size, self.joint_angles_dim), device=self.device)
            
#         # End effector position - should be directly available in the observation
#         if 'end_effector' in obs:
#             end_effector_pos = obs['end_effector'][..., :2]  # Take only x,y coordinates
#         elif 'fingertip' in obs:
#             end_effector_pos = obs['fingertip'][..., :2]  # Take only x,y coordinates
#         elif 'geom_xpos' in obs and 'finger_id' in obs:
#             # If geom_xpos and finger_id are provided, extract directly
#             end_effector_pos = obs['geom_xpos'][..., obs['finger_id'], :2]
#         else:
#             # If not available in any form, return zeros
#             end_effector_pos = torch.zeros((batch_size, self.end_effector_pos_dim), device=self.device)
            
#         # Target position - should be directly available in the observation
#         if 'target' in obs:
#             target_pos = obs['target'][..., :2]  # Take only x,y coordinates
#         elif 'target_position' in obs:
#             target_pos = obs['target_position'][..., :2]
#         elif 'geom_xpos' in obs and 'target_id' in obs:
#             # If geom_xpos and target_id are provided, extract directly
#             target_pos = obs['geom_xpos'][..., obs['target_id'], :2]
#         else:
#             # If not available in any form, return zeros
#             target_pos = torch.zeros((batch_size, self.target_pos_dim), device=self.device)
        
#         # Get arm lengths from the environment
#         arm_lengths = self.get_arm_lengths(batch_size)
        
#         # Concatenate the 4 key properties
#         properties = torch.cat([
#             joint_angles,
#             end_effector_pos,
#             target_pos,
#             arm_lengths
#         ], dim=-1)
        
#         return properties

#     # Method removed since we're getting end effector position directly from observation

#     def get_arm_lengths(self, batch_size):
#         """
#         Returns arm lengths, based on the reference values (0.06 and 0.05)
#         """
#         if hasattr(self.config, 'arm_lengths'):
#             return torch.tensor(self.config.arm_lengths).repeat(batch_size, 1).to(self.device)
#         # Default values from the reference code
#         return torch.tensor([[0.06, 0.05]]).repeat(batch_size, 1).to(self.device)
    
#     def update_normalization(self, properties):
#         """Update running mean and std for normalization"""
#         with torch.no_grad():
#             if self.initialized == 0:
#                 self.means = properties.mean(dim=0)
#                 self.stds = properties.std(dim=0).clip(min=1e-6)
#                 self.initialized.fill_(1)
#             else:
#                 # Exponential moving average
#                 alpha = 0.05
#                 self.means = (1 - alpha) * self.means + alpha * properties.mean(dim=0)
#                 self.stds = (1 - alpha) * self.stds + alpha * properties.std(dim=0).clip(min=1e-6)
    
#     def normalize(self, properties):
#         """Normalize properties using running statistics"""
#         return (properties - self.means) / self.stds
    
#     def forward(self, obs):
#         """
#         Extract physical properties from observations and project to label space
        
#         Args:
#             obs: Dictionary containing observation from DMC Reacher
            
#         Returns:
#             Tensor containing labels for CausalVAE
#         """
#         properties = self.extract_properties(obs)
        
#         # Update normalization statistics during training
#         if self.training:
#             self.update_normalization(properties)
            
#         # Normalize properties
#         normalized_properties = self.normalize(properties)
        
#         # Project to label space
#         labels = self.projection(normalized_properties)
        
#         return labels



# class CausalVAE_WorldModel(nn.Module):
#     """
#     A causal world model for Dreamer that uses CausalVAE for encoding observations and decoding reconstructions.
#     This implements the causal structure described in the CausalVAE paper, but adapted for the DMC environment.
    
#     The causal relationships modeled are:
#     - Joint Angles -> End Effector Position
#     - Joint Velocities -> End Effector Position
#     - Target Position (independent factor)
#     - Arm Lengths -> Joint Angles, End Effector Position
#     - Joint Damping -> Joint Velocities
#     - Arm Mass/Inertia -> Joint Velocities
    
#     The model uses the original RSSM dynamics model from Dreamer but replaces the encoder and decoder
#     with causal versions that respect the above DAG structure.
#     """
#     def __init__(self, obs_space, act_space, step, config):
#         super(CausalVAE_WorldModel, self).__init__()
#         self._step = step
#         self._use_amp = True if config.precision == 16 else False
#         self._config = config
#         self.device = config.device if config is not None and hasattr(config, 'device') else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         self.physical_properties = ReacherPhysicalProperties(config, device=self.device)
        
#         # Extract observation shapes
#         shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        
#         # Create the encoder and configure embedding size
#         self.encoder = networks.MultiEncoder(shapes, **config.encoder)
#         self.embed_size = self.encoder.outdim
        
#         # Define causal structure parameters
#         self.z_dim = config.dyn_stoch  # Total latent dimension
#         self.z1_dim = getattr(config, 'causal_factors', 7)  # Number of causal factors
#         self.z2_dim = self.z_dim // self.z1_dim  # Dimension per factor
        
#         # Setup the dynamics model (RSSM) from Dreamer - unchanged
#         self.dynamics = networks.RSSM(
#             config.dyn_stoch,
#             config.dyn_deter,
#             config.dyn_hidden,
#             config.dyn_rec_depth,
#             config.dyn_discrete,
#             config.act,
#             config.norm,
#             config.dyn_mean_act,
#             config.dyn_std_act,
#             config.dyn_min_std,
#             config.unimix_ratio,
#             config.initial,
#             config.num_actions,
#             self.embed_size,
#             self.device,
#         )
        
#         # Create the causal DAG layer
#         self.dag = DagLayer(self.z1_dim, self.z1_dim, i=False, initial=True)
#         self._initialize_dag_structure()
        
#         # Create attention mechanism for causal disentanglement
#         self.attn = Attention(self.z2_dim)
        
#         # Create mask layers for manipulating latent variables
#         self.mask_z = MaskLayer(self.z_dim, concept=self.z1_dim, z2_dim=self.z2_dim)
#         self.mask_u = MaskLayer(self.z1_dim, concept=self.z1_dim, z2_dim=1)
        
#         # Create the output heads
#         self.heads = nn.ModuleDict()
#         if config.dyn_discrete:
#             feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
#         else:
#             feat_size = config.dyn_stoch + config.dyn_deter
            
#         self.heads["decoder"] = networks.MultiDecoder(
#             feat_size, shapes, **config.decoder
#         )
#         self.heads["reward"] = networks.MLP(
#             feat_size,
#             (255,) if config.reward_head["dist"] == "symlog_disc" else (),
#             config.reward_head["layers"],
#             config.units,
#             config.act,
#             config.norm,
#             dist=config.reward_head["dist"],
#             outscale=config.reward_head["outscale"],
#             device=self.device,
#             name="Reward",
#         )
#         self.heads["cont"] = networks.MLP(
#             feat_size,
#             (),
#             config.cont_head["layers"],
#             config.units,
#             config.act,
#             config.norm,
#             dist="binary",
#             outscale=config.cont_head["outscale"],
#             device=self.device,
#             name="Cont",
#         )
        
#         # Setup optimizer
#         for name in config.grad_heads:
#             assert name in self.heads, name
#         self._model_opt = tools.Optimizer(
#             "model",
#             self.parameters(),
#             config.model_lr,
#             config.opt_eps,
#             config.grad_clip,
#             config.weight_decay,
#             opt=config.opt,
#             use_amp=self._use_amp,
#         )
#         print(
#             f"CausalVAE model_opt has {sum(param.numel() for param in self.parameters())} variables."
#         )
        
#         # Set up loss scaling factors
#         self._scales = dict(
#             reward=config.reward_head["loss_scale"],
#             cont=config.cont_head["loss_scale"],
#         )
        
#         # Define scaling factors for causal structure
#         self.scale = torch.zeros((self.z1_dim, 2), device=self.device)
#         self._initialize_scale()
        
#     def _initialize_dag_structure(self):
#         """
#         Initialize the DAG structure for DMC environment based on the causal relations:
        
#         0: Joint Angles -> 2: End Effector Position
#         1: Joint Velocities -> 2: End Effector Position
#         3: Target Position (independent factor)
#         4: Arm Lengths -> 0: Joint Angles, 2: End Effector Position
#         5: Joint Damping -> 1: Joint Velocities
#         6: Arm Mass/Inertia -> 1: Joint Velocities
#         """
#         # Reset existing DAG structure
#         self.dag.A.data = torch.zeros_like(self.dag.A.data)
        
#         # Set causal relationships based on the number of factors we have
#         if self.z1_dim >= 3:
#             # 0: Joint Angles -> 2: End Effector Position
#             self.dag.A.data[0, 2] = 1.0
            
#             # 1: Joint Velocities -> 2: End Effector Position
#             self.dag.A.data[1, 2] = 1.0
        
#         if self.z1_dim >= 5:
#             # 4: Arm Lengths -> 0: Joint Angles
#             self.dag.A.data[4, 0] = 1.0
#             # 4: Arm Lengths -> 2: End Effector Position
#             self.dag.A.data[4, 2] = 1.0
        
#         if self.z1_dim >= 7:
#             # 5: Joint Damping -> 1: Joint Velocities
#             self.dag.A.data[5, 1] = 1.0
#             # 6: Arm Mass/Inertia -> 1: Joint Velocities
#             self.dag.A.data[6, 1] = 1.0
            
#         print(f"Initialized DAG structure for {self.z1_dim} causal factors in CausalVAE")
        
#     def _initialize_scale(self):
#         """Initialize scale for the causal variables based on DMC environment"""
#         # Approximate ranges for various factors in the DMC environment
#         if self.z1_dim >= 1:
#             self.scale[0] = torch.tensor([0, 6.28])  # Joint angles (0 to 2π)
#         if self.z1_dim >= 2:
#             self.scale[1] = torch.tensor([-10, 10])  # Joint velocities
#         if self.z1_dim >= 3:
#             self.scale[2] = torch.tensor([-1, 1])  # End effector position
#         if self.z1_dim >= 4:
#             self.scale[3] = torch.tensor([-1, 1])  # Target position
#         if self.z1_dim >= 5:
#             self.scale[4] = torch.tensor([0.5, 1.5])  # Arm lengths
#         if self.z1_dim >= 6:
#             self.scale[5] = torch.tensor([0.1, 2.0])  # Joint damping
#         if self.z1_dim >= 7:
#             self.scale[6] = torch.tensor([0.5, 2.0])  # Arm mass/inertia
            
#     def causal_encode(self, embed, label=None):
#         """
#         Apply the causal encoding process to embedded observations.
        
#         Args:
#             embed: Embedded observation from encoder
#             label: Optional labels for supervised training
            
#         Returns:
#             Causally encoded representation
#         """
#         batch_size = embed.size(0)
        
#         # If no labels provided, use zeros (unsupervised mode)
#         if label is None:
#             label = torch.zeros(batch_size, self.z1_dim, device=self.device)
        
#         # Apply encoder to get independent latent factors (q_m and q_v are mean and variance)
#         q_m, q_v = torch.split(self.dynamics._suff_stats_layer("obs", embed), [self.z_dim, self.z_dim], -1)
        
#         # Reshape for causal structure processing
#         q_m = q_m.reshape([batch_size, self.z1_dim, self.z2_dim])
#         q_v = torch.ones(batch_size, self.z1_dim, self.z2_dim).to(self.device)
        
#         # Apply causal DAG to transform independent variables to causally related ones
#         decode_m, decode_v = self.dag.calculate_dag(q_m.to(self.device), q_v.to(self.device))
#         decode_m = decode_m.reshape([batch_size, self.z1_dim, self.z2_dim])
#         decode_v = decode_v.reshape([batch_size, self.z1_dim, self.z2_dim])
        
#         # Apply masking operations for the SCM
#         m_zm = self.dag.mask_z(decode_m.to(self.device)).reshape([batch_size, self.z1_dim, self.z2_dim])
#         m_u = self.dag.mask_u(label.to(self.device))
        
#         # Mix masked variables
#         f_z = self.mask_z.mix(m_zm).reshape([batch_size, self.z1_dim, self.z2_dim]).to(self.device)
        
#         # Apply attention
#         e_tilde, _ = self.attn.attention(
#             decode_m.reshape([batch_size, self.z1_dim, self.z2_dim]).to(self.device),
#             q_m.reshape([batch_size, self.z1_dim, self.z2_dim]).to(self.device)
#         )
        
#         # Generate final causal representation
#         f_z1 = f_z + e_tilde
        
#         # Sample from the causal model
#         z_given_dag = conditional_sample_gaussian(f_z1, decode_v * 0.001)
        
#         # Reshape to flat vector for RSSM
#         z_flat = z_given_dag.reshape([batch_size, self.z_dim])
        
#         return z_flat
    
#     def _train(self, data):
#         """
#         Training function for CausalVAE world model
        
#         Args:
#             data: Batch of experience data
            
#         Returns:
#             Updated model parameters and metrics
#         """
#         # Preprocess data
#         data = self.preprocess(data)
        
#         with tools.RequiresGrad(self):
#             with torch.cuda.amp.autocast(self._use_amp):
#                 # Encode observations
#                 embed = self.encoder(data)
                
#                 # Apply causal encoding if configured
#                 if hasattr(self._config, 'use_causal_encode') and self._config.use_causal_encode:
#                     # Extract labels from data if available
#                     label = None
#                     if hasattr(data, 'labels') and data.labels is not None:
#                         label = data['labels']
#                     embed = self.causal_encode(embed, label)
                
#                 # Use standard RSSM dynamics for temporal modeling
#                 post, prior = self.dynamics.observe(
#                     embed, data["action"], data["is_first"]
#                 )
                
#                 # Compute KL divergence loss
#                 kl_free = self._config.kl_free
#                 dyn_scale = self._config.dyn_scale
#                 rep_scale = self._config.rep_scale
#                 kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
#                     post, prior, kl_free, dyn_scale, rep_scale
#                 )
                
#                 # Get features and generate predictions
#                 preds = {}
#                 for name, head in self.heads.items():
#                     grad_head = name in self._config.grad_heads
#                     feat = self.dynamics.get_feat(post)
#                     feat = feat if grad_head else feat.detach()
#                     pred = head(feat)
#                     if type(pred) is dict:
#                         preds.update(pred)
#                     else:
#                         preds[name] = pred
                
#                 # Compute losses for all predictions
#                 losses = {}
#                 for name, pred in preds.items():
#                     loss = -pred.log_prob(data[name])
#                     assert loss.shape == embed.shape[:2], (name, loss.shape)
#                     losses[name] = loss
                
#                 # Add causal DAG loss if enabled
#                 if hasattr(self._config, 'dag_loss_weight') and self._config.dag_loss_weight > 0:
#                     h_A = self._compute_dag_constraint()
#                     losses["dag"] = h_A * self._config.dag_loss_weight
                
#                 # Scale losses and combine
#                 scaled = {
#                     key: value * self._scales.get(key, 1.0)
#                     for key, value in losses.items()
#                 }
#                 model_loss = sum(scaled.values()) + kl_loss
                
#                 # Optimize model
#                 metrics = self._model_opt(torch.mean(model_loss), self.parameters())
        
#         # Track metrics
#         metrics.update({f"{name}_loss": tools.to_np(loss) for name, loss in losses.items()})
#         metrics["kl_free"] = kl_free
#         metrics["dyn_scale"] = dyn_scale
#         metrics["rep_scale"] = rep_scale
#         metrics["dyn_loss"] = tools.to_np(dyn_loss)
#         metrics["rep_loss"] = tools.to_np(rep_loss)
#         metrics["kl"] = tools.to_np(torch.mean(kl_value))
        
#         with torch.cuda.amp.autocast(self._use_amp):
#             metrics["prior_ent"] = tools.to_np(
#                 torch.mean(self.dynamics.get_dist(prior).entropy())
#             )
#             metrics["post_ent"] = tools.to_np(
#                 torch.mean(self.dynamics.get_dist(post).entropy())
#             )
#             context = dict(
#                 embed=embed,
#                 feat=self.dynamics.get_feat(post),
#                 kl=kl_value,
#                 postent=self.dynamics.get_dist(post).entropy(),
#             )
        
#         post = {k: v.detach() for k, v in post.items()}
#         return post, context, metrics
    
#     def _compute_dag_constraint(self):
#         """Compute the DAG constraint for the adjacency matrix"""
#         # Use the DAGness constraint from Yu et al. (DAG-GNN)
#         # h(A) = tr((I + A○A/d)^d) - d
#         d = self.z1_dim
#         A = self.dag.A
#         M = torch.eye(d, device=A.device) + A * A / d
#         h_A = torch.trace(torch.matrix_power(M, d)) - d
#         return h_A
    
#     def preprocess(self, obs):
#         """
#         Preprocess observations
        
#         Args:
#             obs: Raw observations
            
#         Returns:
#             Preprocessed observations
#         """
#         obs = {
#             k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
#             for k, v in obs.items()
#         }
#         obs["image"] = obs["image"] / 255.0
#         if "discount" in obs:
#             obs["discount"] *= self._config.discount
#             obs["discount"] = obs["discount"].unsqueeze(-1)
        
#         assert "is_first" in obs
#         assert "is_terminal" in obs
#         obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
#         return obs
    
#     def video_pred(self, data):
#         """
#         Generate video predictions for visualization
        
#         Args:
#             data: Batch of data
            
#         Returns:
#             Video prediction tensor
#         """
#         data = self.preprocess(data)
#         embed = self.encoder(data)
        
#         states, _ = self.dynamics.observe(
#             embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
#         )
#         recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[:6]
#         reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        
#         init = {k: v[:, -1] for k, v in states.items()}
#         prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
#         openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        
#         model = torch.cat([recon[:, :5], openl], 1)
#         truth = data["image"][:6]
#         error = (model - truth + 1.0) / 2.0
        
#         return torch.cat([truth, model, error], 2)
    
#     def intervene(self, post, intervention_idx, intervention_value=None):
#         """
#         Perform causal intervention on latent variables
        
#         Args:
#             post: Current posterior latent state
#             intervention_idx: Index of causal factor to intervene on
#             intervention_value: Value to set for intervention (None = random value)
            
#         Returns:
#             Modified latent state after intervention
#         """
#         # Create copy of posterior to modify
#         post_modified = {k: v.clone() for k, v in post.items()}
        
#         # Extract stochastic latent variables
#         stoch = post_modified["stoch"]
#         batch_size = stoch.size(0)
        
#         # Reshape to causal structure
#         z = stoch.reshape(batch_size, self.z1_dim, self.z2_dim)
        
#         # If intervention value not provided, sample a random value
#         if intervention_value is None:
#             # Sample random value within factor's scale range
#             low, high = self.scale[intervention_idx]
#             intervention_value = torch.rand(batch_size, self.z2_dim, device=self.device) * (high - low) + low
#         else:
#             # Ensure correct shape
#             intervention_value = intervention_value.reshape(batch_size, self.z2_dim)
        
#         # Apply intervention by setting the value
#         z[:, intervention_idx, :] = intervention_value
        
#         # Reshape back to flat vector
#         post_modified["stoch"] = z.reshape(batch_size, self.z_dim)
        
#         # If using mean for evaluation, update it too
#         if "mean" in post_modified:
#             mean = post_modified["mean"].reshape(batch_size, self.z1_dim, self.z2_dim)
#             mean[:, intervention_idx, :] = intervention_value
#             post_modified["mean"] = mean.reshape(batch_size, self.z_dim)
        
#         return post_modified


class ReacherPhysicalProperties(nn.Module):
    """
    Module to extract and process physical properties from Reacher environment
    to be used as labels for CausalVAE.
    """
    def __init__(self, config, device=None):
        super(ReacherPhysicalProperties, self).__init__()
        self.config = config
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Define dimensions for each property - keeping only 4 important factors
        self.joint_angles_dim = 2  # 2 joints for standard Reacher
        self.end_effector_pos_dim = 2  # x, y coordinates
        self.target_pos_dim = 2  # x, y coordinates
        self.arm_lengths_dim = 2  # length of each arm segment
        
        # Total dimension of all properties
        self.total_dim = (self.joint_angles_dim + self.end_effector_pos_dim + 
                         self.target_pos_dim + self.arm_lengths_dim)
        
        # Projection layer to map properties to label space
        self.label_dim = config.causal_factors if hasattr(config, 'causal_factors') else 4
        self.projection = nn.Sequential(
            nn.Linear(self.total_dim, 64),
            nn.ELU(),
            nn.Linear(64, self.label_dim)
        )
        
        # Normalization parameters (will be updated during runtime)
        self.register_buffer('means', torch.zeros(self.total_dim, device=self.device))
        self.register_buffer('stds', torch.ones(self.total_dim, device=self.device))
        self.register_buffer('initialized', torch.tensor(0, device=self.device))
        
    def extract_properties(self, obs):
        """
        Extract relevant physical properties directly from observation dictionary
        that now contains properties extracted from the physics object.
        
        Args:
            obs: Dictionary containing enhanced observations from DMC Reacher
            
        Returns:
            Tensor containing concatenated physical properties
        """
        # Get batch size and potentially sequence length
        keys = list(obs.keys())
        if len(keys) == 0:
            raise ValueError("Empty observation dictionary")
        
        # Determine shape of the observations
        first_value = obs[keys[0]]
        batch_size = first_value.shape[0]
        
        # Handle both batch and batch+sequence dimensions
        if len(first_value.shape) >= 3:  # [batch, seq, ...]
            batch_seq_shape = first_value.shape[:2]
            is_sequence = True
        else:
            batch_seq_shape = (batch_size,)
            is_sequence = False

        device = self.device
        
        # 1. Extract joint angles (directly from positions)
        if 'positions' in obs:
            joint_angles = torch.tensor(obs['positions'], device=device)
            if joint_angles.shape[-1] > 2:
                joint_angles = joint_angles[..., :2]  # Take first two angles
        # else:
        #     # Default zeros if not found
        #     shape = batch_seq_shape + (self.joint_angles_dim,) if is_sequence else (batch_size, self.joint_angles_dim)
        #     joint_angles = torch.zeros(shape, device=device)
        
        # 2. Get end effector position directly from physics-enhanced observations
        if 'end_effector_pos' in obs:
            end_effector_pos = torch.tensor(obs['end_effector_pos'], device=device)
            if end_effector_pos.shape[-1] > 2:
                end_effector_pos = end_effector_pos[..., :2]  # Take x,y coordinates
        # else:
        #     # If not available, use a placeholder
        #     shape = batch_seq_shape + (self.end_effector_pos_dim,) if is_sequence else (batch_size, self.end_effector_pos_dim)
        #     end_effector_pos = torch.zeros(shape, device=device)
        
        # 3. Get target position directly from physics-enhanced observations
        if 'target_pos' in obs:
            target_pos = torch.tensor(obs['target_pos'], device=device)
            if target_pos.shape[-1] > 2:
                target_pos = target_pos[..., :2]  # Take x,y coordinates
        # else:
        #     # If not available, try calculating from to_target
        #     if 'to_target' in obs and 'end_effector_pos' in obs:
        #         to_target = torch.tensor(obs['to_target'], device=device)
        #         if to_target.shape[-1] > 2:
        #             to_target = to_target[..., :2]
        #         target_pos = end_effector_pos + to_target
        #     else:
        #         # Last resort: use placeholder
        #         shape = batch_seq_shape + (self.target_pos_dim,) if is_sequence else (batch_size, self.target_pos_dim)
        #         target_pos = torch.zeros(shape, device=device)
        
        # 4. Get arm lengths directly from physics-enhanced observations
        if 'arm_lengths' in obs:
            arm_lengths = torch.tensor(obs['arm_lengths'], device=device)
            # If arm_lengths is 1D but we need 2D
            if arm_lengths.dim() == 1:
                arm_lengths = arm_lengths.unsqueeze(0).expand(batch_size, -1)
        # else:
        #     # Use default values
        #     arm_lengths = self.get_arm_lengths(batch_size)
            
        
        # Expand arm_lengths if we have a sequence dimension
        if is_sequence and arm_lengths.dim() < 3:
            arm_lengths = arm_lengths.unsqueeze(1).expand(-1, batch_seq_shape[1], -1)
            
        print("joint_angles",joint_angles)
        print("end_effector_pos",end_effector_pos)
        print("target_pos", target_pos)
        print("arm_lengths",arm_lengths)
        
        # Concatenate all properties along the last dimension
        properties = torch.cat([
            joint_angles,
            end_effector_pos,
            target_pos,
            arm_lengths
        ], dim=-1)
        
        return properties
    
    # def extract_properties(self, obs):
    #     """
    #     Extract relevant physical properties from observation dictionary
        
    #     Args:
    #         obs: Dictionary containing observation from DMC Reacher
            
    #     Returns:
    #         Tensor containing concatenated physical properties
    #     """
    #     # Extract properties from observation
    #     batch_size = next(iter(obs.values())).shape[0]
        
    #     # Extract joint angles - based on reference, they are in qpos[:2]
    #     if 'qpos' in obs:
    #         joint_angles = obs['qpos'][..., :2]
    #     elif 'joints' in obs:
    #         joint_angles = obs['joints']
    #     else:
    #         joint_angles = torch.zeros((batch_size, self.joint_angles_dim), device=self.device)
            
    #     # End effector position - should be directly available in the observation
    #     if 'end_effector' in obs:
    #         end_effector_pos = obs['end_effector'][..., :2]  # Take only x,y coordinates
    #     elif 'fingertip' in obs:
    #         end_effector_pos = obs['fingertip'][..., :2]  # Take only x,y coordinates
    #     elif 'geom_xpos' in obs and 'finger_id' in obs:
    #         # If geom_xpos and finger_id are provided, extract directly
    #         end_effector_pos = obs['geom_xpos'][..., obs['finger_id'], :2]
    #     else:
    #         # If not available in any form, return zeros
    #         end_effector_pos = torch.zeros((batch_size, self.end_effector_pos_dim), device=self.device)
            
    #     # Target position - should be directly available in the observation
    #     if 'target' in obs:
    #         target_pos = obs['target'][..., :2]  # Take only x,y coordinates
    #     elif 'target_position' in obs:
    #         target_pos = obs['target_position'][..., :2]
    #     elif 'geom_xpos' in obs and 'target_id' in obs:
    #         # If geom_xpos and target_id are provided, extract directly
    #         target_pos = obs['geom_xpos'][..., obs['target_id'], :2]
    #     else:
    #         # If not available in any form, return zeros
    #         target_pos = torch.zeros((batch_size, self.target_pos_dim), device=self.device)
            
    #     print("joint_angles",joint_angles)
    #     print("end_effector_pos",end_effector_pos)
    #     print("target_pos", target_pos)
        
    #     # Get arm lengths from the environment
    #     arm_lengths = self.get_arm_lengths(batch_size)
        
    #     # Concatenate the 4 key properties
    #     properties = torch.cat([
    #         joint_angles,
    #         end_effector_pos,
    #         target_pos,
    #         arm_lengths
    #     ], dim=-1)
        
    #     return properties
    
    def get_arm_lengths(self, batch_size):
        """
        Returns arm lengths, based on the reference values (0.06 and 0.05)
        """
        if hasattr(self.config, 'arm_lengths'):
            return torch.tensor(self.config.arm_lengths).repeat(batch_size, 1).to(self.device)
        # Default values from the reference code
        return torch.tensor([[0.06, 0.05]]).repeat(batch_size, 1).to(self.device)
    
    def update_normalization(self, properties):
        """Update running mean and std for normalization"""
        with torch.no_grad():
            if self.initialized == 0:
                self.means = properties.mean(dim=0)
                self.stds = properties.std(dim=0).clip(min=1e-6)
                self.initialized.fill_(1)
            else:
                # Exponential moving average
                alpha = 0.05
                self.means = (1 - alpha) * self.means + alpha * properties.mean(dim=0)
                self.stds = (1 - alpha) * self.stds + alpha * properties.std(dim=0).clip(min=1e-6)
    
    def normalize(self, properties):
        """Normalize properties using running statistics"""
        return (properties - self.means) / self.stds
    
    def forward(self, obs):
        """
        Extract physical properties from observations and project to label space
        
        Args:
            obs: Dictionary containing observation from DMC Reacher
            
        Returns:
            Tensor containing labels for CausalVAE
        """
        properties = self.extract_properties(obs)
        
        # Update normalization statistics during training
        if self.training:
            self.update_normalization(properties)
            
        # Normalize properties
        normalized_properties = self.normalize(properties)
        
        # Project to label space
        labels = self.projection(normalized_properties)
        
        return labels
    
class CausalVAE_WorldModel(nn.Module):
    """
    A causal world model for Dreamer that uses CausalVAE for encoding observations and decoding reconstructions.
    This implements the causal structure described in the CausalVAE paper, but adapted for the DMC environment.
    
    The model now learns the causal structure through training rather than using a hard-coded adjacency matrix.
    """
    def __init__(self, obs_space, act_space, step, config):
        super(CausalVAE_WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.device = config.device if config is not None and hasattr(config, 'device') else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize the physical properties extractor with 4 causal factors
        self.physical_properties = ReacherPhysicalProperties(config, device=self.device)
        
        # Extract observation shapes
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        
        # Create the encoder and configure embedding size
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        
        # Define causal structure parameters
        self.z_dim = config.dyn_stoch  # Total latent dimension
        self.z1_dim = 4  # Fixed to 4 causal factors
        self.z2_dim = self.z_dim // self.z1_dim  # Dimension per factor
        
        # Setup the dynamics model (RSSM) from Dreamer - unchanged
        self.dynamics = networks.RSSM(
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
            self.device,
        )
        
        # Create the causal DAG layer with structure learning
        self.dag = DagLayer(self.z1_dim, self.z1_dim)
        
        # Set DAG structure learning parameters
        self.dag_lambda = getattr(config, 'dag_lambda', 0.1)  # Lagrangian multiplier for DAG constraint
        self.dag_alpha = getattr(config, 'dag_alpha', 0.0)  # L1 regularization for sparsity
        self.dag_rho = getattr(config, 'dag_rho', 1.0)  # Initial penalty coefficient
        self.dag_rho_max = getattr(config, 'dag_rho_max', 1e6)  # Maximum penalty coefficient
        self.dag_rho_increase = getattr(config, 'dag_rho_increase', 2.0)  # Multiplier for increasing rho
        
        # Create attention mechanism for causal disentanglement
        self.attn = Attention(self.z2_dim)
        
        # Create mask layers for manipulating latent variables
        self.mask_z = MaskLayer(self.z_dim, concept=self.z1_dim, z2_dim=self.z2_dim)
        self.mask_u = MaskLayer(self.z1_dim, concept=self.z1_dim, z2_dim=1)
        
        # Create the output heads
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
            device=self.device,
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
            device=self.device,
            name="Cont",
        )
        
        # Setup optimizer
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
        print(
            f"CausalVAE model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        
        # Set up loss scaling factors
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        
        # Define scaling factors for causal structure - adjusted for 4 factors
        self.scale = torch.zeros((self.z1_dim, 2), device=self.device)
        self._initialize_scale()
        
    def _initialize_scale(self):
        """Initialize scale for the 4 causal variables based on DMC environment"""
        # Approximate ranges for our 4 physical factors
        self.scale[0] = torch.tensor([0, 6.28])  # Joint angles (0 to 2π)
        self.scale[1] = torch.tensor([-1, 1])    # End effector position
        self.scale[2] = torch.tensor([-1, 1])    # Target position
        self.scale[3] = torch.tensor([0.01, 0.1]) # Arm lengths
            
    def causal_encode(self, embed, label=None):
        """
        Apply the causal encoding process to embedded observations.
        
        Args:
            embed: Embedded observation from encoder
            label: Optional labels for supervised training
            
        Returns:
            Causally encoded representation
        """
        batch_size = embed.size(0)
        
        # If no labels provided, use zeros (unsupervised mode)
        if label is None:
            label = torch.zeros(batch_size, self.z1_dim, device=self.device)
        
        # Apply encoder to get independent latent factors (q_m and q_v are mean and variance)
        q_m, q_v = torch.split(self.dynamics._suff_stats_layer("obs", embed), [self.z_dim, self.z_dim], -1)
        
        # Reshape for causal structure processing
        q_m = q_m.reshape([batch_size, self.z1_dim, self.z2_dim])
        q_v = torch.ones(batch_size, self.z1_dim, self.z2_dim).to(self.device)
        
        # Apply causal DAG to transform independent variables to causally related ones
        decode_m, decode_v = self.dag.calculate_dag(q_m.to(self.device), q_v.to(self.device))
        decode_m = decode_m.reshape([batch_size, self.z1_dim, self.z2_dim])
        decode_v = decode_v.reshape([batch_size, self.z1_dim, self.z2_dim])
        
        # Apply masking operations for the SCM
        m_zm = self.dag.mask_z(decode_m.to(self.device)).reshape([batch_size, self.z1_dim, self.z2_dim])
        m_u = self.dag.mask_u(label.to(self.device))
        
        # Mix masked variables
        f_z = self.mask_z.mix(m_zm).reshape([batch_size, self.z1_dim, self.z2_dim]).to(self.device)
        
        # Apply attention
        e_tilde, _ = self.attn.attention(
            decode_m.reshape([batch_size, self.z1_dim, self.z2_dim]).to(self.device),
            q_m.reshape([batch_size, self.z1_dim, self.z2_dim]).to(self.device)
        )
        
        # Generate final causal representation
        f_z1 = f_z + e_tilde
        
        # Sample from the causal model
        z_given_dag = conditional_sample_gaussian(f_z1, decode_v * 0.001)
        
        # Reshape to flat vector for RSSM
        z_flat = z_given_dag.reshape([batch_size, self.z_dim])
        
        return z_flat
    
    def _train(self, data):
        """
        Training function for CausalVAE world model
        
        Args:
            data: Batch of experience data
            
        Returns:
            Updated model parameters and metrics
        """
        # Preprocess data
        data = self.preprocess(data)
        
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # Encode observations
                embed = self.encoder(data)
                
                # Extract physical properties to use as causal labels
                labels = self.physical_properties(data)
                
                # Apply causal encoding if configured
                if hasattr(self._config, 'use_causal_encode') and self._config.use_causal_encode:
                    embed = self.causal_encode(embed, labels)
                
                # Use standard RSSM dynamics for temporal modeling
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
                
                # Get features and generate predictions
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
                
                # Compute losses for all predictions
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                
                # Add DAG losses for structure learning
                h_A = self._compute_dag_constraint()
                l1_reg = self._compute_l1_regularization()
                
                # Augmented Lagrangian Method (ALM) for enforcing the DAG constraint
                dag_loss = self.dag_lambda * h_A + 0.5 * self.dag_rho * (h_A ** 2) + self.dag_alpha * l1_reg
                losses["dag"] = dag_loss
                
                # Scale losses and combine
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
                
                # Optimize model
                metrics = self._model_opt(torch.mean(model_loss), self.parameters())
                
                # Update DAG constraint parameters
                self._update_dag_parameters(h_A)
        
        # Track metrics
        metrics.update({f"{name}_loss": tools.to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = tools.to_np(dyn_loss)
        metrics["rep_loss"] = tools.to_np(rep_loss)
        metrics["kl"] = tools.to_np(torch.mean(kl_value))
        metrics["dag_h_A"] = tools.to_np(h_A)
        metrics["dag_l1_reg"] = tools.to_np(l1_reg)
        metrics["dag_rho"] = self.dag_rho
        metrics["dag_lambda"] = tools.to_np(self.dag_lambda)
        
        # Print adjacency matrix every 1000 steps for monitoring
        if self._step % 1000 == 0:
            print(f"DAG Adjacency Matrix at step {self._step}:")
            print(to_np(self.dag.A))
        
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = tools.to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = tools.to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
                labels=labels,  # Add labels to context for potential use elsewhere
            )
        
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics
    
    def _compute_dag_constraint(self):
        """Compute the DAG constraint for the adjacency matrix"""
        # Use the DAGness constraint from Yu et al. (DAG-GNN)
        # h(A) = tr((I + A○A/d)^d) - d
        d = self.z1_dim
        A = self.dag.A
        M = torch.eye(d, device=A.device) + A * A / d
        h_A = torch.trace(torch.matrix_power(M, d)) - d
        return h_A
    
    def _compute_l1_regularization(self):
        """Compute L1 regularization for sparsity"""
        return torch.sum(torch.abs(self.dag.A))
    
    def _update_dag_parameters(self, h_A):
        """Update parameters for DAG constraint optimization"""
        # Update Lagrangian multiplier
        with torch.no_grad():
            self.dag_lambda = self.dag_lambda + self.dag_rho * h_A
            
            # Increase penalty parameter if constraint not satisfied enough
            if h_A > 0.25:  # Threshold for increasing rho
                self.dag_rho = min(self.dag_rho * self.dag_rho_increase, self.dag_rho_max)
    
    def preprocess(self, obs):
        """
        Preprocess observations for the Reacher environment
        
        Args:
            obs: Raw observations
            
        Returns:
            Preprocessed observations
        """
        # Convert numpy arrays to tensors
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        
        # Normalize image data if present
        if "image" in obs:
            obs["image"] = obs["image"] / 255.0
            
        # Process discount factor if present
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = obs["discount"].unsqueeze(-1)
        
        # Process terminal flags
        assert "is_first" in obs
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        
        assert "end_effector_pos" in obs
        assert "target_pos" in obs
        assert "arm_lengths" in obs
        
        # For Reacher environment, extract joint positions, velocities, etc.
        # These are needed for the physical_properties module
        # if "qpos" not in obs and "state" in obs:
        #     # Some environments provide state instead of separate qpos/qvel
        #     # Try to extract the required fields from state
        #     state = obs["state"]
        #     if state.shape[-1] >= 4:
        #         # Typical Reacher state has: [joint_angles, joint_velocities, target_pos]
        #         # Extract joint angles (first 2 dimensions)
        #         obs["qpos"] = state[..., :2]
                
        #         # Extract target position if available
        #         if state.shape[-1] >= 6:
        #             obs["target"] = state[..., 4:6]
        
        # Add the needed Reacher-specific keys if available in the raw observation
        # but not yet in our processed dict
        # for key in ["joints", "end_effector", "fingertip", "target"]:
        #     if key in obs.keys():
        #         # These keys already exist, no need to process
        #         continue
                
        #     # Standard DMC keys that might correspond to what we need
        #     if key == "joints" and "joints" not in obs and "qpos" in obs:
        #         obs["joints"] = obs["qpos"][..., :2]
        #     elif key in ["end_effector", "fingertip"] and "fingertip" not in obs and "end_effector" not in obs:
        #         # If we have joint positions but not end effector position,
        #         # we can calculate it in the physical_properties module
        #         pass
        
        return obs
    
    def video_pred(self, data):
        """
        Generate video predictions for visualization
        
        Args:
            data: Batch of data
            
        Returns:
            Video prediction tensor
        """
        data = self.preprocess(data)
        embed = self.encoder(data)
        
        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[:6]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        error = (model - truth + 1.0) / 2.0
        
        return torch.cat([truth, model, error], 2)
    
    def intervene(self, post, intervention_idx, intervention_value=None):
        """
        Perform causal intervention on latent variables
        
        Args:
            post: Current posterior latent state
            intervention_idx: Index of causal factor to intervene on (0-3)
            intervention_value: Value to set for intervention (None = random value)
            
        Returns:
            Modified latent state after intervention
        """
        # Create copy of posterior to modify
        post_modified = {k: v.clone() for k, v in post.items()}
        
        # Extract stochastic latent variables
        stoch = post_modified["stoch"]
        batch_size = stoch.size(0)
        
        # Reshape to causal structure
        z = stoch.reshape(batch_size, self.z1_dim, self.z2_dim)
        
        # If intervention value not provided, sample a random value
        if intervention_value is None:
            # Sample random value within factor's scale range
            low, high = self.scale[intervention_idx]
            intervention_value = torch.rand(batch_size, self.z2_dim, device=self.device) * (high - low) + low
        else:
            # Ensure correct shape
            intervention_value = intervention_value.reshape(batch_size, self.z2_dim)
        
        # Apply intervention by setting the value
        z[:, intervention_idx, :] = intervention_value
        
        # Reshape back to flat vector
        post_modified["stoch"] = z.reshape(batch_size, self.z_dim)
        
        # If using mean for evaluation, update it too
        if "mean" in post_modified:
            mean = post_modified["mean"].reshape(batch_size, self.z1_dim, self.z2_dim)
            mean[:, intervention_idx, :] = intervention_value
            post_modified["mean"] = mean.reshape(batch_size, self.z_dim)
        
        return post_modified

    def get_dag_adjacency(self):
        """Return the current adjacency matrix for visualization/analysis"""
        return to_np(self.dag.A)
    
# class CausalVAE_WorldModel(nn.Module):
#     """
#     A causal world model for Dreamer that uses CausalVAE for encoding observations and decoding reconstructions.
#     This implements the causal structure described in the CausalVAE paper, but adapted for the DMC environment.
    
#     The model now learns the causal structure through training rather than using a hard-coded adjacency matrix.
#     """
#     def __init__(self, obs_space, act_space, step, config):
#         super(CausalVAE_WorldModel, self).__init__()
#         self._step = step
#         self._use_amp = True if config.precision == 16 else False
#         self._config = config
#         self.device = config.device if config is not None and hasattr(config, 'device') else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         self.physical_properties = ReacherPhysicalProperties(config, device=self.device)
        
#         # Extract observation shapes
#         shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        
#         # Create the encoder and configure embedding size
#         self.encoder = networks.MultiEncoder(shapes, **config.encoder)
#         self.embed_size = self.encoder.outdim
        
#         # Define causal structure parameters
#         self.z_dim = config.dyn_stoch  # Total latent dimension
#         self.z1_dim = getattr(config, 'causal_factors', 7)  # Number of causal factors
#         self.z2_dim = self.z_dim // self.z1_dim  # Dimension per factor
        
#         # Setup the dynamics model (RSSM) from Dreamer - unchanged
#         self.dynamics = networks.RSSM(
#             config.dyn_stoch,
#             config.dyn_deter,
#             config.dyn_hidden,
#             config.dyn_rec_depth,
#             config.dyn_discrete,
#             config.act,
#             config.norm,
#             config.dyn_mean_act,
#             config.dyn_std_act,
#             config.dyn_min_std,
#             config.unimix_ratio,
#             config.initial,
#             config.num_actions,
#             self.embed_size,
#             self.device,
#         )
        
#         # Create the causal DAG layer with structure learning
#         self.dag = DagLayer(self.z1_dim, self.z1_dim)
        
#         # Set DAG structure learning parameters
#         self.dag_lambda = getattr(config, 'dag_lambda', 0.1)  # Lagrangian multiplier for DAG constraint
#         self.dag_alpha = getattr(config, 'dag_alpha', 0.0)  # L1 regularization for sparsity
#         self.dag_rho = getattr(config, 'dag_rho', 1.0)  # Initial penalty coefficient
#         self.dag_rho_max = getattr(config, 'dag_rho_max', 1e6)  # Maximum penalty coefficient
#         self.dag_rho_increase = getattr(config, 'dag_rho_increase', 2.0)  # Multiplier for increasing rho
        
#         # Create attention mechanism for causal disentanglement
#         self.attn = Attention(self.z2_dim)
        
#         # Create mask layers for manipulating latent variables
#         self.mask_z = MaskLayer(self.z_dim, concept=self.z1_dim, z2_dim=self.z2_dim)
#         self.mask_u = MaskLayer(self.z1_dim, concept=self.z1_dim, z2_dim=1)
        
#         # Create the output heads
#         self.heads = nn.ModuleDict()
#         if config.dyn_discrete:
#             feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
#         else:
#             feat_size = config.dyn_stoch + config.dyn_deter
            
#         self.heads["decoder"] = networks.MultiDecoder(
#             feat_size, shapes, **config.decoder
#         )
#         self.heads["reward"] = networks.MLP(
#             feat_size,
#             (255,) if config.reward_head["dist"] == "symlog_disc" else (),
#             config.reward_head["layers"],
#             config.units,
#             config.act,
#             config.norm,
#             dist=config.reward_head["dist"],
#             outscale=config.reward_head["outscale"],
#             device=self.device,
#             name="Reward",
#         )
#         self.heads["cont"] = networks.MLP(
#             feat_size,
#             (),
#             config.cont_head["layers"],
#             config.units,
#             config.act,
#             config.norm,
#             dist="binary",
#             outscale=config.cont_head["outscale"],
#             device=self.device,
#             name="Cont",
#         )
        
#         # Setup optimizer
#         for name in config.grad_heads:
#             assert name in self.heads, name
#         self._model_opt = tools.Optimizer(
#             "model",
#             self.parameters(),
#             config.model_lr,
#             config.opt_eps,
#             config.grad_clip,
#             config.weight_decay,
#             opt=config.opt,
#             use_amp=self._use_amp,
#         )
#         print(
#             f"CausalVAE model_opt has {sum(param.numel() for param in self.parameters())} variables."
#         )
        
#         # Set up loss scaling factors
#         self._scales = dict(
#             reward=config.reward_head["loss_scale"],
#             cont=config.cont_head["loss_scale"],
#         )
        
#         # Define scaling factors for causal structure
#         self.scale = torch.zeros((self.z1_dim, 2), device=self.device)
#         self._initialize_scale()
        
#     def _initialize_scale(self):
#         """Initialize scale for the causal variables based on DMC environment"""
#         # Approximate ranges for various factors in the DMC environment
#         if self.z1_dim >= 1:
#             self.scale[0] = torch.tensor([0, 6.28])  # Joint angles (0 to 2π)
#         if self.z1_dim >= 2:
#             self.scale[1] = torch.tensor([-10, 10])  # Joint velocities
#         if self.z1_dim >= 3:
#             self.scale[2] = torch.tensor([-1, 1])  # End effector position
#         if self.z1_dim >= 4:
#             self.scale[3] = torch.tensor([-1, 1])  # Target position
#         if self.z1_dim >= 5:
#             self.scale[4] = torch.tensor([0.5, 1.5])  # Arm lengths
#         if self.z1_dim >= 6:
#             self.scale[5] = torch.tensor([0.1, 2.0])  # Joint damping
#         if self.z1_dim >= 7:
#             self.scale[6] = torch.tensor([0.5, 2.0])  # Arm mass/inertia
            
#     def causal_encode(self, embed, label=None):
#         """
#         Apply the causal encoding process to embedded observations.
        
#         Args:
#             embed: Embedded observation from encoder
#             label: Optional labels for supervised training
            
#         Returns:
#             Causally encoded representation
#         """
#         batch_size = embed.size(0)
        
#         # If no labels provided, use zeros (unsupervised mode)
#         if label is None:
#             label = torch.zeros(batch_size, self.z1_dim, device=self.device)
        
#         # Apply encoder to get independent latent factors (q_m and q_v are mean and variance)
#         q_m, q_v = torch.split(self.dynamics._suff_stats_layer("obs", embed), [self.z_dim, self.z_dim], -1)
        
#         # Reshape for causal structure processing
#         q_m = q_m.reshape([batch_size, self.z1_dim, self.z2_dim])
#         q_v = torch.ones(batch_size, self.z1_dim, self.z2_dim).to(self.device)
        
#         # Apply causal DAG to transform independent variables to causally related ones
#         decode_m, decode_v = self.dag.calculate_dag(q_m.to(self.device), q_v.to(self.device))
#         decode_m = decode_m.reshape([batch_size, self.z1_dim, self.z2_dim])
#         decode_v = decode_v.reshape([batch_size, self.z1_dim, self.z2_dim])
        
#         # Apply masking operations for the SCM
#         m_zm = self.dag.mask_z(decode_m.to(self.device)).reshape([batch_size, self.z1_dim, self.z2_dim])
#         m_u = self.dag.mask_u(label.to(self.device))
        
#         # Mix masked variables
#         f_z = self.mask_z.mix(m_zm).reshape([batch_size, self.z1_dim, self.z2_dim]).to(self.device)
        
#         # Apply attention
#         e_tilde, _ = self.attn.attention(
#             decode_m.reshape([batch_size, self.z1_dim, self.z2_dim]).to(self.device),
#             q_m.reshape([batch_size, self.z1_dim, self.z2_dim]).to(self.device)
#         )
        
#         # Generate final causal representation
#         f_z1 = f_z + e_tilde
        
#         # Sample from the causal model
#         z_given_dag = conditional_sample_gaussian(f_z1, decode_v * 0.001)
        
#         # Reshape to flat vector for RSSM
#         z_flat = z_given_dag.reshape([batch_size, self.z_dim])
        
#         return z_flat
    
#     def _train(self, data):
#         """
#         Training function for CausalVAE world model
        
#         Args:
#             data: Batch of experience data
            
#         Returns:
#             Updated model parameters and metrics
#         """
#         # Preprocess data
#         data = self.preprocess(data)
        
#         with tools.RequiresGrad(self):
#             with torch.cuda.amp.autocast(self._use_amp):
#                 # Encode observations
#                 embed = self.encoder(data)
                
#                 # Extract physical properties to use as causal labels
#                 labels = self.physical_properties(data)
                
#                 # Apply causal encoding if configured
#                 if hasattr(self._config, 'use_causal_encode') and self._config.use_causal_encode:
#                     embed = self.causal_encode(embed, labels)
                
#                 # Use standard RSSM dynamics for temporal modeling
#                 post, prior = self.dynamics.observe(
#                     embed, data["action"], data["is_first"]
#                 )
                
#                 # Compute KL divergence loss
#                 kl_free = self._config.kl_free
#                 dyn_scale = self._config.dyn_scale
#                 rep_scale = self._config.rep_scale
#                 kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
#                     post, prior, kl_free, dyn_scale, rep_scale
#                 )
                
#                 # Get features and generate predictions
#                 preds = {}
#                 for name, head in self.heads.items():
#                     grad_head = name in self._config.grad_heads
#                     feat = self.dynamics.get_feat(post)
#                     feat = feat if grad_head else feat.detach()
#                     pred = head(feat)
#                     if type(pred) is dict:
#                         preds.update(pred)
#                     else:
#                         preds[name] = pred
                
#                 # Compute losses for all predictions
#                 losses = {}
#                 for name, pred in preds.items():
#                     loss = -pred.log_prob(data[name])
#                     assert loss.shape == embed.shape[:2], (name, loss.shape)
#                     losses[name] = loss
                
#                 # Add DAG losses for structure learning
#                 h_A = self._compute_dag_constraint()
#                 l1_reg = self._compute_l1_regularization()
                
#                 # Augmented Lagrangian Method (ALM) for enforcing the DAG constraint
#                 dag_loss = self.dag_lambda * h_A + 0.5 * self.dag_rho * (h_A ** 2) + self.dag_alpha * l1_reg
#                 losses["dag"] = dag_loss
                
#                 # Add property prediction loss - supervise the causal factors if needed
#                 if hasattr(self._config, 'use_property_supervision') and self._config.use_property_supervision:
#                     # Use the latent state to predict physical properties
#                     predicted_properties = self.heads.get("properties", None)
#                     if predicted_properties is not None:
#                         property_loss = F.mse_loss(predicted_properties(self.dynamics.get_feat(post)), labels)
#                         losses["properties"] = property_loss.unsqueeze(0).unsqueeze(0).expand(embed.shape[:2])
                
#                 # Scale losses and combine
#                 scaled = {
#                     key: value * self._scales.get(key, 1.0)
#                     for key, value in losses.items()
#                 }
#                 model_loss = sum(scaled.values()) + kl_loss
                
#                 # Optimize model
#                 metrics = self._model_opt(torch.mean(model_loss), self.parameters())
                
#                 # Update DAG constraint parameters
#                 self._update_dag_parameters(h_A)
        
#         # Track metrics
#         metrics.update({f"{name}_loss": tools.to_np(loss) for name, loss in losses.items()})
#         metrics["kl_free"] = kl_free
#         metrics["dyn_scale"] = dyn_scale
#         metrics["rep_scale"] = rep_scale
#         metrics["dyn_loss"] = tools.to_np(dyn_loss)
#         metrics["rep_loss"] = tools.to_np(rep_loss)
#         metrics["kl"] = tools.to_np(torch.mean(kl_value))
#         metrics["dag_h_A"] = tools.to_np(h_A)
#         metrics["dag_l1_reg"] = tools.to_np(l1_reg)
#         metrics["dag_rho"] = self.dag_rho
#         metrics["dag_lambda"] = self.dag_lambda
        
#         # Print adjacency matrix every 1000 steps for monitoring
#         if self._step % 1000 == 0:
#             print(f"DAG Adjacency Matrix at step {self._step}:")
#             print(to_np(self.dag.A))
        
#         with torch.cuda.amp.autocast(self._use_amp):
#             metrics["prior_ent"] = tools.to_np(
#                 torch.mean(self.dynamics.get_dist(prior).entropy())
#             )
#             metrics["post_ent"] = tools.to_np(
#                 torch.mean(self.dynamics.get_dist(post).entropy())
#             )
#             context = dict(
#                 embed=embed,
#                 feat=self.dynamics.get_feat(post),
#                 kl=kl_value,
#                 postent=self.dynamics.get_dist(post).entropy(),
#                 labels=labels,  # Add labels to context for potential use elsewhere
#             )
        
#         post = {k: v.detach() for k, v in post.items()}
#         return post, context, metrics
    
#     def _compute_dag_constraint(self):
#         """Compute the DAG constraint for the adjacency matrix"""
#         # Use the DAGness constraint from Yu et al. (DAG-GNN)
#         # h(A) = tr((I + A○A/d)^d) - d
#         d = self.z1_dim
#         A = self.dag.A
#         M = torch.eye(d, device=A.device) + A * A / d
#         h_A = torch.trace(torch.matrix_power(M, d)) - d
#         return h_A
    
#     def _compute_l1_regularization(self):
#         """Compute L1 regularization for sparsity"""
#         return torch.sum(torch.abs(self.dag.A))
    
#     def _update_dag_parameters(self, h_A):
#         """Update parameters for DAG constraint optimization"""
#         # Update Lagrangian multiplier
#         with torch.no_grad():
#             self.dag_lambda = self.dag_lambda + self.dag_rho * h_A
            
#             # Increase penalty parameter if constraint not satisfied enough
#             if h_A > 0.25:  # Threshold for increasing rho
#                 self.dag_rho = min(self.dag_rho * self.dag_rho_increase, self.dag_rho_max)
    
#     def preprocess(self, obs):
#         """
#         Preprocess observations
        
#         Args:
#             obs: Raw observations
            
#         Returns:
#             Preprocessed observations
#         """
#         obs = {
#             k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
#             for k, v in obs.items()
#         }
#         obs["image"] = obs["image"] / 255.0
#         if "discount" in obs:
#             obs["discount"] *= self._config.discount
#             obs["discount"] = obs["discount"].unsqueeze(-1)
        
#         assert "is_first" in obs
#         assert "is_terminal" in obs
#         obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
#         return obs
    
 
#     def video_pred(self, data):
#         """
#         Generate video predictions for visualization
        
#         Args:
#             data: Batch of data
            
#         Returns:
#             Video prediction tensor
#         """
#         data = self.preprocess(data)
#         embed = self.encoder(data)
        
#         states, _ = self.dynamics.observe(
#             embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
#         )
#         recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[:6]
#         reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        
#         init = {k: v[:, -1] for k, v in states.items()}
#         prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
#         openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        
#         model = torch.cat([recon[:, :5], openl], 1)
#         truth = data["image"][:6]
#         error = (model - truth + 1.0) / 2.0
        
#         return torch.cat([truth, model, error], 2)
    
#     def intervene(self, post, intervention_idx, intervention_value=None):
#         """
#         Perform causal intervention on latent variables
        
#         Args:
#             post: Current posterior latent state
#             intervention_idx: Index of causal factor to intervene on
#             intervention_value: Value to set for intervention (None = random value)
            
#         Returns:
#             Modified latent state after intervention
#         """
#         # Create copy of posterior to modify
#         post_modified = {k: v.clone() for k, v in post.items()}
        
#         # Extract stochastic latent variables
#         stoch = post_modified["stoch"]
#         batch_size = stoch.size(0)
        
#         # Reshape to causal structure
#         z = stoch.reshape(batch_size, self.z1_dim, self.z2_dim)
        
#         # If intervention value not provided, sample a random value
#         if intervention_value is None:
#             # Sample random value within factor's scale range
#             low, high = self.scale[intervention_idx]
#             intervention_value = torch.rand(batch_size, self.z2_dim, device=self.device) * (high - low) + low
#         else:
#             # Ensure correct shape
#             intervention_value = intervention_value.reshape(batch_size, self.z2_dim)
        
#         # Apply intervention by setting the value
#         z[:, intervention_idx, :] = intervention_value
        
#         # Reshape back to flat vector
#         post_modified["stoch"] = z.reshape(batch_size, self.z_dim)
        
#         # If using mean for evaluation, update it too
#         if "mean" in post_modified:
#             mean = post_modified["mean"].reshape(batch_size, self.z1_dim, self.z2_dim)
#             mean[:, intervention_idx, :] = intervention_value
#             post_modified["mean"] = mean.reshape(batch_size, self.z_dim)
        
#         return post_modified

#     def get_dag_adjacency(self):
#         """Return the current adjacency matrix for visualization/analysis"""
#         return to_np(self.dag.A)
    
    
# Import necessary classes from mask.py
class MaskLayer(nn.Module):
    def __init__(self, z_dim, concept=4, z2_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.z2_dim = z2_dim
        self.concept = concept
        
        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z2_dim, 32),
            nn.ELU(),
            nn.Linear(32, z2_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(z2_dim, 32),
            nn.ELU(),
            nn.Linear(32, z2_dim),
        )
        self.net3 = nn.Sequential(
            nn.Linear(z2_dim, 32),
            nn.ELU(),
            nn.Linear(32, z2_dim),
        )
        self.net4 = nn.Sequential(
            nn.Linear(z2_dim, 32),
            nn.ELU(),
            nn.Linear(32, z2_dim)
        )
        
        # Add more networks if needed to handle more concepts
        nets = [self.net1, self.net2, self.net3, self.net4]
        
        # Create additional networks for concepts > 4
        for i in range(4, concept):
            setattr(self, f"net{i+1}", nn.Sequential(
                nn.Linear(z2_dim, 32),
                nn.ELU(),
                nn.Linear(32, z2_dim),
            ))
            nets.append(getattr(self, f"net{i+1}"))
        
        self.nets = nn.ModuleList(nets)
        
        self.net = nn.Sequential(
            nn.Linear(z2_dim, 32),
            nn.ELU(),
            nn.Linear(32, z2_dim),
        )
        
    def masked(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z
   
    def masked_sep(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z
   
    def mix(self, z):
        zy = z.view(-1, self.concept*self.z2_dim)
        
        if self.z2_dim == 1:
            zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
            components = [zy[:, i] for i in range(self.concept)]
        else:
            split_size = self.z_dim // self.concept
            components = torch.split(zy, split_size, dim=1)
        
        results = []
        for i, component in enumerate(components):
            if i < len(self.nets):
                results.append(self.nets[i](component))
            else:
                # Fallback for any extra components
                results.append(self.net(component))
        
        # Concatenate all processed components
        h = torch.cat(results, dim=1)
        return h


class Attention(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.M = nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features, in_features), mean=0, std=1))
        self.sigmd = torch.nn.Sigmoid()
    
    def attention(self, z, e):
        a = z.matmul(self.M).matmul(e.permute(0, 2, 1))
        a = self.sigmd(a)
        A = torch.softmax(a, dim=1)
        e = torch.matmul(A, e)
        return e, A
    
class DagLayer(nn.Linear):
    def __init__(self, in_features, out_features, i=False, bias=False):
        super(nn.Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.i = i
        
        # Initialize adjacency matrix with small random values to break symmetry
        # We use small values to start with an almost-zero matrix
        # but with enough randomness to break ties during optimization
        self.A = nn.Parameter(torch.randn(out_features, out_features) * 0.01)
        
        # No self-loops in DAG
        with torch.no_grad():
            self.A.diagonal().fill_(0)
        
        # Identity matrix for calculations
        self.I = nn.Parameter(torch.eye(out_features))
        self.I.requires_grad = False
        
        # Temporary matrix for mask operations
        self.B = nn.Parameter(torch.eye(out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
    def mask_z(self, x):
        self.B = self.A
        x = torch.matmul(self.B.t(), x)
        return x
        
    def mask_u(self, x):
        self.B = self.A
        x = x.view(-1, x.size()[1], 1)
        x = torch.matmul(self.B.t(), x)
        return x
        
    def calculate_dag(self, x, v):
        if x.dim() > 2:
            x = x.permute(0, 2, 1)
            
        # Ensure no self-loops in DAG during computation
        A_processed = self.A.clone()
        A_processed.diagonal().fill_(0)
        
        # Add numerical stability
        eps = 1e-6
        
        # Ensure matrix I - A is invertible
        try:
            inv_matrix = torch.inverse(self.I - A_processed.t())
        except RuntimeError:
            # Fallback in case of numerical issues
            A_stable = A_processed * 0.9  # Scale down adjacency matrix
            inv_matrix = torch.inverse(self.I - A_stable.t() + eps * torch.eye(self.out_features, device=A_processed.device))
            
        x = F.linear(x, inv_matrix, self.bias)
        
        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
        
        return x, v
    
    def forward(self, x):
        # Ensure no self-loops
        A_processed = self.A.clone()
        A_processed.diagonal().fill_(0)
        
        x = x * torch.inverse((A_processed) + self.I)
        return x

# class DagLayer(nn.Linear):
#     def __init__(self, in_features, out_features, i=False, bias=False, initial=True):
#         super(nn.Linear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.i = i
        
#         # Initialize adjacency matrix
#         self.a = torch.zeros(out_features, out_features)
        
#         # Set initial causal structure if needed
#         if initial and out_features >= 4:
#             self.a[0][1], self.a[0][2], self.a[0][3] = 1, 1, 1
#             self.a[1][2], self.a[1][3] = 1, 1
            
#         self.A = nn.Parameter(self.a)
        
#         # Identity matrix
#         self.b = torch.eye(out_features)
#         self.B = nn.Parameter(self.b)
        
#         # Fixed identity for computations
#         self.I = nn.Parameter(torch.eye(out_features))
#         self.I.requires_grad = False
        
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
            
#     def mask_z(self, x):
#         self.B = self.A
#         x = torch.matmul(self.B.t(), x)
#         return x
        
#     def mask_u(self, x):
#         self.B = self.A
#         x = x.view(-1, x.size()[1], 1)
#         x = torch.matmul(self.B.t(), x)
#         return x
        
#     def calculate_dag(self, x, v):
#         if x.dim() > 2:
#             x = x.permute(0, 2, 1)
        
#         x = F.linear(x, torch.inverse(self.I - self.A.t()), self.bias)
        
#         if x.dim() > 2:
#             x = x.permute(0, 2, 1).contiguous()
        
#         return x, v
    
#     def forward(self, x):
#         x = x * torch.inverse((self.A) + self.I)
#         return x
    
    
    

    
    
