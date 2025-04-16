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


class ReacherPhysicalProperties(nn.Module):
    """
    Module to extract and process physical properties from Reacher environment
    to be used as labels for CausalVAE.
    """
    def __init__(self, config, device=None):
        super(ReacherPhysicalProperties, self).__init__()
        self.config = config
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Define dimensions for each property
        self.joint_angles_dim = 2  # 2 joints for standard Reacher
        self.joint_velocities_dim = 2
        self.end_effector_pos_dim = 2  # x, y coordinates
        self.target_pos_dim = 2  # x, y coordinates
        self.arm_lengths_dim = 2  # length of each arm segment
        self.joint_damping_dim = 2  # damping of each joint
        self.arm_mass_dim = 2  # mass/inertia of each arm
        
        # Total dimension of all properties
        self.total_dim = (self.joint_angles_dim + self.joint_velocities_dim + 
                         self.end_effector_pos_dim + self.target_pos_dim + 
                         self.arm_lengths_dim + self.joint_damping_dim + 
                         self.arm_mass_dim)
        
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
        Extract relevant physical properties from observation dictionary
        
        Args:
            obs: Dictionary containing observation from DMC Reacher
            
        Returns:
            Tensor containing concatenated physical properties
        """
        # Extract properties from observation
        # Note: Actual keys may differ based on your environment setup
        batch_size = next(iter(obs.values())).shape[0]
        
        # Extract available properties, with fallbacks
        # Joint angles
        if 'joints' in obs:
            joint_angles = obs['joints']
        elif 'orientation' in obs:
            joint_angles = obs['orientation']
        elif 'proprioception' in obs and obs['proprioception'].shape[-1] >= 2:
            joint_angles = obs['proprioception'][..., :2]
        else:
            # If not available, create zeros with proper shape
            joint_angles = torch.zeros((batch_size, self.joint_angles_dim), device=self.device)
            
        # Joint velocities
        if 'jointvel' in obs:
            joint_velocities = obs['jointvel']
        elif 'velocity' in obs:
            joint_velocities = obs['velocity']
        elif 'proprioception' in obs and obs['proprioception'].shape[-1] >= 4:
            joint_velocities = obs['proprioception'][..., 2:4]
        else:
            joint_velocities = torch.zeros((batch_size, self.joint_velocities_dim), device=self.device)
            
        # End effector position
        if 'end_effector' in obs:
            end_effector_pos = obs['end_effector']
        else:
            # Calculate from joint angles if possible
            end_effector_pos = self._calculate_end_effector_position(joint_angles)
            
        # Target position
        if 'target' in obs:
            target_pos = obs['target']
        elif 'target_position' in obs:
            target_pos = obs['target_position']
        elif 'proprioception' in obs and obs['proprioception'].shape[-1] >= 6:
            target_pos = obs['proprioception'][..., 4:6]
        else:
            target_pos = torch.zeros((batch_size, self.target_pos_dim), device=self.device)
            
        # Physical parameters - these might be constants or extracted elsewhere
        # For now, setting as learnable parameters
        arm_lengths = self.get_arm_lengths(batch_size)
        joint_damping = self.get_joint_damping(batch_size)
        arm_mass = self.get_arm_mass(batch_size)
        
        # Concatenate all properties
        properties = torch.cat([
            joint_angles,
            joint_velocities,
            end_effector_pos,
            target_pos,
            arm_lengths,
            joint_damping,
            arm_mass
        ], dim=-1)
        
        return properties
    
    def _calculate_end_effector_position(self, joint_angles):
        """
        Calculate end effector position from joint angles
        
        Args:
            joint_angles: Tensor of shape [..., 2] containing joint angles
            
        Returns:
            Tensor of shape [..., 2] containing end effector x, y coordinates
        """
        # Get original shape and reshape for calculation
        original_shape = joint_angles.shape[:-1]
        joint_angles = joint_angles.reshape(-1, 2)
        
        # Calculate forward kinematics
        arm_lengths = self.get_arm_lengths(joint_angles.shape[0])
        l1, l2 = arm_lengths[0, 0].item(), arm_lengths[0, 1].item()
        theta1 = joint_angles[:, 0]
        theta2 = joint_angles[:, 1]
        
        # Position of first joint
        x1 = l1 * torch.cos(theta1)
        y1 = l1 * torch.sin(theta1)
        
        # Position of end effector
        x2 = x1 + l2 * torch.cos(theta1 + theta2)
        y2 = y1 + l2 * torch.sin(theta1 + theta2)
        
        # Combine coordinates
        end_effector = torch.stack([x2, y2], dim=-1)
        
        # Reshape to original dimensions plus 2 for x,y
        return end_effector.reshape(*original_shape, 2)
    
    def get_arm_lengths(self, batch_size):
        """Returns arm lengths, either from config or defaults"""
        if hasattr(self.config, 'arm_lengths'):
            return torch.tensor(self.config.arm_lengths).repeat(batch_size, 1).to(self.device)
        return torch.tensor([[0.12, 0.12]]).repeat(batch_size, 1).to(self.device)
    
    def get_joint_damping(self, batch_size):
        """Returns joint damping, either from config or defaults"""
        if hasattr(self.config, 'joint_damping'):
            return torch.tensor(self.config.joint_damping).repeat(batch_size, 1).to(self.device)
        return torch.tensor([[0.1, 0.1]]).repeat(batch_size, 1).to(self.device)
    
    def get_arm_mass(self, batch_size):
        """Returns arm mass/inertia, either from config or defaults"""
        if hasattr(self.config, 'arm_mass'):
            return torch.tensor(self.config.arm_mass).repeat(batch_size, 1).to(self.device)
        return torch.tensor([[1.0, 1.0]]).repeat(batch_size, 1).to(self.device)
    
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


class DagLayer(nn.Module):
    """Directed Acyclic Graph layer from mask.py"""
    def __init__(self, in_features, out_features, i=False, bias=False, initial=True):
        super(DagLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.i = i
        self.a = torch.zeros(out_features, out_features)
        
        if initial:
            # Modified initial DAG matrix for Reacher physical properties
            # 0: Joint Angles -> 2: End Effector Position
            # 1: Joint Velocities -> 2: End Effector Position
            # 3: Target Position (independent factor)
            # 4: Arm Lengths -> 0: Joint Angles, 2: End Effector Position
            # 5: Joint Damping -> 1: Joint Velocities
            # 6: Arm Mass/Inertia -> 1: Joint Velocities
            
            # Default initialization (can be adjusted based on Reacher physics)
            self.a[0][2] = 1  # Joint Angles -> End Effector
            self.a[1][2] = 1  # Joint Velocities -> End Effector
            self.a[4][0] = 1  # Arm Lengths -> Joint Angles
            self.a[4][2] = 1  # Arm Lengths -> End Effector
            self.a[5][1] = 1  # Joint Damping -> Joint Velocities
            self.a[6][1] = 1  # Arm Mass -> Joint Velocities

        self.A = nn.Parameter(self.a)
        
        self.b = torch.eye(out_features)
        self.B = nn.Parameter(self.b)
        
        self.I = nn.Parameter(torch.eye(out_features))
        self.I.requires_grad = False
        
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
        x = F.linear(x, torch.inverse(self.I - self.A.t()), self.bias)
       
        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
        return x, v


class MaskLayer(nn.Module):
    """Mask layer from mask.py"""
    def __init__(self, z_dim, concept=4, z2_dim=4):
        super(MaskLayer, self).__init__()
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
        # Add additional networks for more causal factors if needed
        self.net5 = nn.Sequential(
            nn.Linear(z2_dim, 32),
            nn.ELU(),
            nn.Linear(32, z2_dim),
        )
        self.net6 = nn.Sequential(
            nn.Linear(z2_dim, 32),
            nn.ELU(),
            nn.Linear(32, z2_dim),
        )
        self.net7 = nn.Sequential(
            nn.Linear(z2_dim, 32),
            nn.ELU(),
            nn.Linear(32, z2_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(z2_dim, 32),
            nn.ELU(),
            nn.Linear(32, z2_dim),
        )
    
    def mix(self, z):
        zy = z.view(-1, self.concept*self.z2_dim)
        if self.z2_dim == 1:
            zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
            if self.concept == 7:  # For 7 concepts (including all reacher properties)
                zy1, zy2, zy3, zy4, zy5, zy6, zy7 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3], zy[:, 4], zy[:, 5], zy[:, 6]
            elif self.concept == 4:  # Legacy 4 concepts
                zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
            elif self.concept == 3:  # Legacy 3 concepts
                zy1, zy2, zy3 = zy[:, 0], zy[:, 1], zy[:, 2]
        else:
            if self.concept == 7:  # For 7 concepts (including all reacher properties)
                zy1, zy2, zy3, zy4, zy5, zy6, zy7 = torch.split(zy, self.z_dim//self.concept, dim=1)
            elif self.concept == 4:  # Legacy 4 concepts
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim=1)
            elif self.concept == 3:  # Legacy 3 concepts
                zy1, zy2, zy3 = torch.split(zy, self.z_dim//self.concept, dim=1)
        
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        
        if self.concept >= 4:
            rx4 = self.net4(zy4)
        
        if self.concept >= 5:
            rx5 = self.net5(zy5)
            
        if self.concept >= 6:
            rx6 = self.net6(zy6)
            
        if self.concept >= 7:
            rx7 = self.net7(zy7)
        
        # Combine based on number of concepts
        if self.concept == 7:
            h = torch.cat((rx1, rx2, rx3, rx4, rx5, rx6, rx7), dim=1)
        elif self.concept == 4:
            h = torch.cat((rx1, rx2, rx3, rx4), dim=1)
        elif self.concept == 3:
            h = torch.cat((rx1, rx2, rx3), dim=1)
        
        return h


class Attention(nn.Module):
    """Attention mechanism from mask.py"""
    def __init__(self, in_features, bias=False):
        super(Attention, self).__init__()
        self.M = nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features, in_features), mean=0, std=1))
        self.sigmd = torch.nn.Sigmoid()
    
    def attention(self, z, e):
        a = z.matmul(self.M).matmul(e.permute(0, 2, 1))
        a = self.sigmd(a)
        A = torch.softmax(a, dim=1)
        e = torch.matmul(A, e)
        return e, A


class CausalVAE(nn.Module):
    """CausalVAE implementation with RSSM-compatible interface"""
    def __init__(
        self,
        obs_space,
        feature_dim,
        stoch_dim,
        z1_dim=4,
        z2_dim=4,
        hidden_dim=200,
        inference=False, 
        alpha=0.3, 
        beta=1, 
        initial=True,
        config=None
    ):
        super(CausalVAE, self).__init__()
        self.obs_space = obs_space
        self.feature_dim = feature_dim
        self.z_dim = stoch_dim   # the full dimension
        self.z1_dim = z1_dim     # number of concepts (causal factors)
        self.z2_dim = z2_dim     # the dimension of each concept
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        self.config = config
        
        # Set device from config if available
        self.device = config.device if config is not None and hasattr(config, 'device') else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Add physical properties extractor for Reacher environment
        self.physical_properties = ReacherPhysicalProperties(config, device=self.device)
        
        # Scale parameter from mask_vae_pendulum.py - adjust for Reacher
        # Scale ranges for joint angles, velocities, positions, etc.
        self.scale = np.array([
            [0, np.pi],           # Joint angle 1
            [0, np.pi],           # Joint angle 2
            [-1, 1],              # End effector x
            [-1, 1],              # End effector y
            [0, 1],               # Target position x
            [0, 1],               # Target position y
            [0.1, 0.3]            # Physical parameter scale
        ])
        
        # Causal structure components from mask_vae_pendulum.py and mask.py
        self.dag = DagLayer(self.z1_dim, self.z1_dim, i=inference, initial=initial)
        self.attn = Attention(self.z2_dim)
        self.mask_z = MaskLayer(self.z_dim, concept=self.z1_dim, z2_dim=self.z2_dim)
        self.mask_u = MaskLayer(self.z1_dim, z2_dim=1)
        
        # Using encoder and decoder from networks.py
        if config is not None:
            # Use MultiEncoder and MultiDecoder
            self.shapes = {k: tuple(v.shape) for k, v in self.obs_space.spaces.items()}
            
            self.encoder = networks.MultiEncoder(self.shapes, **config.encoder)
            self.embed_size = self.encoder.outdim
            
            # Define a mapping layer from encoder output to latent space
            self.enc_to_latent = nn.Sequential(
                nn.Linear(self.embed_size, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 2 * stoch_dim)
            )
            
            # Define decoder for reconstruction
            self.decoder = networks.MultiDecoder(
                stoch_dim, self.shapes, **config.decoder
            )
        else:
            # Fallback to simple MLP encoder/decoder
            self.encoder = networks.MLP(
                feature_dim,
                None,
                2,  # layers
                hidden_dim,
                act="SiLU",
                norm=True,
                dist=None  # We'll handle distribution ourselves
            )
            
            self.enc_to_latent = nn.Linear(hidden_dim, 2 * stoch_dim)
            
            # Simple decoder
            self.decoder = networks.MLP(
                stoch_dim,
                feature_dim,
                2,  # layers
                hidden_dim,
                act="SiLU",
                norm=True,
                dist="normal"
            )
            
        # Add transition model for RSSM compatibility
        self.transition = nn.Sequential(
            nn.Linear(stoch_dim + config.num_actions, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )
        
        # Register deterministic state size for RSSM compatibility
        self._deter = stoch_dim  # Using stoch_dim as deter_dim for simplicity
        self._stoch = stoch_dim
        self._discrete = config.dyn_discrete
        self._num_actions = config.num_actions if config is not None else None
        self._device = self.device
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        if isinstance(self.encoder, networks.MultiEncoder):
            # For MultiEncoder
            h = self.encoder(x)
            params = self.enc_to_latent(h)
        else:
            # For MLP
            h = self.encoder(x)
            params = self.enc_to_latent(h)
        
        mean, log_std = torch.chunk(params, 2, dim=-1)
        std = torch.exp(log_std)
        
        # Reshape to match the causal structure requirements
        q_m = mean.reshape([mean.size()[0], self.z1_dim, self.z2_dim])
        q_v = std.reshape([std.size()[0], self.z1_dim, self.z2_dim])
        
        return q_m, q_v
    
    def apply_causal_structure(self, q_m, q_v, label, mask=None, adj=None, sample=False, lambdav=0.001):
        """Apply causal structure to latent variables"""
        # Calculate DAG structure
        decode_m, decode_v = self.dag.calculate_dag(q_m.to(self.device), torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(self.device))
        decode_m, decode_v = decode_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]), decode_v
        
        # Apply mask if specified
        if not sample and mask is not None and mask < 2:
            z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(self.device) * adj
            decode_m[:, mask, :] = z_mask[:, mask, :]
            decode_v[:, mask, :] = z_mask[:, mask, :]
        
        # Apply masking and attention mechanisms
        m_zm, m_zv = self.dag.mask_z(decode_m.to(self.device)).reshape([q_m.size()[0], self.z1_dim, self.z2_dim]), decode_v.reshape([q_m.size()[0], self.z1_dim, self.z2_dim])
        m_u = self.dag.mask_u(label.to(self.device))
        
        f_z = self.mask_z.mix(m_zm).reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(self.device)
        e_tilde = self.attn.attention(decode_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(self.device), q_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(self.device))[0]
        
        if mask is not None and mask < 2:
            z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(self.device) * adj
            e_tilde[:, mask, :] = z_mask[:, mask, :]
        
        f_z1 = f_z + e_tilde
        
        if mask is not None and mask == 2:
            z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(self.device) * adj
            f_z1[:, mask, :] = z_mask[:, mask, :]
            m_zv[:, mask, :] = z_mask[:, mask, :]
        
        if mask is not None and mask == 3:
            z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(self.device) * adj
            f_z1[:, mask, :] = z_mask[:, mask, :]
            m_zv[:, mask, :] = z_mask[:, mask, :]
        
        g_u = self.mask_u.mix(m_u).to(self.device)
        
        # Sample from the distribution
        z_given_dag = conditional_sample_gaussian(f_z1, m_zv * lambdav, self.device)
        
        # Reshape for decoder
        z_reshaped = z_given_dag.reshape([z_given_dag.size()[0], self.z_dim])
        
        return z_reshaped, g_u, q_m, q_v, decode_m, decode_v, f_z1, m_zv
    
    def decode(self, z, label=None):
        """Decode latent variables to reconstructions"""
        if isinstance(self.decoder, networks.MultiDecoder):
            # For MultiDecoder
            return self.decoder(z)
        else:
            # For MLP decoder
            return self.decoder(z)
    
    def forward(self, x, label=None, mask=None, adj=None, sample=False, alpha=None, beta=None, lambdav=0.001):
        """Forward pass through the Causal VAE model"""
        # If label is not provided, extract it from observations
        if label is None and isinstance(x, dict):
            label = self.physical_properties(x)
        
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.beta
        
        # Encode input to latent distribution
        q_m, q_v = self.encode(x)
        
        # Apply causal structure and get latent samples
        z, g_u, q_m_orig, q_v_orig, decode_m, decode_v, f_z1, m_zv = self.apply_causal_structure(
            q_m, q_v, label, mask, adj, sample, lambdav
        )
        
        # Decode latent samples to reconstruction
        recon = self.decode(z, label)
        
        # Calculate losses
        if isinstance(recon, dict):
            # For MultiDecoder output
            recon_loss = 0
            for key, dist in recon.items():
                if key in x:
                    recon_loss += -torch.mean(dist.log_prob(x[key]))
        else:
            # For MLP decoder output
            recon_loss = -torch.mean(recon.log_prob(x))
        
        # KL divergence calculation similar to mask_vae_pendulum.py
        p_m, p_v = torch.zeros(q_m.size()), torch.ones(q_m.size())
        cp_m, cp_v = condition_prior(self.scale, label, self.z2_dim, self.device)
        cp_v = torch.ones([q_m.size()[0], self.z1_dim, self.z2_dim]).to(self.device)
        
        kl = alpha * kl_normal(
            q_m_orig.view(-1, self.z_dim).to(self.device), 
            q_v_orig.view(-1, self.z_dim).to(self.device), 
            p_m.view(-1, self.z_dim).to(self.device), 
            p_v.view(-1, self.z_dim).to(self.device)
        )
        
        for i in range(self.z1_dim):
            kl = kl + beta * kl_normal(
                decode_m[:, i, :].to(self.device), 
                cp_v[:, i, :].to(self.device),
                cp_m[:, i, :].to(self.device), 
                cp_v[:, i, :].to(self.device)
            )
        
        kl = torch.mean(kl)
        
        mask_kl = torch.zeros(1).to(self.device)
        for i in range(self.z1_dim):
            mask_kl = mask_kl + kl_normal(
                f_z1[:, i, :].to(self.device), 
                cp_v[:, i, :].to(self.device),
                cp_m[:, i, :].to(self.device), 
                cp_v[:, i, :].to(self.device)
            )
        
        u_loss = torch.nn.MSELoss()
        mask_l = torch.mean(mask_kl) + u_loss(g_u, label.float().to(self.device))
        
        # Total loss
        total_loss = recon_loss + kl + mask_l
        
        return {
            'z': z,
            'recon': recon,
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl,
            'mask_loss': mask_l,
            'g_u': g_u
        }
    
    # RSSM compatibility methods
    def initial(self, batch_size):
        """Create initial state dict similar to RSSM"""
        if self._discrete:
            state = dict(
                logit=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                stoch=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                deter=torch.zeros([batch_size, self._deter], device=self._device),
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch], device=self._device),
                std=torch.ones([batch_size, self._stoch], device=self._device),
                stoch=torch.zeros([batch_size, self._stoch], device=self._device),
                deter=torch.zeros([batch_size, self._deter], device=self._device),
            )
        return state
    
    def observe(self, embed, action, is_first, state=None):
        """Process a sequence of observations, similar to RSSM observe"""
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        
        # Create batch of physical properties for CausalVAE
        batch_size = embed.shape[1]
        
        # Use obs_step to process each timestep
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first, None  # We'll handle labels in obs_step
            ),
            (action, embed, is_first),
            (state, state),
        )
        
        # (time, batch, ...) -> (batch, time, ...)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        
        return post, prior
    
    def obs_step(self, prev_state, prev_action, embed, is_first, label=None):
        """Update state based on observation, action, and previous state"""
        # Initialize all prev_state
        if prev_state is None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros(
                (len(is_first), self._num_actions), device=self._device
            )
        # Overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        # Compute prior by imagining next state from previous
        prior = self.img_step(prev_state, prev_action)
        
        # Create dummy physical properties if not provided
        # In practice, this should be constructed from appropriate observations
        if label is None:
            batch_size = embed.shape[0]
            # Just create a placeholder - in real usage this should come from observation data
            label = torch.zeros(batch_size, self.z1_dim).to(self._device)
        
        # Convert embed to observation format expected by CausalVAE
        # We assume embed is directly usable as an embedding of the image
        # In a real scenario, we might need proper structure based on the environment
        x = {"image": embed}
        
        # Convert to input format
        q_m, q_v = self.encode(x)
        
        # Apply causal structure to get latent z
        z, _, _, _, _, _, _, _ = self.apply_causal_structure(q_m, q_v, label)
        
        # Construct posterior state
        post = {
            "mean": q_m.reshape(embed.shape[0], -1),
            "std": q_v.reshape(embed.shape[0], -1),
            "stoch": z,
            "deter": prior["deter"],  # Use prior's deterministic part
        }
        
        return post, prior
    
    def img_step(self, prev_state, prev_action, sample=True):
        """Predict next state based on previous state and action using causal mechanisms"""
        # Extract previous stochastic state
        prev_stoch = prev_state["stoch"]
        
        # Reshape to causal format if needed
        batch_size = prev_stoch.shape[0]
        z_causal = prev_stoch.reshape(batch_size, self.z1_dim, self.z2_dim)
        
        # Create dummy label (or use action as intervention)
        dummy_labels = torch.zeros(batch_size, self.z1_dim).to(self._device)
        
        # Use your causal mechanisms to predict the next state
        # This could involve setting up the right conditions for your DAG
        # and applying the causal influence of the action
        decode_m, decode_v = self.dag.calculate_dag(z_causal, torch.ones_like(z_causal))
        
        # Apply your causal propagation
        m_zm = self.dag.mask_z(decode_m).reshape([batch_size, self.z1_dim, self.z2_dim])
        f_z = self.mask_z.mix(m_zm).reshape([batch_size, self.z1_dim, self.z2_dim])
        
        # Sample next state
        next_z = conditional_sample_gaussian(f_z, decode_v, self.device) if sample else f_z
        next_z_flat = next_z.reshape(batch_size, self.z_dim)
        
        # Create the prior state dictionary in RSSM format
        prior = {
            "mean": f_z.reshape(batch_size, self.z_dim),
            "std": decode_v.reshape(batch_size, self.z_dim),
            "stoch": next_z_flat,
            "deter": prev_state["deter"],  # Maintain deterministic state
        }
        
        return prior
    
    def get_feat(self, state):
        """Extract features from state dict"""
        return state["stoch"]
    
    def get_dist(self, state, dtype=None):
        """Get distribution from state"""
        return tools.ContDist(
            torchd.independent.Independent(
                torchd.normal.Normal(state["mean"], state["std"]), 1
            )
        )
    
    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        """Compute KL divergence loss between posterior and prior"""
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # Apply free bits
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss
    
    def intervene(self, z, index, value):
        """Perform intervention on a specific latent variable"""
        # Reshape to causal format
        z_causal = z.reshape([-1, self.z1_dim, self.z2_dim])
        
        # Create intervention
        z_intervened = z_causal.clone()
        z_intervened[:, index, :] = value
        
        # Apply causal effect propagation
        batch_size = z_intervened.size(0)
        dummy_label = torch.zeros(batch_size, self.z1_dim).to(self.device)
        
        # Process through causal mechanism (without sampling)
        decoded_m, decoded_v = self.dag.calculate_dag(
            z_intervened, 
            torch.ones_like(z_intervened)
        )
        
        m_zm = self.dag.mask_z(decoded_m).reshape([batch_size, self.z1_dim, self.z2_dim])
        f_z = self.mask_z.mix(m_zm).reshape([batch_size, self.z1_dim, self.z2_dim])
        
        # Reshape for decoder
        z_final = f_z.reshape([batch_size, self.z_dim])
        
        return z_final
    
    def counterfactual(self, x, label, index, value):
        """Generate counterfactual by intervening on a specific latent"""
        # If label is not provided, extract it from observations
        if label is None and isinstance(x, dict):
            label = self.physical_properties(x)
            
        # Encode to latent space
        q_m, q_v = self.encode(x)
        
        # Sample from the posterior
        z = conditional_sample_gaussian(q_m, q_v, self.device)
        z_reshaped = z.reshape([z.size()[0], self.z_dim])
        
        # Perform intervention
        z_cf = self.intervene(z_reshaped, index, value)
        
        # Decode counterfactual
        x_cf = self.decode(z_cf, label)
        
        return x_cf, z_cf


class CausalVAE_WorldModel(nn.Module):
    """Wrapper to integrate CausalVAE into a world model framework"""
    def __init__(self, obs_space, act_space, step, config):
        super(CausalVAE_WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        
        # Create CausalVAE
        self.dynamics = CausalVAE(
            obs_space = obs_space,
            feature_dim=None,  # Will be determined by encoder
            stoch_dim=config.dyn_stoch,
            z1_dim=getattr(config, 'causal_factors', 4),
            z2_dim=getattr(config, 'causal_dim', 4),
            hidden_dim=config.dyn_hidden,
            inference=False,
            alpha=getattr(config, 'kl_alpha', 0.3),
            beta=getattr(config, 'kl_beta', 1.0),
            config=config
        )
        
        # Define heads for auxiliary predictions (reward, continuation, etc.)
        self.heads = nn.ModuleDict()
        feat_size = config.dyn_stoch
        
        self.encoder = self.dynamics.encoder
        self.decoder = self.dynamics.decoder
        
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
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
            device=config.device,
            name="Cont",
        )
        
        # Create optimizer
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
        
        # Store loss scales
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        
        # Register intervention parameters
        self.register_buffer("intervention_index", torch.tensor(-1, device=config.device))
        self.register_buffer("intervention_value", torch.tensor(0.0, device=config.device))
        
        # Print model size
        print(
            f"CausalVAE World Model has {sum(param.numel() for param in self.parameters())} variables."
        )
    
    def _train(self, data):
        """Training step"""
        data = self.preprocess(data)
        
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # Extract physical properties from observations
                physical_properties = self.dynamics.physical_properties(data)
                
                # Forward pass through CausalVAE
                vae_out = self.dynamics(data, physical_properties)
                
                # Get latent representation
                z = vae_out['z']
                
                # Compute predictions from heads
                preds = {}
                for name, head in self.heads.items():
                    pred = head(z)
                    preds[name] = pred
                
                # Calculate losses
                losses = {}
                losses['vae'] = vae_out['total_loss']
                
                for name, pred in preds.items():
                    if name in data:
                        loss = -pred.log_prob(data[name])
                        losses[name] = loss.mean()
                
                # Scale losses
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                
                # Combine all losses
                model_loss = sum(scaled.values())
            
            # Optimization step
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())
        
        # Collect metrics
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["recon_loss"] = to_np(vae_out['recon_loss'])
        metrics["kl_loss"] = to_np(vae_out['kl_loss'])
        metrics["mask_loss"] = to_np(vae_out['mask_loss'])
        
        # Context for policy learning
        context = dict(
            feat=z,
            vae_loss=vae_out['total_loss'],
        )
        
        return z, context, metrics
    
    def preprocess(self, obs):
        """Preprocess observations"""
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        if "image" in obs:
            obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = obs["discount"].unsqueeze(-1)
        if "is_terminal" in obs:
            obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
            
        return obs
    
    def intervene(self, index, value):
        """Perform an intervention on a specific latent variable"""
        self.intervention_index = torch.tensor(index, device=self._config.device)
        self.intervention_value = torch.tensor(value, device=self._config.device)
        return self
    
    def remove_intervention(self):
        """Remove any active intervention"""
        self.intervention_index = torch.tensor(-1, device=self._config.device)
        return self
    
    def get_causal_mask(self):
        """Get the current causal adjacency matrix"""
        return self.dynamics.dag.A.detach().cpu().numpy()