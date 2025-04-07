import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class CausalMask(nn.Module):
    """Learns and applies a causal mask between variables."""
    def __init__(self, num_latents, hidden_dim=64, threshold=0.1, temperature=0.1, device="cuda"):
        super(CausalMask, self).__init__()
        self.num_latents = num_latents
        self.temperature = temperature
        self.threshold = threshold
        self.device = device
        
        # Parameters for causal adjacency matrix (upper triangular for DAG constraint)
        self.log_weight = nn.Parameter(torch.zeros(num_latents, num_latents, device=device))
        
        # Mutual information estimator networks
        self.mi_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_latents * (num_latents - 1) // 2)
        ])
        
    def forward(self, hard=False):
        """Get the causal adjacency matrix."""
        # Add negative diagonal values to ensure no self-loops
        diagonal_mask = -10.0 * torch.eye(self.num_latents, device=self.device)
        
        # Create upper triangular mask for DAG constraint (no cycles)
        upper_mask = torch.triu(torch.ones(self.num_latents, self.num_latents, device=self.device), diagonal=1)
        
        # Apply masks to ensure proper structure
        log_weight = self.log_weight * upper_mask + diagonal_mask
        
        # Apply temperature for better gradient flow
        adjacency = torch.sigmoid(log_weight / self.temperature)
        
        if hard:
            # For inference, use hard thresholding
            hard_adjacency = (adjacency > self.threshold).float()
            return hard_adjacency
        
        # For training, use soft adjacency
        return adjacency
    
    def calculate_sparsity_loss(self):
        """Calculate L1 regularization for sparsity."""
        adjacency = torch.sigmoid(self.log_weight)
        return adjacency.abs().sum()
    
    def calculate_mi_loss(self, latents):
        """Estimate mutual information between pairs of latents."""
        batch_size = latents.shape[0]
        total_mi = 0.0
        idx = 0
        
        for i in range(self.num_latents):
            for j in range(i+1, self.num_latents):
                # Take pairs of latent variables
                pair = torch.cat([latents[:, i:i+1], latents[:, j:j+1]], dim=1)
                
                # Estimate MI using the corresponding network
                mi_estimate = self.mi_estimator[idx](pair).mean()
                
                # Weight by causal adjacency
                adjacency_ij = torch.sigmoid(self.log_weight[i, j])
                total_mi += adjacency_ij * mi_estimate
                
                idx += 1
                
        return total_mi
    
    def calculate_dag_loss(self):
        """Calculate loss to enforce DAG constraint using trace exponential method."""
        adjacency = torch.sigmoid(self.log_weight)
        
        # h(A) = tr(expm(A ⊙ A)) - d
        # For DAG, we want h(A) = 0
        adjacency_squared = adjacency * adjacency
        
        # Using matrix multiplication for the expm calculation
        # Identity matrix
        identity = torch.eye(self.num_latents, device=self.device)
        
        # Power series approximation of matrix exponential
        expm_approx = identity
        matrix_power = identity
        factorial = 1.0
        
        # Use 10 terms of the power series for approximation
        for i in range(1, 10):
            factorial *= i
            matrix_power = matrix_power @ adjacency_squared
            expm_approx += matrix_power / factorial
        
        # Calculate trace
        trace_expm = torch.trace(expm_approx)
        
        # DAG loss: (tr(expm(A ⊙ A)) - d)^2
        dag_loss = (trace_expm - self.num_latents)**2
        
        return dag_loss


class CausalVAE(nn.Module):
    """Causal VAE implementation for Dreamer's latent space."""
    def __init__(self, feature_dim, stoch_dim, hidden_dim=200, causal_hidden_dim=64, device="cuda"):
        super(CausalVAE, self).__init__()
        self.feature_dim = feature_dim
        self.stoch_dim = stoch_dim
        self.device = device
        
        # Encoder to get latent distribution parameters (mean and std)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2 * stoch_dim)  # Mean and log_std
        )
        
        # Causal mask for structured latent space
        self.causal_mask = CausalMask(stoch_dim, causal_hidden_dim, device=device)
        
        # Causal mechanism functions (one per latent dimension)
        self.mechanisms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(stoch_dim, hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(stoch_dim)
        ])
        
        # Decoder to reconstruct features from latent space
        self.decoder = nn.Sequential(
            nn.Linear(stoch_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Hyperparameters for loss weighting
        self.sparsity_weight = 0.1
        self.mi_weight = 0.01
        self.dag_weight = 1.0
        
    def encode(self, features):
        """Encode features to parameters of latent distribution."""
        params = self.encoder(features)
        mean, log_std = torch.chunk(params, 2, dim=-1)
        std = torch.exp(log_std)
        return mean, std
    
    def sample(self, mean, std):
        """Sample from the latent distribution with reparameterization trick."""
        eps = torch.randn_like(mean)
        z = mean + std * eps
        return z
    
    def apply_causal_structure(self, z):
        """Apply causal structure to latent variables."""
        batch_size = z.shape[0]
        
        # Get causal adjacency matrix (during training: soft, during inference: hard)
        adjacency = self.causal_mask(hard=not self.training)
        
        # Initialize output with zeros
        z_causal = torch.zeros_like(z)
        
        # For each latent dimension
        for i in range(self.stoch_dim):
            # Get parent indices (where adjacency[j, i] = 1)
            # During training, this is weighted by the soft adjacency
            parents = torch.arange(self.stoch_dim, device=self.device)
            
            # Build input for this mechanism by combining parents
            # Mask out non-parents using adjacency matrix
            z_parents = z * adjacency[:, i].unsqueeze(0)
            
            # Apply causal mechanism
            z_causal[:, i:i+1] = self.mechanisms[i](z_parents)
        
        return z_causal
    
    def decode(self, z):
        """Decode latent variables to features."""
        return self.decoder(z)
    
    def forward(self, features):
        """Forward pass through the Causal VAE."""
        # Encode features to latent distribution
        mean, std = self.encode(features)
        
        # Sample from latent distribution
        z = self.sample(mean, std)
        
        # Apply causal structure
        z_causal = self.apply_causal_structure(z)
        
        # Decode back to features
        features_recon = self.decode(z_causal)
        
        # Calculate losses
        recon_loss = F.mse_loss(features_recon, features)
        
        # KL divergence for each variable
        kl_loss = 0.5 * torch.sum(mean.pow(2) + std.pow(2) - torch.log(std.pow(2)) - 1, dim=-1).mean()
        
        # Sparsity loss to encourage few causal connections
        sparsity_loss = self.causal_mask.calculate_sparsity_loss()
        
        # Mutual information loss to encourage meaningful connections
        mi_loss = self.causal_mask.calculate_mi_loss(z)
        
        # DAG loss to ensure proper causal graph structure
        dag_loss = self.causal_mask.calculate_dag_loss()
        
        # Combine losses
        total_loss = recon_loss + kl_loss + \
                    self.sparsity_weight * sparsity_loss + \
                    self.mi_weight * mi_loss + \
                    self.dag_weight * dag_loss
        
        return {
            'z': z_causal,
            'mean': mean,
            'std': std,
            'features_recon': features_recon,
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'sparsity_loss': sparsity_loss,
            'mi_loss': mi_loss,
            'dag_loss': dag_loss
        }
    
    def intervene(self, z, index, value):
        """Perform intervention on a specific latent variable."""
        z_intervened = z.clone()
        z_intervened[:, index] = value
        
        # Calculate effects of intervention through causal structure
        z_causal = self.apply_causal_structure(z_intervened)
        
        return z_causal
    
    def counterfactual(self, features, index, value):
        """Generate counterfactual by intervening on a specific latent."""
        # Encode to latent space
        mean, std = self.encode(features)
        z = self.sample(mean, std)
        
        # Perform intervention
        z_cf = self.intervene(z, index, value)
        
        # Decode counterfactual
        features_cf = self.decode(z_cf)
        
        return features_cf, z_cf


class CausalRSSM(nn.Module):
    """RSSM with a CausalVAE for the stochastic component."""
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        causal_hidden=64,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super(CausalRSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        self._act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device
        
        # Create input layers (same as original RSSM)
        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(self._act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        
        # GRU Cell (same as original RSSM)
        self._cell = networks.GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)
        
        # Image output layers (same as original RSSM)
        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(self._act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)
        
        # Observation output layers (same as original RSSM)
        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(self._act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)
        
        # Stats layers (similar to original RSSM)
        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        
        # Initial state parameter (same as original RSSM)
        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )
        
        # Add CausalVAE for stochastic latent space
        # This is what makes this RSSM causal
        self.causal_vae = CausalVAE(
            feature_dim=self._hidden,
            stoch_dim=self._stoch,
            hidden_dim=self._hidden,
            causal_hidden_dim=causal_hidden,
            device=self._device
        )
        
        # Buffer to store latent values for tracking causal effects
        self.register_buffer("intervention_index", torch.tensor(-1, device=self._device))
        self.register_buffer("intervention_value", torch.tensor(0.0, device=self._device))
    
    def initial(self, batch_size):
        """Return initial state (same as original RSSM)."""
        deter = torch.zeros(batch_size, self._deter, device=self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                stoch=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch], device=self._device),
                std=torch.zeros([batch_size, self._stoch], device=self._device),
                stoch=torch.zeros([batch_size, self._stoch], device=self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)
    
    def observe(self, embed, action, is_first, state=None):
        """Process observation sequence (similar to original RSSM)."""
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )
        
        # (time, batch, stoch) -> (batch, time, stoch)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior
    
    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """Single step with observation (modified to use CausalVAE)."""
        # Initialize or reset state if needed
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros(
                (len(is_first), self._num_actions), device=self._device
            )
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
        
        # Compute prior (similar to original RSSM)
        prior = self.img_step(prev_state, prev_action)
        
        # Combine deterministic state and embedding
        x = torch.cat([prior["deter"], embed], -1)
        x = self._obs_out_layers(x)
        
        # Use CausalVAE for posterior computation
        if self._discrete:
            # For discrete case - keep original approach
            stats = self._suff_stats_layer("obs", x)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
            post = {"stoch": stoch, "deter": prior["deter"], **stats}
        else:
            # For continuous case - use CausalVAE
            mean, std = self.encode(x)
            
            # Sample stochastic state
            if sample:
                noise = torch.randn_like(mean)
                stoch = mean + std * noise
            else:
                stoch = mean
            
            # Apply causal structure if not in intervention mode
            if self.intervention_index >= 0:
                # Apply intervention
                stoch = stoch.clone()
                stoch[:, self.intervention_index] = self.intervention_value
            
            # Apply causal structure
            stoch_causal = self.causal_vae.apply_causal_structure(stoch)
            
            # Create posterior state
            post = {
                "mean": mean,
                "std": std,
                "stoch": stoch_causal,
                "deter": prior["deter"]
            }
        
        return post, prior
    
    def img_step(self, prev_state, prev_action, sample=True):
        """Single step without observation (modified to use CausalVAE)."""
        # Get stochastic state from previous state
        prev_stoch = prev_state["stoch"]
        
        # Reshape for discrete case
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.reshape(shape)
        
        # Combine stochastic state and action
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self._img_in_layers(x)
        
        # Update deterministic state via GRU
        deter = prev_state["deter"]
        for _ in range(self._rec_depth):
            deter = self._cell(x, [deter])[1][0]
        
        # Compute prior distribution parameters
        x = self._img_out_layers(deter)
        
        # Use CausalVAE for prior computation
        if self._discrete:
            # For discrete case - keep original approach
            stats = self._suff_stats_layer("ims", x)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
            prior = {"stoch": stoch, "deter": deter, **stats}
        else:
            # For continuous case - use CausalVAE
            mean, std = self.encode(x)
            
            # Sample stochastic state
            if sample:
                noise = torch.randn_like(mean)
                stoch = mean + std * noise
            else:
                stoch = mean
            
            # Apply causal structure if not in intervention mode
            if self.intervention_index >= 0:
                # Apply intervention
                stoch = stoch.clone()
                stoch[:, self.intervention_index] = self.intervention_value
            
            # Apply causal structure
            stoch_causal = self.causal_vae.apply_causal_structure(stoch)
            
            # Create prior state
            prior = {
                "mean": mean,
                "std": std,
                "stoch": stoch_causal,
                "deter": deter
            }
        
        return prior
    
    def encode(self, features):
        """Encode features to latent distribution parameters."""
        stats = self._suff_stats_layer("ims", features)
        return stats["mean"], stats["std"]
    
    def get_stoch(self, deter):
        """Get stochastic state from deterministic state."""
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()
    
    def get_feat(self, state):
        """Extract features from state (same as original RSSM)."""
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)
    
    def get_dist(self, state, dtype=None):
        """Get distribution from state (same as original RSSM)."""
        if self._discrete:
            logit = state["logit"]
            dist = tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio)
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torch.distributions.Normal(mean, std)
            )
        return dist
    
    def _suff_stats_layer(self, name, x):
        """Compute sufficient statistics (modified to use CausalVAE for continuous case)."""
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.chunk(x, 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}
    
    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        """Compute KL loss (same as original RSSM)."""
        kld = torch.distributions.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post),
            dist(sg(prior)) if self._discrete else dist(sg(prior)),
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post)),
            dist(prior) if self._discrete else dist(prior),
        )
        # Clip losses to avoid extreme values
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        
        # Add CausalVAE specific losses (sparsity and DAG constraints)
        if hasattr(self, 'causal_vae'):
            # Get latent samples from posterior
            latents = post["stoch"]
            
            # Calculate causal structure losses
            sparsity_loss = self.causal_vae.causal_mask.calculate_sparsity_loss()
            dag_loss = self.causal_vae.causal_mask.calculate_dag_loss()
            
            # Add to total loss with small weights
            sparsity_weight = 0.01
            dag_weight = 0.1
            
            loss = loss + sparsity_weight * sparsity_loss + dag_weight * dag_loss
        
        return loss, value, dyn_loss, rep_loss
    
    def intervene(self, index, value):
        """Set up an intervention on a specific latent variable."""
        self.intervention_index = torch.tensor(index, device=self._device)
        self.intervention_value = torch.tensor(value, device=self._device)
        return self
    
    def remove_intervention(self):
        """Remove any active intervention."""
        self.intervention_index = torch.tensor(-1, device=self._device)
        return self
    
    def get_causal_mask(self):
        """Get the current causal mask matrix."""
        if hasattr(self, 'causal_vae'):
            return self.causal_vae.causal_mask(hard=True).detach().cpu().numpy()
        return None


# Wrapper class to use with Dreamer
class CausalVAEDreamer(nn.Module):
    """Wrapper to integrate CausalRSSM into Dreamer."""
    def __init__(self, obs_space, act_space, step, config):
        super(CausalVAEDreamer, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        
        # Create encoder (same as original WorldModel)
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        
        # Create CausalRSSM instead of standard RSSM
        self.dynamics = CausalRSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.get('causal_hidden', 64),  # New parameter for causal networks
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
            config.device,
        )
        
        # Create heads (same as original WorldModel)
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
        
        # Filter which heads receive gradients
        for name in config.grad_heads:
            assert name in self.heads, name
            
        # Create optimizer (same as original WorldModel)
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
        
        # Store loss scales (same as original WorldModel)
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        
        # Print model size
        print(
            f"CausalVAE Dreamer has {sum(param.numel() for param in self.parameters())} variables."
        )
    
    def _train(self, data):
        """Training step (modified to include CausalVAE specific losses)."""
        data = self.preprocess(data)
        
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # Compute embeddings and latent states
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                
                # Compute KL divergence loss (includes causal structure losses)
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                
                # Compute prediction losses
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
                
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    losses[name] = loss
                
                # Scale losses
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                
                # Combine all losses
                model_loss = sum(scaled.values()) + kl_loss
                
                # Add causal structure specific losses
                if hasattr(self.dynamics, 'causal_vae'):
                    # Get latent samples from posterior
                    latents = post["stoch"]
                    
                    # Calculate causal structure losses
                    sparsity_loss = self.dynamics.causal_vae.causal_mask.calculate_sparsity_loss()
                    mi_loss = self.dynamics.causal_vae.causal_mask.calculate_mi_loss(latents)
                    dag_loss = self.dynamics.causal_vae.causal_mask.calculate_dag_loss()
                    
                    # Add to total loss with appropriate weights
                    causal_weight = self._config.get('causal_weight', 0.1)
                    sparsity_weight = self._config.get('sparsity_weight', 0.01)
                    mi_weight = self._config.get('mi_weight', 0.001)
                    dag_weight = self._config.get('dag_weight', 0.1)
                    
                    causal_loss = sparsity_weight * sparsity_loss + \
                                 mi_weight * mi_loss + \
                                 dag_weight * dag_loss
                    
                    # Add causal loss to total loss
                    model_loss = model_loss + causal_weight * causal_loss
                
            # Optimization step
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())
        
        # Collect metrics
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        
        # Add causal structure metrics
        if hasattr(self.dynamics, 'causal_vae'):
            metrics["sparsity_loss"] = to_np(sparsity_loss)
            metrics["mi_loss"] = to_np(mi_loss)
            metrics["dag_loss"] = to_np(dag_loss)
            
            # Get adjacency matrix sparsity
            adjacency = self.dynamics.causal_vae.causal_mask(hard=True)
            metrics["causal_sparsity"] = to_np(adjacency.sum() / (self._config.dyn_stoch**2))
        
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            
        # Context for policy learning
        context = dict(
            embed=embed,
            feat=self.dynamics.get_feat(post),
            kl=kl_value,
            postent=self.dynamics.get_dist(post).entropy(),
        )
        
        # Detach post for return
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics
    
    def preprocess(self, obs):
        """Preprocess observations (same as original WorldModel)."""
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = obs["discount"].unsqueeze(-1)
        assert "is_first" in obs
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs
    
    def video_pred(self, data):
        """Generate video predictions (same as original WorldModel)."""
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        
        # Combine observed and predicted frames
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)
    
    def intervene(self, index, value):
        """Perform an intervention on a specific latent variable."""
        self.dynamics.intervene(index, value)
        return self
    
    def remove_intervention(self):
        """Remove any active intervention."""
        self.dynamics.remove_intervention()
        return self
    
    def get_causal_mask(self):
        """Get the current causal adjacency matrix."""
        return self.dynamics.get_causal_mask()
    
    def get_causal_effects(self, state, target_index, num_samples=10):
        """Analyze causal effects on a specific target variable."""
        effects = []
        
        # Try different interventions on all variables
        for var_idx in range(self._config.dyn_stoch):
            if var_idx == target_index:
                continue  # Skip self-effect
                
            var_effects = []
            values = torch.linspace(-2.0, 2.0, num_samples)
            
            for val in values:
                # Setup intervention
                self.intervene(var_idx, val)
                
                # Get effect on target
                feat = self.dynamics.get_feat(state)
                next_state = self.dynamics.img_step(state, torch.zeros((state["deter"].shape[0], self._config.num_actions), device=self._config.device))
                next_feat = self.dynamics.get_feat(next_state)
                
                # Measure effect on target variable
                target_value = next_state["stoch"][:, target_index]
                
                var_effects.append({
                    "intervention_value": val.item(),
                    "target_value": target_value.mean().item()
                })
            
            # Remove intervention
            self.remove_intervention()
            
            effects.append({
                "source_variable": var_idx,
                "target_variable": target_index,
                "effects": var_effects
            })
        
        return effects