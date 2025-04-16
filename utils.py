import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

# Using function-defined device instead of global variable
def conditional_sample_gaussian(m, v, device=None):
    """Sample from a conditional Gaussian"""
    if device is None:
        device = m.device
    sample = torch.randn(m.size()).to(device)
    z = m + (v**0.5) * sample
    return z

def condition_prior(scale, label, dim, device=None):
    """Calculate condition prior"""
    if device is None:
        device = label.device
    mean = torch.ones(label.size()[0], label.size()[1], dim)
    var = torch.ones(label.size()[0], label.size()[1], dim)
    for i in range(label.size()[0]):
        for j in range(label.size()[1]):
            mul = (float(label[i][j]) - scale[j][0]) / (scale[j][1] - 0)
            mean[i][j] = torch.ones(dim) * mul
            var[i][j] = torch.ones(dim) * 1
    return mean.to(device), var.to(device)

def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl

def log_bernoulli_with_logits(x, logits):
    """
    Computes the log probability of a Bernoulli given its logits
    """
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    log_prob = -bce(input=logits, target=x).sum(-1)
    return log_prob

def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def vector_expand(v):
    """Expand vector to diagonal matrix"""
    device = v.device
    V = torch.zeros(v.size()[0], v.size()[1], v.size()[1]).to(device)
    for i in range(v.size()[0]):
        for j in range(v.size()[1]):
            V[i, j, j] = v[i, j]
    return V