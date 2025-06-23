import torch 
import torch.nn as nn
import torch.nn.functional as F

# def vae_loss(recon_x, x, mu, logvar):
#     # Reconstruction loss (per-pixel)

#     recon_loss = F.mse_loss(recon_x, x)
    
#     # KL divergence (per pixel)
#     kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
#     beta = 0.00025
#     return recon_loss + kl_loss * beta

# ================================
# VAE LOSS FUNCTIONS
# ================================
def xent_continuous_ber(recon_x, x, pixelwise=False):
    """Continuous Bernoulli cross-entropy approximation"""
    epsilon = 1e-6
    log_norm = torch.log1p((1 - 2 * recon_x).clamp(min=-1 + epsilon, max=1 - epsilon))
    xent = x * torch.log(recon_x + epsilon) + (1 - x) * torch.log(1 - recon_x + epsilon)
    loss = -xent - log_norm

    if pixelwise:
        return loss  # shape (B, 1, H, W)
    return loss.sum(dim=(1, 2, 3)).mean()  # scalar

def kld(mu, logvar):
    """KL divergence between posterior and standard Gaussian prior"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def vae_loss_function(recon_x, x, mu, logvar):
    recon = xent_continuous_ber(recon_x, x)
    kl = kld(mu, logvar)
    loss = recon + kl
    return loss, {'recon_loss': recon.item(), 'kl_div': kl.item()}