import torch
import torch.nn as nn

class FCEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(FCEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),  # Keep size the same
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Downsample: HxW -> H/2xW/2
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # Downsample: H/2xW/2 -> H/4xW/4
            nn.ReLU()
        )
        self.fc_mu = nn.Conv2d(128, latent_dim, 1)      # 1x1 Conv for mean
        self.fc_logvar = nn.Conv2d(128, latent_dim, 1)  # 1x1 Conv for log variance

    def forward(self, x):
        x = self.conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
