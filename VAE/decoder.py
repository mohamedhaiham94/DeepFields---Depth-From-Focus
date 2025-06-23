import torch
import torch.nn as nn

class FCDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(FCDecoder, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 3, stride=2, padding=1, output_padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # Upsample
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),  # Final reconstruction
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, z):
        return self.deconv(z)
