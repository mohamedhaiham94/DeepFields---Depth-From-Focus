import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet18

class VAE(nn.Module):
    def __init__(self, latent_dim, depth):
        super(VAE, self).__init__()
    
        self.latent_dim = latent_dim
        self.beta = 0.00025
        # Encoder: input (B, 1, 2, 2)
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (16, 4, 4)
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=2, stride=2),            # (32, 2, 2)
        #     nn.ReLU(),
        #     nn.Flatten()
        # )

        # self.fc_mu = nn.Linear(32 * 1 * 1, latent_dim)
        # self.fc_logvar = nn.Linear(32 * 1 * 1, latent_dim)

        # # Decoder
        # self.decoder_fc = nn.Linear(latent_dim, 32 * 1 * 1)
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),   # (16, 4, 4)
        #     nn.ReLU(),
        #     nn.Conv2d(16, 1, kernel_size=3, padding=1)            # (1, 4, 4)
        #     # nn.Sigmoid()
        # )
        ########################################
        # Encoder: input (B, 1, 4, 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(depth, 16, kernel_size=3, stride=1, padding=1),  # (16, 2, 2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),            # (32, 1, 1)
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(32 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(32 * 2 * 2, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 32 * 2 * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),   # (16, 2, 2)
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)            # (1, 2, 2)
            # nn.Sigmoid()
        )
        #####################################
        # Encoder size 8x8
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output: 8x8
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: 4x4
        #     nn.ReLU(),
        #     nn.Flatten()
        # )
        # self.fc_mu = nn.Linear(32 * 2 * 2, latent_dim)
        # self.fc_logvar = nn.Linear(32 * 2 * 2, latent_dim)

        # # Decoder
        # self.decoder_fc = nn.Linear(latent_dim, 32 * 2 * 2)
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: 8x8
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # Output: 16x16
        #     nn.Sigmoid()
        # )
        ##########################################
                # Encoder: input (1, 32, 32)
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (32, 16, 16)
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 8, 8)
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 4, 4)
        #     nn.ReLU(),
        #     nn.Flatten()
        # )
        # self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        # self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # # Decoder
        # self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 8, 8)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 16, 16)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # (1, 32, 32)
        #     nn.Sigmoid()
        # )
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(torch.mul(logvar, 0.5))
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        h = self.decoder_fc(z)

        ######### size 4x4
        h = h.view(-1, 32, 2, 2)


        ######### size 2x2
        # h = h.view(-1, 32, 1, 1)


        ###### size 32x32
        # h = h.view(-1, 128, 4, 4)


        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        self.mu = mu
        self.logvar = logvar
        return x_recon, (mu, logvar)



    def loss_function(self, recon_x, x):
        # Reconstruction loss (per-pixel)
        x = x[:, -1, :, :].unsqueeze(1)

        recon_loss = F.mse_loss(recon_x, x)
        
        # KL divergence (per pixel)
        #kl_loss = torch.mean(-0.5 * torch.sum(1 + self.logvar - self.mu ** 2 - self.logvar.exp(), dim = 1), dim = 0)
        # kl_loss = -torch.mean(1 + torch.log(self.logvar.pow(2)) - self.mu.pow(2) - self.logvar.pow(2))

        kl_loss = 0

        return recon_loss + kl_loss * self.beta

    def step(self, input_mb):
        recon_mb, _ = self.forward(input_mb)

        loss = self.loss_function(recon_mb, input_mb)

        # recon_mb = self.mean_from_lambda(recon_mb)

        return loss, recon_mb, self.mu, self.logvar



    def tarctanh(self, x):
        return 0.5 * torch.log((1+x)/(1-x))

        