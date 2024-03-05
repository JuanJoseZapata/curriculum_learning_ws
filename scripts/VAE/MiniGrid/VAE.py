import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, image_channels=3, output_dim=None):
        super(VAE, self).__init__()

        if output_dim is None:
            output_dim = input_dim - 1
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, latent_dim * 2, bias=False)  # Two outputs for mean and variance
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim//2, bias=False),  # Add conditional input
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),  # Remove conditional input
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x, y):

        # Linear VAE
        t = torch.cat((x, y), dim=-1)
        # Encode
        z_params = self.encoder(t)
        mu = z_params[:, :latent_dim]
        logvar = z_params[:, latent_dim:]
        z = self.reparameterize(mu, logvar)

        # Concatenate z and y
        z = torch.cat([z, y], dim=-1)
        
        # Linear decoder
        x_hat = self.decoder(z)
        
        return x_hat, mu, logvar