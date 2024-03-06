import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, latent_dim * 2, bias=True)  # Two outputs for mean and variance
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim//2, bias=True),  # Add conditional input
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim, bias=True),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim - 1),  # Remove conditional input
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x, y):
        t = torch.cat((x, y), dim=-1)
        # Encode
        z_params = self.encoder(t)
        mu = z_params[:, :latent_dim]
        logvar = z_params[:, latent_dim:]
        z = self.reparameterize(mu, logvar)

        # Concatenate z and y
        z = torch.cat([z, y], dim=-1)
        
        # Decode
        x_hat = self.decoder(z)
        
        return x_hat, mu, logvar