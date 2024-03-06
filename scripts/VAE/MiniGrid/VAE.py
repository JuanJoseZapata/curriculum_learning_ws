import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        # Decode
        x_hat = self.decoder(z)
        
        return x_hat, mu, logvar


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, 1, 1)

class VAE_conv(nn.Module):
    def __init__(self, image_channels=1, h_dim=768, z_dim=32):
        super(VAE_conv, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels + 1, 16, kernel_size=7, stride=1),  # Add conditional input
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(32*3*3, z_dim)
        self.fc2 = nn.Linear(32*3*3, z_dim)
        self.fc3 = nn.Linear(z_dim + 1, h_dim)  # Add conditional input
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 24, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x, y):
        y = y.reshape((y.shape[0],1,1,1)).to(device)
        y = torch.ones(x.shape[0],1,x.shape[2],x.shape[3]).to(device)*y
        t = torch.cat((x,y),dim=1)
        h = self.encoder(t)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x, y):
        z, mu, logvar = self.encode(x, y)
        y = y.to(device)
        z = torch.cat((z,y),dim=1)
        z = self.decode(z)
        return z, mu, logvar