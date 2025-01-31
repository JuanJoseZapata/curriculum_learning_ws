import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, latent_dim * 2, bias=True)  # Two outputs for mean and variance
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim//4, bias=True),  # Add conditional input
            nn.ReLU(),
            nn.Linear(hidden_dim//4, hidden_dim//2, bias=True),
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
        mu = z_params[:, :self.latent_dim]
        logvar = z_params[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)

        # Concatenate z and y
        z = torch.cat([z, y], dim=-1)
        
        # Decode
        x_hat = self.decoder(z)
        
        return x_hat, mu, logvar


# Number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, 4, 4)

class VAE_CNN(nn.Module):
    def __init__(self, image_channels=1, h_dim=2048, z_dim=16):
        super(VAE_CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels + 1, 16, kernel_size=3, stride=1, padding=1),  # Reduced filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten()
        )
        
        # Single fully connected layer to output both mu and logvar
        self.fc_mu_logvar = nn.Linear(h_dim, 2 * z_dim)
        
        # Decoder fully connected layer with conditional input
        self.fc3 = nn.Linear(z_dim + 1, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim // 16, 32, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 24, kernel_size=4, stride=2, padding=1),    # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1),    # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            #nn.Tanh(),
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, h):
        mu_logvar = self.fc_mu_logvar(h)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x, y):
        y = y.view(-1, 1, 1, 1)
        y = y.expand(x.size(0), 1, x.size(2), x.size(3))
        x_cond = torch.cat([x, y], dim=1)
        h = self.encoder(x_cond)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x, y):
        z, mu, logvar = self.encode(x, y)
        y = y.view(-1, 1)
        z = torch.cat([z, y], dim=1)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# Loss function
def loss_function(x_hat, x, mu, logvar):

    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    
    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_divergence