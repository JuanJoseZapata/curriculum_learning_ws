import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class TracksDataset(Dataset):
    """Tracks dataset.
    """

    def __init__(self, tracks, difficulties):
        self.tracks = tracks
        self.difficulties = difficulties

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        x = TF.to_tensor(self.tracks[index])
        y = torch.tensor(self.difficulties[index])
        return x, y

    
# Loss function
def loss_function(x_hat, x, mu, logvar):

    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    
    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_divergence


def train_vae(model, train_loader, val_loader,
              num_epochs=10, learning_rate=1e-4,
              early_stopping={'patience': 5, 'min_delta': 0.001},
              images=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True, min_lr=1e-6)

    loss_history = {'train_loss': [], 'val_loss': []}
    early_stop_counter = 0
    best_val_loss = float('inf')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Training
        for x, y in train_loader:
            optimizer.zero_grad()

            # Flatten the input data if it is not images
            if not images:
                x = x.view(x.size(0), -1)
            x = x.to(device)

            y = y.view(y.size(0), -1)
            y = y.to(device)

            x_hat, mu, logvar = model(x, y)
            loss = loss_function(x_hat, x, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                if not images:
                    x = x.view(x.size(0), -1)
                x = x.to(device)

                y = y.view(y.size(0), -1)
                y = y.to(device)

                x_hat, mu, logvar = model(x, y)
                loss = loss_function(x_hat, x, mu, logvar)

                val_loss += loss.item()
        val_loss /= len(val_loader.dataset)

        average_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], train_loss: {average_loss:.4f}, val_loss: {val_loss:.4f}')

        loss_history['train_loss'].append(average_loss)
        loss_history['val_loss'].append(val_loss)

        # Early stopping
        if epoch > 0:
            if loss_history['val_loss'][-1] < best_val_loss - early_stopping['min_delta']:
                best_val_loss = loss_history['val_loss'][-1]
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stopping['patience']:
                print('Early stopping.')
                break
        
        scheduler.step(val_loss)

    print('Training finished.')

    return loss_history