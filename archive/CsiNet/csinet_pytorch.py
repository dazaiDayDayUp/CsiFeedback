import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# CsiNet reproduced in PyTorch
# Reference: Wen et al., "Deep Learning for Massive MIMO CSI Feedback" (IEEE WCL 2018)

img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels
envir = 'indoor'  # 'indoor' or 'outdoor'
encoded_dim = 512  # 1/4->512, 1/16->128, 1/32->64, 1/64->32

epochs = 1000
batch_size = 200
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 2, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2)

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.3)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope=0.3)
        out = self.bn3(self.conv3(out))
        out = F.leaky_relu(out + identity, negative_slope=0.3)
        return out


class CsiNet(nn.Module):
    def __init__(self, encoded_dim=512):
        super(CsiNet, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(negative_slope=0.3)
        )
        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(img_total, encoded_dim)

        self.decoder_fc = nn.Linear(encoded_dim, img_total)
        self.decoder_res1 = ResidualBlock()
        self.decoder_res2 = ResidualBlock()
        self.decoder_conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (N, 2, 32, 32)
        enc = self.encoder_conv(x)
        enc = self.flatten(enc)
        encoded = self.encoder_fc(enc)

        dec = self.decoder_fc(encoded)
        dec = dec.view(-1, img_channels, img_height, img_width)
        dec = self.decoder_res1(dec)
        dec = self.decoder_res2(dec)
        dec = self.decoder_conv(dec)
        out = self.sigmoid(dec)
        return out


def nmse_db(x, x_hat):
    # x, x_hat: (N, 2, 32, 32) tensors on CPU or GPU
    x_real = x[:, 0, :, :] - 0.5
    x_imag = x[:, 1, :, :] - 0.5
    x_c = x_real + 1j * x_imag

    x_hat_real = x_hat[:, 0, :, :] - 0.5
    x_hat_imag = x_hat[:, 1, :, :] - 0.5
    x_hat_c = x_hat_real + 1j * x_hat_imag

    power = torch.sum(torch.abs(x_c) ** 2, dim=(1, 2))
    mse = torch.sum(torch.abs(x_c - x_hat_c) ** 2, dim=(1, 2))
    return 10 * torch.log10((mse / power).mean())


def load_data(envir):
    prefix = 'in' if envir == 'indoor' else 'out'
    train = sio.loadmat(f'../data/DATA_Htrain{prefix}.mat')['HT']
    val = sio.loadmat(f'../data/DATA_Hval{prefix}.mat')['HT']
    test = sio.loadmat(f'../data/DATA_Htest{prefix}.mat')['HT']

    train = torch.tensor(train, dtype=torch.float32).view(-1, img_channels, img_height, img_width)
    val = torch.tensor(val, dtype=torch.float32).view(-1, img_channels, img_height, img_width)
    test = torch.tensor(test, dtype=torch.float32).view(-1, img_channels, img_height, img_width)
    return train, val, test


print('Loading data...')
x_train, x_val, x_test = load_data(envir)
x_train = x_train.to(device)
x_val = x_val.to(device)
x_test = x_test.to(device)
print(f'Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}')

model = CsiNet(encoded_dim=encoded_dim).to(device)
print(model)

optimizer = Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()


def train_epoch(model, data, optimizer, criterion, batch_size):
    model.train()
    n = data.size(0)
    perm = torch.randperm(n)
    total_loss = 0.0
    n_batches = 0
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        batch = data[idx]
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def evaluate(model, data, batch_size=200):
    model.eval()
    n = data.size(0)
    total_loss = 0.0
    n_batches = 0
    all_pred = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = data[i:i + batch_size]
            out = model(batch)
            loss = criterion(out, batch)
            total_loss += loss.item()
            n_batches += 1
            all_pred.append(out.cpu())
    avg_loss = total_loss / n_batches
    preds = torch.cat(all_pred, dim=0).to(device)
    nmse = nmse_db(data, preds)
    return avg_loss, nmse.item()


best_val_loss = float('inf')
train_losses = []
val_losses = []
val_nmses = []

print('Training starts...')
start_time = time.time()
for epoch in range(epochs):
    train_loss = train_epoch(model, x_train, optimizer, criterion, batch_size)
    val_loss, val_nmse = evaluate(model, x_val, batch_size)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_nmses.append(val_nmse)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'Epoch {epoch + 1}/{epochs}  train_loss: {train_loss:.4e}  val_loss: {val_loss:.4e}  val_NMSE: {val_nmse:.4f} dB  time: {(time.time()-start_time)/60:.2f}min')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_nmse': val_nmse,
        }, f'./csinet_pytorch_{envir}_dim{encoded_dim}_best.pth')

    # Periodic checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f'./csinet_pytorch_{envir}_dim{encoded_dim}_epoch{epoch+1}.pth')

# Final test
test_loss, test_nmse = evaluate(model, x_test, batch_size)
print(f'\nFinal test loss: {test_loss:.4e}  test NMSE: {test_nmse:.4f} dB')

# Save final model
torch.save(model.state_dict(), f'./csinet_pytorch_{envir}_dim{encoded_dim}_final.pth')

# Plot losses
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('CsiNet-PyTorch Loss')

plt.subplot(1, 2, 2)
plt.plot(val_nmses, label='val NMSE (dB)')
plt.xlabel('Epoch')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.title('CsiNet-PyTorch Validation NMSE')
plt.tight_layout()
plt.savefig(f'./csinet_pytorch_{envir}_dim{encoded_dim}_curves.png')
plt.close()

# Save results
with open(f'./csinet_pytorch_{envir}_dim{encoded_dim}_result.txt', 'w') as f:
    f.write(f'CsiNet reproduced in PyTorch\n')
    f.write(f'Environment: {envir}\n')
    f.write(f'Encoded dim: {encoded_dim}\n')
    f.write(f'Epochs: {epochs}\n')
    f.write(f'Batch size: {batch_size}\n')
    f.write(f'Learning rate: {lr}\n')
    f.write(f'Final test loss: {test_loss:.4e}\n')
    f.write(f'Final test NMSE (dB): {test_nmse:.4f}\n')
    f.write(f'Best val loss: {best_val_loss:.4e}\n')

print('Done.')
