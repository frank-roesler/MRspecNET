import torch.nn as nn


class MRspecNET(nn.Module):
    """1d convolutional autoencoder for denoising"""
    def __init__(self, kernel_size=32, n_channels=64, n_layers=6):
        super().__init__()
        self.name = 'MRspecNET_'+str(kernel_size)+'_'+str(n_channels)+'_'+str(n_layers)
        if n_layers % 2 != 0:
            raise ValueError('Number of layers must be even.')
        nc = n_channels
        layers_enc = [nn.Conv1d(2, nc, kernel_size=kernel_size, padding=kernel_size), nn.ReLU(inplace=True)]
        layers_dec = [nn.ConvTranspose1d(nc, 2, kernel_size=kernel_size, padding=kernel_size)]
        for i in range(n_layers // 2 - 1):
            layers_enc.append(nn.Conv1d(nc, 2 * nc, kernel_size=kernel_size, padding=kernel_size))
            layers_enc.append(nn.ReLU(inplace=True))
            layers_dec.insert(0, nn.ReLU(inplace=True))
            layers_dec.insert(0, nn.ConvTranspose1d(2 * nc, nc, kernel_size=kernel_size, padding=kernel_size))
            nc *= 2

        self.encoder = nn.Sequential(*layers_enc)
        self.decoder = nn.Sequential(*layers_dec)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

