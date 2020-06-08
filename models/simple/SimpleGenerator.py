import torch
import torch.nn as nn


class SimpleGenerator(nn.Module):
    """
    :param c_dim: number of channels in input
    :param v_dim: dimensions of the vestibular vector
    :param r_dim: dimensions of representation
    """

    def __init__(self, c_dim, v_dim, r_dim):
        super().__init__()

        self.mapping = nn.Sequential(
            nn.Linear(r_dim + v_dim, 256),
            nn.ReLU(inplace=True)
        )
        self.features = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, c_dim, kernel_size=3, padding=1)
        )

    def forward(self, r, v):
        """
        Attempt to reconstruct x with corresponding
        vestibular input v and context representation r.

        :param r: representation of the environment
        :param v: vestibular input
        :return reconstruction of x
        """
        if v is not None:
            r = torch.cat([r, v], dim=1)

        x = self.mapping(r)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.features(x)

        x = torch.sigmoid(x)
        return x
