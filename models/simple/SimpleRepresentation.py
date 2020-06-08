import torch
import torch.nn as nn


class SimpleRepresentation(nn.Module):
    def __init__(self, c_dim, v_dim, r_dim):
        """
        Network that generates a condensed representation
        vector from a joint input of image and viewpoint.

        Employs the tower/pool architecture described in the paper.

        :param c_dim: number of color channels in input image
        :param v_dim: dimensions of the vestibular vector
        :param r_dim: dimensions of representation
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(c_dim + v_dim, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.mapping = nn.Linear(256, r_dim)

    def forward(self, x, v):
        """
        Send an (image, vestibular) pair into the
        network to generate a representation
        :param x: image
        :param v: vestibular
        :return: representation
        """
        if v is not None:
            v = v.view(v.size(0), -1, 1, 1)
            v = v.repeat(1, 1, x.shape[-2], x.shape[-1])
            x = torch.cat([x, v], dim=1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        r = self.mapping(x)

        return r
