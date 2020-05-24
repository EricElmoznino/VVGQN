import torch.nn as nn

from .Representation import Representation
from .Generator import Generator


class VAE(nn.Module):
    """
    Single-image VAE version of the Generative Query Network (GQN) as described
    in "Neural scene representation and rendering"
    [Eslami 2018].

    :param c_dim: number of channels in input
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param l: Number of refinements of density
    """
    def __init__(self, c_dim, r_dim, z_dim=3, h_dim=128, l=8):
        super().__init__()
        self.r_dim = r_dim

        self.representation = Representation(c_dim, 0, r_dim)
        self.generator = Generator(c_dim, 0, r_dim, z_dim, h_dim, l)

    def forward(self, x):
        """
        Forward through the GQN.

        :param x: batch of images [b, c, h, w]
        """
        # Representation generated from input image
        r = self.representation(x, None)

        # Use random (image, viewpoint) pair in batch as query
        x_mu, kl = self.generator(x, None, r)

        return x_mu, r, kl

    def sample(self, x):
        """
        Sample from the network given some image.

        :param x: image to generate representation
        """
        r = self.representation(x, None)
        x_mu = self.generator.sample(None, r)
        return x_mu, r
