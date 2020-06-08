import torch.nn as nn

from .SimpleRepresentation import SimpleRepresentation
from .SimpleGenerator import SimpleGenerator


class SimpleAE(nn.Module):
    """
    :param c_dim: number of channels in image input
    :param v_dim: dimensions of vestibular input
    :param r_dim: dimensions of representation
    """
    def __init__(self, c_dim, r_dim):
        super().__init__()
        self.r_dim = r_dim

        self.representation = SimpleRepresentation(c_dim, 0, r_dim)
        self.generator = SimpleGenerator(c_dim, 0, r_dim)

    def forward(self, x):
        """
        :param x: batch of images [b, c, h, w]
        """
        r = self.representation(x, None)
        x_reconstructed = self.generator(r, None)
        return x_reconstructed, r
