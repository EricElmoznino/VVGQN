import torch
import torch.nn as nn

from .Representation import Representation
from .Generator import Generator


class GQN(nn.Module):
    """
    Generative Query Network (GQN) as described
    in "Neural scene representation and rendering"
    [Eslami 2018].

    :param c_dim: number of channels in input
    :param v_dim: dimensions of viewpoint
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param l: Number of refinements of density
    """
    def __init__(self, c_dim, v_dim, r_dim, z_dim=3, h_dim=128, l=8):
        super().__init__()
        self.r_dim = r_dim

        self.representation = Representation(c_dim, v_dim, r_dim)
        self.generator = Generator(c_dim, v_dim, r_dim, z_dim, h_dim, l)

    def forward(self, context_x, context_v, query_x, query_v):
        """
        Forward through the GQN.

        :param context_x: batch of context images [b, m, c, h, w]
        :param context_v: batch of context viewpoints for image [b, m, k]
        :param query_x: batch of query images [b, c, h, w]
        :param query_v: batch of query viewpoints [b, k]
        """
        # Merge batch and view dimensions.
        batch_size, n_views, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.reshape((-1, *x_dims))
        v = context_v.reshape((-1, *v_dims))

        # Representation generated from input images
        # and corresponding viewpoints
        phi = self.representation(x, v)

        # Separate batch and view dimensions
        _, *phi_dims = phi.shape
        phi = phi.view((batch_size, n_views, *phi_dims))

        # Mean of view representations
        r = torch.mean(phi, dim=1)

        # Use random (image, viewpoint) pair in batch as query
        x_mu, kl = self.generator(query_x, query_v, r)

        return x_mu, r, kl

    def sample(self, context_x, context_v, query_v):
        """
        Sample from the network given some context and viewpoint.

        :param context_x: set of context images to generate representation
        :param context_v: viewpoints of `context_x`
        :param query_v: viewpoint to generate image from
        """
        batch_size, n_views, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.reshape((-1, *x_dims))
        v = context_v.reshape((-1, *v_dims))

        phi = self.representation(x, v)

        _, *phi_dims = phi.shape
        phi = phi.view((batch_size, n_views, *phi_dims))

        r = torch.sum(phi, dim=1)

        x_mu = self.generator.sample(query_v, r)

        return x_mu, r
