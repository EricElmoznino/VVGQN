import torch
import torch.nn as nn

from .Representation import Representation
from .Generator import Generator


class VVGQN(nn.Module):
    """
    Visuo-Vestibular Generative Query Network (VVGQN).
    Modified version of GQN as described
    in "Neural scene representation and rendering"
    [Eslami 2018].

    :param c_dim: number of channels in image input
    :param v_dim: dimensions of vestibular input
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param l: Number of refinements of density
    """
    def __init__(self, c_dim, v_dim, r_dim, z_dim=3, h_dim=128, l=8):
        super().__init__()
        self.r_dim = r_dim

        self.visual = Representation(c_dim, 0, r_dim)
        self.representation = nn.LSTM(v_dim + r_dim, r_dim, batch_first=True)
        self.projection = nn.LSTM(v_dim, r_dim, batch_first=True)
        self.generator = Generator(c_dim, 0, r_dim, z_dim, h_dim, l)

    def forward(self, context_x, context_v, query_x, query_v):
        """
        Forward through the VVGQN.

        :param context_x: batch of context images [b, m, c, h, w]
        :param context_v: batch of context vestibular actions [b, m, k]
        :param query_x: batch of query images [b, c, h, w]
        :param query_v: batch of query vestibular actions [b, m, k]
        """
        # Merge batch and sequence dimensions.
        batch_size, n_views, *x_dims = context_x.shape
        _, _, v_dim = context_v.shape

        # Representation generated from input images
        context_x = context_x.reshape((-1, *x_dims))
        phi = self.visual(context_x, None)

        # Separate batch and sequence dimensions
        _, phi_dim = phi.shape
        phi = phi.view((batch_size, n_views, phi_dim))

        # Generate representation from visual and vestibular sequences
        phi = torch.cat((phi, context_v), dim=-1)
        _, state = self.representation(phi)
        r = state[0].squeeze(dim=0)

        # Project the representation into the future given new vestibular input
        _, (r_projected, _) = self.projection(query_v, state)
        r_projected = r_projected.squeeze(dim=0)

        # Generate predicted image for final representation
        x_mu, kl = self.generator(query_x, None, r_projected)

        return x_mu, r, kl

    def sample(self, context_x, context_v, query_v, generate_all=False):
        """
        Sample from the network given some context and vestibular actions.

        :param context_x: batch of context images [b, m, c, h, w]
        :param context_v: batch of context vestibular actions [b, m, k]
        :param query_v: batch of query vestibular actions [b, m, k]
        :param generate_all: whether to generate images for each query vestibular action or just the last one
        """
        batch_size, n_views, *x_dims = context_x.shape
        _, _, v_dim = context_v.shape
        _, _, p_dim = query_v.shape

        # Representation generated from input images
        context_x = context_x.reshape((-1, *x_dims))
        phi = self.visual(context_x, None)

        # Separate batch and sequence dimensions
        _, phi_dim = phi.shape
        phi = phi.view((batch_size, n_views, phi_dim))

        # Generate representation from visual and vestibular sequences
        phi = torch.cat((phi, context_v), dim=-1)
        _, state = self.representation(phi)
        r = state[0].squeeze(dim=0)

        # Project the representation into the future given new vestibular input
        r_projected, _ = self.projection(query_v, state)
        if not generate_all:
            r_projected = r_projected[:, -1]
        r_projected = r_projected.view((-1, phi_dim))

        x_mu = self.generator.sample(None, r_projected)
        if generate_all:
            x_mu = x_mu.view((-1, p_dim, *x_dims))

        return x_mu, r
