import torch
import torch.nn as nn

from .SimpleRepresentation import SimpleRepresentation
from .SimpleGenerator import SimpleGenerator


class SimpleGQN(nn.Module):
    """
    :param c_dim: number of channels in image input
    :param v_dim: dimensions of vestibular input
    :param r_dim: dimensions of representation
    """
    def __init__(self, c_dim, v_dim, r_dim):
        super().__init__()
        self.r_dim = r_dim

        self.representation = SimpleRepresentation(c_dim, v_dim, r_dim)
        self.generator = SimpleGenerator(c_dim, v_dim, r_dim)

    def forward(self, context_x, context_v, query_v):
        """
        :param context_x: batch of context images [b, m, c, h, w]
        :param context_v: batch of context vestibular actions [b, m, k]
        :param query_v: batch of query vestibular actions [b, m, k]
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

        # Generate predicted image for final representation
        x_gen = self.generator(r, query_v)

        return x_gen, r
