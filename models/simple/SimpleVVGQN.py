import torch
import torch.nn as nn

from .SimpleRepresentation import SimpleRepresentation
from .SimpleGenerator import SimpleGenerator


class SimpleVVGQN(nn.Module):
    """
    :param c_dim: number of channels in image input
    :param v_dim: dimensions of vestibular input
    :param r_dim: dimensions of representation
    """
    def __init__(self, c_dim, v_dim, r_dim):
        super().__init__()
        self.r_dim = r_dim

        self.visual = SimpleRepresentation(c_dim, 0, r_dim)
        self.representation = nn.LSTM(v_dim + r_dim, r_dim, batch_first=True)
        self.projection = nn.LSTM(v_dim, r_dim, batch_first=True)
        self.generator = SimpleGenerator(c_dim, 0, r_dim)

    def forward(self, context_x, context_v, query_v):
        """
        :param context_x: batch of context images [b, m, c, h, w]
        :param context_v: batch of context vestibular actions [b, m, k]
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
        query_x = self.generator(r_projected, None)

        return query_x, r
