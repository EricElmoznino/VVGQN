import random
random.seed(27)
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.environments import SimpleRectangularEnvironment
from datasets.path_generators import start_state


class MultiViewDataset(Dataset):

    def __init__(self, n_views=20, width=10, height=10, n_samples=1000):
        self.v_dim = 4
        self.n_samples = n_samples
        self.env = SimpleRectangularEnvironment(width, height)
        self.n_views = n_views

    def sample(self):
        visual, viewpoints = [], []
        for _ in range(self.n_views):
            pos, dir = start_state(self.env, d_from_walls=3)
            self.env.place_agent_at(pos, dir)

            v = self.env.render_obs() / 255
            v = torch.from_numpy(v).float()
            v = v.permute([2, 0, 1])
            visual.append(v)

            pos = pos / np.array([self.env.width, self.env.height], dtype=np.float32) - 0.5
            view = torch.FloatTensor([pos[0], pos[1], math.cos(dir), -math.sin(dir)])    # -sin because of righthand OpenGL coordinates
            viewpoints.append(view)

        visual, viewpoints = torch.stack(visual), torch.stack(viewpoints)

        return visual, viewpoints

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.sample()
