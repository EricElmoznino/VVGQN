import random
random.seed(27)
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.environments import SimpleRectangularEnvironment
from datasets.path_generators import start_state


class SingleViewDataset(Dataset):

    def __init__(self, width=10, height=10, n_samples=1000):
        self.n_samples = n_samples
        self.env = SimpleRectangularEnvironment(width, height)

    def sample(self):
        pos, dir = start_state(self.env, d_from_walls=3)
        self.env.place_agent_at(pos, dir)

        visual = self.env.render_obs() / 255
        visual = torch.from_numpy(visual).float()
        visual = visual.permute([2, 0, 1])

        return visual

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.sample()
