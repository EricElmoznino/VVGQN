import random
random.seed(27)
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.environments import SimpleRectangularEnvironment
from datasets.path_generators import noisy_path


class SequenceDataset(Dataset):

    def __init__(self, path_length=20, width=10, height=10, n_samples=1000):
        self.v_dim = 2
        self.n_samples = n_samples
        self.env = SimpleRectangularEnvironment(width, height)
        self.path_length = path_length

    def sample(self, return_path=False):
        path = noisy_path(self.env, self.path_length)
        self.env.place_agent_at(path['positions'][0], path['directions'][0])

        visual = [self.env.move(path['movements'][i], path['rotations'][i]) / 255
                  for i in range(self.path_length)]
        visual = np.stack(visual)
        visual = torch.from_numpy(visual).float()
        visual = visual.permute([0, 3, 1, 2])

        movements = torch.from_numpy(path['movements'])
        rotations = torch.from_numpy(path['rotations'])
        vestibular = torch.stack([movements, rotations], dim=1)

        if return_path:
            positions = torch.from_numpy(path['positions'])
            directions = torch.from_numpy(path['directions'])
            return visual, vestibular, positions, directions
        else:
            return visual, vestibular

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.sample()
