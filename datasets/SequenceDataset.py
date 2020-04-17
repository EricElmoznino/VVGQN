import random
random.seed(27)
import numpy as np
import torch
from datasets.environments import SimpleRectangularEnvironment
from datasets.path_generators import noisy_path


class SequenceDataset():

    def __init__(self, path_length=20, width=10, height=10):
        self.env = SimpleRectangularEnvironment(width, height)
        self.path_length = path_length

    def sample(self, return_path=False, return_top_view=False):
        path = noisy_path(self.env, self.path_length)
        self.env.place_agent_at(path['positions'][0], path['directions'][0])

        visual = [self.env.move(path['movements'][i], path['rotations'][i]) / 255
                  for i in range(self.path_length)]
        visual = np.stack(visual)
        visual = torch.from_numpy(visual).float()
        visual = visual.permute([0, 3, 1, 2])

        movements = torch.from_numpy(path['movements'])
        rotations = torch.from_numpy(path['rotations'])
        vestibular = torch.stack([movements, rotations.cos(), rotations.sin()], dim=1)

        if return_path:
            positions = torch.from_numpy(path['positions'])
            directions = torch.from_numpy(path['directions'])
            return visual, vestibular, positions, directions
        else:
            return visual, vestibular

    def sample_batch(self, batch_size, return_path=False):
        samples = zip(*[self.sample(return_path) for _ in range(batch_size)])
        samples = tuple(torch.stack(tensor) for tensor in samples)
        return samples
