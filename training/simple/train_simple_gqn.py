from argparse import ArgumentParser
import random
random.seed(27)
from training.simple.trainer_simple import train
from models.simple.SimpleGQN import SimpleGQN
from datasets.MultiViewDataset import MultiViewDataset


if __name__ == '__main__':
    parser = ArgumentParser(description='Simple AE training')
    parser.add_argument('--run_name', required=True, type=str,
                        help='name of the current run (where runs are saved)')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs run')
    parser.add_argument('--samples_per_epoch', type=int, default=10000, help='number of samples per epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--r_dim', type=int, default=256, help='r_dim for model')
    parser.add_argument('--min_obs', type=int, default=5, help='Minimum number of observations')
    parser.add_argument('--max_obs', type=int, default=10, help='Maximum number of observations')
    args = parser.parse_args()

    assert 0 < args.min_obs <= args.max_obs

    def forward_func(model, batch):
        x, v = batch

        n_obs = random.randint(args.min_obs, args.max_obs)
        x, v, x_q, v_q = x[:, :n_obs], v[:, :n_obs], x[:, -1], v[:, -1]

        x_gen, r = model(x, v, v_q)
        return x_gen, x_q, r

    train_set = MultiViewDataset(n_views=args.max_obs + 1, n_samples=args.samples_per_epoch)
    val_set = MultiViewDataset(n_views=args.max_obs + 1, n_samples=args.batch_size)
    model = SimpleGQN(c_dim=3, v_dim=train_set.v_dim, r_dim=args.r_dim)

    train(args.run_name, forward_func, model, train_set, val_set,
          args.n_epochs, args.batch_size, args.lr)
