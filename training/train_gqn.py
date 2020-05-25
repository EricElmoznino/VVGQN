from argparse import ArgumentParser
import random
from training.trainer import train
from models.GQN import GQN
from datasets.SequenceDataset import SequenceDataset

random.seed(27)


if __name__ == '__main__':
    parser = ArgumentParser(description='VAE training')
    parser.add_argument('--run_name', required=True, type=str,
                        help='name of the current run (where runs are saved)')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs run')
    parser.add_argument('--samples_per_epoch', type=int, default=10000, help='number of samples per epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--r_dim', type=int, default=256, help='r_dim for model')
    parser.add_argument('--h_dim', type=int, default=128, help='h_dim for model')
    parser.add_argument('--z_dim', type=int, default=3, help='z_dim for model')
    parser.add_argument('--l', type=int, default=8, help='L for model')
    parser.add_argument('--path_length', type=int, default=20, help='Length of path')
    parser.add_argument('--min_obs', type=int, default=5, help='Minimum number of observations')
    parser.add_argument('--max_obs', type=int, default=10, help='Maximum number of observations')
    args = parser.parse_args()

    assert 0 < args.min_obs <= args.max_obs < args.path_length

    def forward_func(model, batch):
        x, v = batch

        n_obs = random.randint(args.min_obs, args.max_obs)
        x, v, x_q, v_q = x[:, :n_obs], v[:, :n_obs], x[:, -1], v[:, -1]

        x_mu, _, kl = model(x, v, x_q, v_q)
        return x_mu, x_q, kl


    def sample_func(model, batch):
        x, v = batch

        n_obs = random.randint(args.min_obs, args.max_obs)
        x, v, x_q, v_q = x[:, :n_obs], v[:, :n_obs], x[:, -1], v[:, -1]

        x_mu, r = model.sample(x, v, v_q)
        return x_mu, x_q, r

    train_set = SequenceDataset(path_length=args.path_length, n_samples=args.samples_per_epoch)
    val_set = SequenceDataset(path_length=args.path_length, n_samples=args.batch_size)
    model = GQN(c_dim=3, v_dim=train_set.v_dim, r_dim=args.r_dim, h_dim=args.h_dim, z_dim=args.z_dim, l=args.l)

    train(args.run_name, forward_func, sample_func, model, train_set, val_set,
          args.n_epochs, args.batch_size, args.lr)
