from argparse import ArgumentParser
import random
random.seed(27)
from training.trainer import train
from models.GQN import GQN
from datasets.MultiViewDataset import MultiViewDataset


if __name__ == '__main__':
    parser = ArgumentParser(description='GQN training')
    parser.add_argument('--run_name', required=True, type=str,
                        help='name of the current run (where runs are saved)')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs run')
    parser.add_argument('--samples_per_epoch', type=int, default=10000, help='number of samples per epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr_i', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('--lr_f', type=float, default=5e-5, help='final learning rate')
    parser.add_argument('--lr_n', type=float, default=1.6e6, help='number of steps for learning rate transition')
    parser.add_argument('--sig_i', type=float, default=2.0, help='initial likelihood sigma')
    parser.add_argument('--sig_f', type=float, default=0.7, help='final likelihood sigma')
    parser.add_argument('--sig_n', type=float, default=2e5, help='number of steps for likelihood sigma transition')
    parser.add_argument('--r_dim', type=int, default=256, help='r_dim for model')
    parser.add_argument('--h_dim', type=int, default=128, help='h_dim for model')
    parser.add_argument('--z_dim', type=int, default=3, help='z_dim for model')
    parser.add_argument('--l', type=int, default=8, help='L for model')
    parser.add_argument('--min_obs', type=int, default=5, help='Minimum number of observations')
    parser.add_argument('--max_obs', type=int, default=10, help='Maximum number of observations')
    args = parser.parse_args()

    assert 0 < args.min_obs <= args.max_obs

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

    train_set = MultiViewDataset(n_views=args.max_obs + 1, n_samples=args.samples_per_epoch)
    val_set = MultiViewDataset(n_views=args.max_obs + 1, n_samples=args.batch_size)
    model = GQN(c_dim=3, v_dim=train_set.v_dim, r_dim=args.r_dim, h_dim=args.h_dim, z_dim=args.z_dim, l=args.l)

    train(args.run_name, forward_func, sample_func, model, train_set, val_set,
          args.n_epochs, args.batch_size,
          args.lr_i, args.lr_f, args.lr_n, args.sig_i, args.sig_f, args.sig_n)
