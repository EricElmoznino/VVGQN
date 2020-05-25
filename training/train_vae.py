from argparse import ArgumentParser
from training.trainer import train
from models.VAE import VAE
from datasets.SingleViewDataset import SingleViewDataset


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
    args = parser.parse_args()

    def forward_func(model, batch):
        x = batch
        x_mu, _, kl = model(x)
        return x_mu, x, kl


    def sample_func(model, batch):
        x = batch
        x_mu, r = model.sample(x)
        return x_mu, x, r

    model = VAE(c_dim=3, r_dim=args.r_dim, h_dim=args.h_dim, z_dim=args.z_dim, l=args.l)
    train_set = SingleViewDataset(n_samples=args.samples_per_epoch)
    val_set = SingleViewDataset(n_samples=args.batch_size)

    train(args.run_name, forward_func, sample_func, model, train_set, val_set,
          args.n_epochs, args.batch_size, args.lr)
