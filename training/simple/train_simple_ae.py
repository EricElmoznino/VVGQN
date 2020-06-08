from argparse import ArgumentParser
from training.simple.trainer_simple import train
from models.simple.SimpleAE import SimpleAE
from datasets.SingleViewDataset import SingleViewDataset


if __name__ == '__main__':
    parser = ArgumentParser(description='Simple AE training')
    parser.add_argument('--run_name', required=True, type=str,
                        help='name of the current run (where runs are saved)')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs run')
    parser.add_argument('--samples_per_epoch', type=int, default=10000, help='number of samples per epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--r_dim', type=int, default=256, help='r_dim for model')
    args = parser.parse_args()

    def forward_func(model, batch):
        x = batch
        x_rec, r = model(x)
        return x_rec, x, r

    model = SimpleAE(c_dim=3, r_dim=args.r_dim)
    train_set = SingleViewDataset(n_samples=args.samples_per_epoch)
    val_set = SingleViewDataset(n_samples=args.batch_size)

    train(args.run_name, forward_func, model, train_set, val_set,
          args.n_epochs, args.batch_size, args.lr)
