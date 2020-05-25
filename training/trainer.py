import os
import shutil
import math
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from . import utils

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')


def train(run_name, forward_func, sample_func, model, train_set, val_set,
          n_epochs, batch_size, lr):
    # Make the run directory
    save_dir = os.path.join('training/saved_runs', run_name)
    if run_name == 'debug':
        shutil.rmtree(save_dir, ignore_errors=True)
    os.mkdir(save_dir)

    model = model.to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = utils.AnnealingStepLR(optimizer, mu_i=lr, mu_f=lr/10, n=1.6e6)
    sigma_scheduler = utils.AnnealingStepSigma(2.0, 0.7, 2e5)

    # Training step
    def step(engine, batch):
        model.train()

        if isinstance(batch, list):
            batch = [tensor.to(device) for tensor in batch]
        else:
            batch = batch.to(device)
        x_mu, x_q, kl = forward_func(model, batch)

        # Log likelihood
        sigma = sigma_scheduler.sigma
        lr = lr_scheduler.get_lr()[0]
        ll = Normal(x_mu, sigma).log_prob(x_q)

        likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
        kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

        # Evidence lower bound
        elbo = likelihood - kl_divergence
        loss = -elbo
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        lr_scheduler.step()
        sigma_scheduler.step()

        return {'elbo': elbo.item(), 'likelihood': likelihood.item(), 'kl': kl_divergence.item(),
                'lr': lr, 'sigma': sigma}

    # Trainer and metrics
    trainer = Engine(step)
    metric_names = ['elbo', 'likelihood', 'kl', 'lr', 'sigma']
    RunningAverage(output_transform=lambda x: x['elbo']).attach(trainer, 'elbo')
    RunningAverage(output_transform=lambda x: x['likelihood']).attach(trainer, 'likelihood')
    RunningAverage(output_transform=lambda x: x['kl']).attach(trainer, 'kl')
    RunningAverage(output_transform=lambda x: x['lr']).attach(trainer, 'lr')
    RunningAverage(output_transform=lambda x: x['sigma']).attach(trainer, 'sigma')
    ProgressBar().attach(trainer, metric_names=metric_names)
    Timer(average=True).attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                               pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # Model checkpointing
    checkpoint_handler = ModelCheckpoint(os.path.join(save_dir, 'checkpoints'), type(model).__name__,
                                         save_interval=1, n_saved=3, require_empty=False)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={'model': model, 'optimizer': optimizer,
                                       'lr_scheduler': lr_scheduler, 'sigma_scheduler': sigma_scheduler})

    # Tensorbard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_metrics(engine):
        if engine.state.iteration % 100 == 0:
            for metric, value in engine.state.metrics.items():
                writer.add_scalar('training/{}'.format(metric), value, engine.state.iteration)

    def save_images(engine, batch):
        x_mu, x_q, r = sample_func(model, batch)
        r_dim = r.shape[1]
        r = r.view(-1, 1, int(math.sqrt(r_dim)), int(math.sqrt(r_dim)))

        x_mu = x_mu.detach().cpu().float()
        r = r.detach().cpu().float()

        writer.add_image('representation', make_grid(r), engine.state.epoch)
        writer.add_image('generation', make_grid(x_mu), engine.state.epoch)
        writer.add_image('query', make_grid(x_q), engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            if isinstance(batch, list):
                batch = [tensor.to(device) for tensor in batch]
            else:
                batch = batch.to(device)
            x_mu, x_q, kl = forward_func(model, batch)

            # Validate at last sigma
            ll = Normal(x_mu, sigma_scheduler.sigma).log_prob(x_q)

            likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
            kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

            # Evidence lower bound
            elbo = likelihood - kl_divergence

            writer.add_scalar('validation/elbo', elbo.item(), engine.state.epoch)
            writer.add_scalar('validation/likelihood', likelihood.item(), engine.state.epoch)
            writer.add_scalar('validation/kl', kl_divergence.item(), engine.state.epoch)

            save_images(engine, batch)

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            checkpoint_handler(engine, {'model_exception': model})
        else:
            raise e

    trainer.run(train_loader, n_epochs)
    writer.close()
