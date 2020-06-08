import os
import shutil
import time
from datetime import timedelta
import math
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from models.simple.SimpleVVGQN import SimpleVVGQN

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')


def train(run_name, forward_func, model, train_set, val_set,
          n_epochs, batch_size, lr):

    # Make the run directory
    save_dir = os.path.join('training/simple/saved_runs', run_name)
    if run_name == 'debug':
        shutil.rmtree(save_dir, ignore_errors=True)
    os.mkdir(save_dir)

    model = model.to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training step
    def step(engine, batch):
        model.train()

        if isinstance(batch, list):
            batch = [tensor.to(device) for tensor in batch]
        else:
            batch = batch.to(device)
        x_rec, x_q, _ = forward_func(model, batch)

        loss = F.l1_loss(x_rec, x_q)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return {'L1': loss}

    # Trainer and metrics
    trainer = Engine(step)
    metric_names = ['L1']
    RunningAverage(output_transform=lambda x: x['L1']).attach(trainer, 'L1')
    ProgressBar().attach(trainer, metric_names=metric_names)
    Timer(average=True).attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                               pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # Model checkpointing
    checkpoint_handler = ModelCheckpoint(os.path.join(save_dir, 'checkpoints'), type(model).__name__,
                                         save_interval=1, n_saved=3, require_empty=False)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={'model': model, 'optimizer': optimizer})

    # Tensorbard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_metrics(engine):
        if engine.state.iteration % 100 == 0:
            for metric, value in engine.state.metrics.items():
                writer.add_scalar('training/{}'.format(metric), value, engine.state.iteration)

    def save_images(engine, batch):
        x_rec, x_q, r = forward_func(model, batch)
        r_dim = r.shape[1]
        if isinstance(model, SimpleVVGQN):
            r = (r + 1) / 2
        r = r.view(-1, 1, int(math.sqrt(r_dim)), int(math.sqrt(r_dim)))

        x_rec = x_rec.detach().cpu().float()
        r = r.detach().cpu().float()

        writer.add_image('representation', make_grid(r), engine.state.epoch)
        writer.add_image('generation', make_grid(x_rec), engine.state.epoch)
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
            x_rec, x_q, r = forward_func(model, batch)

            loss = F.l1_loss(x_rec, x_q)

            writer.add_scalar('validation/L1', loss.item(), engine.state.epoch)

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

    start_time = time.time()
    trainer.run(train_loader, n_epochs)
    writer.close()
    end_time = time.time()
    print('Total training time: {}'.format(timedelta(seconds=end_time - start_time)))
