from architectures import ConvNet, VAE, PuctConfig
from training import OptimConfig, TrainingConfig, training_step

import numpy as np
import os
import random
import torch
from typing import Optional, Tuple

def training_with_load(
        start: Tuple[str, str], end: Tuple[str, str],
        training_config, puct_config, convnet_optim, defogger_optim):

    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)

    conv = ConvNet()
    vae = VAE(512)

    conv.load_state_dict(
        torch.load(os.path.join(os.getcwd(), 'models', start[0]),
        weights_only=True))
    vae.load_state_dict(
        torch.load(os.path.join(os.getcwd(), 'models', start[1]),
        weights_only=True))

    conv, vae = training_step(
        (conv, vae),
        config=training_config, puct_config=puct_config,
        convnet_optim=convnet_optim, defogger_optim=defogger_optim)

    torch.save(conv.state_dict(), os.path.join(os.getcwd(), 'models', end[0]))
    torch.save(vae.state_dict(), os.path.join(os.getcwd(), 'models', end[1]))

if __name__ == '__main__':
    random.seed(19937)
    np.random.seed(19937)
    torch.manual_seed(19937)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    conv = ConvNet()
    vae = VAE(512)

    # Start off with saving the very first untrained model.
    # torch.save(conv.state_dict(), os.path.join(os.getcwd(), 'models', 'convnet-0.pt'))
    # torch.save(vae.state_dict(), os.path.join(os.getcwd(), 'models', 'vae-0.pt'))

    # Initial hyperparameter settings.
    initial_training_config = TrainingConfig(
        sample_games=1,
        vae_epochs=20,
        convnet_epochs=20,
        possibilities=1,
        simulations=200,
        move_limit=128,
        seed=19937,
    )
    initial_puct_config = PuctConfig(
        c_puct=0.5,
        dirichlet_alpha=0.3,
        epsilon=0.25,
        move_limit=128,
    )
    initial_convnet_optim = OptimConfig(lr=1e-4, weight_decay=1e-6)
    initial_defogger_optim = OptimConfig(lr=1e-4, weight_decay=1e-6)

    setting = 'initial'
    for step in range(3):
        prev_model = (('convnet-0.pt', 'vae-0.pt') if step == 0
            else (f'convnet-{setting}-{step}.pt', f'vae-{setting}-{step}.pt'))
        training_with_load(
            prev_model,
            (f'convnet-{setting}-{step + 1}.pt', f'vae-{setting}-{step + 1}.pt'),
            initial_training_config,
            initial_puct_config,
            initial_convnet_optim,
            initial_defogger_optim)

    # Change `c_puct` to `1.0`.
    puct_config = PuctConfig(
        **(initial_puct_config._asdict() | {'c_puct': 1.0}))

    setting = 'cpuct1.0'
    for step in range(3):
        prev_model = (('convnet-0.pt', 'vae-0.pt') if step == 0
            else (f'convnet-{setting}-{step}.pt', f'vae-{setting}-{step}.pt'))
        training_with_load(
            prev_model,
            (f'convnet-{setting}-{step + 1}.pt', f'vae-{setting}-{step + 1}.pt'),
            initial_training_config,
            puct_config,
            initial_convnet_optim,
            initial_defogger_optim)

    # Change `dirichlet_alpha` to `0.15`.
    puct_config = PuctConfig(
        **(initial_puct_config._asdict() | {'dirichlet_alpha': 0.15}))

    setting = 'dirichletalpha0.15'
    for step in range(3):
        prev_model = (('convnet-0.pt', 'vae-0.pt') if step == 0
            else (f'convnet-{setting}-{step}.pt', f'vae-{setting}-{step}.pt'))
        training_with_load(
            prev_model,
            (f'convnet-{setting}-{step + 1}.pt', f'vae-{setting}-{step + 1}.pt'),
            initial_training_config,
            puct_config,
            initial_convnet_optim,
            initial_defogger_optim)

    # Change `epsilon` to `0.5`.
    puct_config = PuctConfig(
        **(initial_puct_config._asdict() | {'epsilon': 0.5}))

    setting = 'epsilon0.5'
    for step in range(3):
        prev_model = (('convnet-0.pt', 'vae-0.pt') if step == 0
            else (f'convnet-{setting}-{step}.pt', f'vae-{setting}-{step}.pt'))
        training_with_load(
            prev_model,
            (f'convnet-{setting}-{step + 1}.pt', f'vae-{setting}-{step + 1}.pt'),
            initial_training_config,
            puct_config,
            initial_convnet_optim,
            initial_defogger_optim)

    # Change `weight_decay` to `0`.
    convnet_optim = OptimConfig(
        **(initial_convnet_optim._asdict() | {'weight_decay': 0}))
    defogger_optim = OptimConfig(
        **(initial_defogger_optim._asdict() | {'weight_decay': 0}))

    setting = 'weightdecay0'
    for step in range(3):
        prev_model = (('convnet-0.pt', 'vae-0.pt') if step == 0
            else (f'convnet-{setting}-{step}.pt', f'vae-{setting}-{step}.pt'))
        training_with_load(
            prev_model,
            (f'convnet-{setting}-{step + 1}.pt', f'vae-{setting}-{step + 1}.pt'),
            initial_training_config,
            initial_puct_config,
            convnet_optim,
            defogger_optim)
