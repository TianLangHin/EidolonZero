from architectures import ConvNet, VAE, PuctConfig
from evaluation import PlayConfig, RandomBaselinePlayer, against_baseline

import numpy as np
import os
import random
import torch
from typing import List, Tuple

FORMAT = '''
[White "{0}"]
[Black "{1}"]
[Result "{2}"]

1. c4 Nf6 {{ {3} }}
'''

def tourney_against_baseline(
        models_and_config: List[Tuple[ConvNet, VAE, PlayConfig, str]],
        games: int, seed: int, filename: str):

    with open(filename, 'wt') as f:

        for conv, vae, play_config, name in models_and_config:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            baseline = RandomBaselinePlayer(seed)

            conv.eval()
            vae.eval()

            for i in range(games):
                hist, outcome = against_baseline(
                    baseline, (conv, vae), play_config=play_config, random_as_white=True,
                    verbose=False)
                result = ('1-0' if outcome == 1.0 else '0-1' if outcome == -1.0 else '1/2-1/2')
                f.writeline(FORMAT.format(f'Random-{seed}', name, result, ' '.join(hist)))
                f.flush()

            for i in range(games):
                hist, outcome = against_baseline(
                    baseline, (conv, vae), play_config=play_config, random_as_white=False,
                    verbose=False)
                result = ('1-0' if outcome == 1.0 else '0-1' if outcome == -1.0 else '1/2-1/2')
                f.writeline(FORMAT.format(name, f'Random-{seed}', result, ' '.join(hist)))
                f.flush()

def loaded_models(variant: str, iteration: int) -> Tuple[ConvNet, VAE]:
    conv = ConvNet()
    vae = VAE(512)

    conv.load_state_dict(
        torch.load(
            os.path.join(os.getcwd(), 'models', f'convnet-{variant}-{iteration}.pt'),
            weights_only=True))
    vae.load_state_dict(
        torch.load(
            os.path.join(os.getcwd(), 'models', f'vae-{variant}-{iteration}.pt'),
            weights_only=True))
    return conv, vae

if __name__ == '__main__':
    initial_puct_config = PuctConfig(
        c_puct=0.5,
        dirichlet_alpha=0.3,
        epsilon=0.25,
        move_limit=128)
    initial_play_config = PlayConfig(
        puct_config=initial_puct_config,
        possibilities=1,
        simulations=200)
    configs = {
        'cpuct1.0': PlayConfig(
            **(initial_play_config._asdict() | {
                'puct_config': PuctConfig(
                    **(initial_puct_config._asdict() | {
                        'c_puct': 1.0
                    })
                )
            })
        ),
        'dirichletalpha0.15': PlayConfig(
            **(initial_play_config._asdict() | {
                'puct_config': PuctConfig(
                    **(initial_puct_config._asdict() | {
                        'dirichlet_alpha': 0.15
                    })
                )
            })
        ),
        'epsilon0.5': PlayConfig(
            **(initial_play_config._asdict() | {
                'puct_config': PuctConfig(
                    **(initial_puct_config._asdict() | {
                        'epsilon': 0.5
                    })
                )
            })
        ),
        'weightdecay0': initial_play_config
    }
    models_and_config = [
        (*loaded_models(variant, i), configs[variant], f'{variant}-{i}')
        for variant in configs
        for i in range(1, 4)
    ]

    tourney_against_baseline(models_and_config, 10, 19937, 'game.txt')
