from architectures import ConvNet, VAE, PuctConfig
from evaluation import PlayConfig, RandomBaselinePlayer, against_baseline, head_to_head

import argparse
import numpy as np
import os
import random
import torch
from typing import Dict, List, Tuple

FORMAT = '''
[White "{0}"]
[Black "{1}"]
[Result "{2}"]

1. c4 Nf6 {{ {3} }}
'''

def tourney_against_baseline(
        models_and_config: List[Tuple[ConvNet, VAE, PlayConfig, str]],
        games: int, seed: int, seed_increment: int, filename: str, verbose: bool):

    with open(filename, 'wt') as f:
        for conv, vae, play_config, name in models_and_config:
            # This random seed is for the AI model, and does not affect baseline.
            random.seed(19937)
            np.random.seed(19937)
            torch.manual_seed(19937)

            conv.eval()
            vae.eval()

            game_seed = seed
            for _ in range(games):
                baseline = RandomBaselinePlayer(game_seed)
                hist, outcome = against_baseline(
                    baseline, (conv, vae), play_config=play_config, random_as_white=True,
                    verbose=verbose)
                result = ('1-0' if outcome == 1.0 else '0-1' if outcome == -1.0 else '1/2-1/2')
                f.write(FORMAT.format(f'Random-{game_seed}', name, result, ' '.join(hist)))
                f.flush()
                game_seed += seed_increment

            game_seed = seed
            for _ in range(games):
                baseline = RandomBaselinePlayer(game_seed)
                hist, outcome = against_baseline(
                    baseline, (conv, vae), play_config=play_config, random_as_white=False,
                    verbose=verbose)
                result = ('1-0' if outcome == 1.0 else '0-1' if outcome == -1.0 else '1/2-1/2')
                f.write(FORMAT.format(name, f'Random-{game_seed}', result, ' '.join(hist)))
                f.flush()
                game_seed += seed_increment

def tourney_within_models(
        models_and_config: List[Tuple[ConvNet, VAE, PlayConfig, str]],
        games: int, filename: str, verbose: bool):

    length = len(models_and_config)
    with open(filename, 'wt') as f:
        for model_number in range(length):
            random.seed(19937)
            np.random.seed(19937)
            torch.manual_seed(19937)

            conv1, vae1, play_config1, name1 = models_and_config[model_number % length]
            conv2, vae2, play_config2, name2 = models_and_config[(model_number+1) % length]

            conv1.eval()
            vae1.eval()
            conv2.eval()
            vae2.eval()

            for _ in range(games):
                hist, outcome = head_to_head(
                    (conv1, vae1), (conv2, vae2), play_config1, play_config2, verbose=verbose)
                result = ('1-0' if outcome == 1.0 else '0-1' if outcome == -1.0 else '1/2-1/2')
                f.write(FORMAT.format(name1, name2, result, ' '.join(hist)))
                f.flush()

            for _ in range(games):
                hist, outcome = head_to_head(
                    (conv2, vae2), (conv1, vae1), play_config2, play_config1, verbose=verbose)
                result = ('1-0' if outcome == 1.0 else '0-1' if outcome == -1.0 else '1/2-1/2')
                f.write(FORMAT.format(name2, name1, result, ' '.join(hist)))
                f.flush()

def loaded_initial() -> Tuple[ConvNet, VAE]:
    conv = ConvNet()
    vae = VAE(512)

    conv.load_state_dict(
        torch.load(
            os.path.join(os.getcwd(), 'models', f'convnet-0.pt'),
            weights_only=True))
    vae.load_state_dict(
        torch.load(
            os.path.join(os.getcwd(), 'models', f'vae-0.pt'),
            weights_only=True))
    return conv, vae

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

def adjusted_config(
        initial_play: PlayConfig,
        initial_puct: PuctConfig,
        change: Dict[str, float]) -> PlayConfig:
    play_dict, puct_dict = initial_play._asdict(), initial_puct._asdict()
    updated_puct = PuctConfig(**(puct_dict | change))
    return PlayConfig(**(play_dict | {'puct_config': updated_puct}))

INITIAL_PUCT_CONFIG = PuctConfig(
    c_puct=0.5,
    dirichlet_alpha=0.3,
    epsilon=0.25,
    move_limit=128)

INITIAL_PLAY_CONFIG = PlayConfig(
    puct_config=INITIAL_PUCT_CONFIG,
    possibilities=1,
    simulations=200)

CONFIGS = {
    'initial': INITIAL_PLAY_CONFIG,
    'cpuct1.0': adjusted_config(
        INITIAL_PLAY_CONFIG, INITIAL_PUCT_CONFIG, {'c_puct': 1.0}),
    'dirichletalpha0.15': adjusted_config(
        INITIAL_PLAY_CONFIG, INITIAL_PUCT_CONFIG, {'dirichlet_alpha': 0.15}),
    'epsilon0.5': adjusted_config(
        INITIAL_PLAY_CONFIG, INITIAL_PUCT_CONFIG, {'epsilon': 0.5}),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'file',
        help='Output file of game histories')
    parser.add_argument(
        '-b', '--baseline',
        action='store_true',
        help='Specify whether each model is playing against the baseline or one another')
    parser.add_argument(
        '-m', '--model',
        choices=['initial', 'cpuct1.0', 'dirichletalpha0.15', 'epsilon0.5'],
        help='The model hyperparameter setting to test')
    parser.add_argument(
        '-g', '--games',
        type=int, default=5,
        help='The number of games of each colour that each model should play')
    parser.add_argument(
        '-s', '--seed',
        type=int, default=19937,
        help='Starting random seed for the random baseline player')
    parser.add_argument(
        '-i', '--inc',
        type=int, default=1,
        help='Seed increments over each game for the random baseline player')
    parser.add_argument(
        '--silent',
        action='store_true',
        help='If set, silences the verbosity of outputs to the terminal')

    args = parser.parse_args()
    model_and_config_list = [(*loaded_initial(), CONFIGS['initial'], 'EidolonZero-untrained')] + [
        (*loaded_models(args.model, i),
            CONFIGS[args.model], f'EidolonZero-{args.model}-{i}')
        for i in range(3, 13, 3)
    ]
    if args.baseline:
        tourney_against_baseline(model_and_config_list,
            args.games, args.seed, args.inc, args.file, not args.silent)
    else:
        tourney_within_models(model_and_config_list,
            args.games, args.file, not args.silent)
