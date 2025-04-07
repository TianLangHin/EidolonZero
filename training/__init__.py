from architectures import ConvNet, VAE, ConvNetLossFn, DefoggerLossFn, \
    PuctConfig, Stats, p_uct, most_likely_predicted_state
from boards import FoggedBoard, MaterialCounter, fogged_board_to_tensor, position_to_tensor, \
    move_gen_to_tensor, move_policy_to_tensor, tensor_to_position, tensor_to_move_gen

from collections import namedtuple
import chess
import numpy as np
import random
import time
import torch
from typing import List, Tuple

TrainingConfig = namedtuple('TrainingConfig', [
    'sample_games',
    'vae_epochs',
    'convnet_epochs',
    'possibilities',
    'simulations',
    'move_limit',
    'seed',
])

OptimConfig = namedtuple('OptimConfig', [
    'lr',
    'weight_decay'
])

# Will mutate the given model.
# Also returns the sample games for record keeping.
def training_step(
    model: Tuple[ConvNet, VAE],
    *, config: TrainingConfig, puct_config: PuctConfig,
    convnet_optim: OptimConfig, defogger_optim: OptimConfig
) -> Tuple[ConvNet, VAE]:

    sample_games, vae_epochs, convnet_epochs, possibilities, simulations, move_limit, seed = config
    convnet_lr, convnet_weight_decay = convnet_optim
    defogger_lr, defogger_weight_decay = defogger_optim
    convnet_model, defogger_model = model

    convnet_model.eval()
    defogger_model.eval()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    convnet_data = []
    convnet_policy_label = []
    convnet_value_label = []
    defogger_data = []
    defogger_label = []

    for game in range(sample_games):
        print(f'Game {game + 1}', flush=True)

        game_convnet_data = []
        game_convnet_label = []
        game_defogger_data = []
        game_defogger_label = []

        game_outcome = 0.0

        board = chess.Board()

        for move_number in range(move_limit):
            print(f'Move {move_number + 1}', flush=True)

            move_tensor = move_gen_to_tensor(FoggedBoard.generate_fow_chess_moves(board), board.turn)

            fogged_board = FoggedBoard.derived_from_full_state(board)
            fogged_tensor = fogged_board_to_tensor(fogged_board)

            # Record the ground truth board and the fogged board.
            game_defogger_data.append(fogged_tensor[:13,:,:])
            game_defogger_label.append(position_to_tensor(board)[:13,:,:])

            summed_move_policy = torch.zeros(torch.Size([73, 8, 8]))

            for possible_board in range(possibilities):
                print(f'Possibility {possible_board + 1}', flush=True)

                defogged_state = most_likely_predicted_state(
                    defogger_model(fogged_tensor[:13,:,:])[0],
                    fogged_tensor,
                    fogged_board.hidden_material
                )
                print(defogged_state, flush=True)
                print('Search start', flush=True)

                s = time.perf_counter()
                move_search_stats = p_uct(defogged_state, move_tensor, simulations, convnet_model, puct_config)
                e = time.perf_counter()

                print(f'Search time: {e-s} seconds', flush=True)

                total_visit_count = sum(stats.visit_count for stats in move_search_stats.values())
                move_policy_generator = (
                    (action, stats.visit_count / total_visit_count)
                    for action, stats in move_search_stats.items())
                move_policy_tensor = move_policy_to_tensor(move_policy_generator, board.turn)

                summed_move_policy += move_policy_tensor

                # Record the given position and the calculated move policy.
                # Game value is added later.
                game_convnet_data.append(position_to_tensor(defogged_state))
                game_convnet_label.append(move_policy_tensor)

            # Average over all the considered positions.
            chosen_move = next(tensor_to_move_gen(
                (summed_move_policy == summed_move_policy.max()).int(),
                position=board
            ))

            print(f'Chosen move: {chosen_move}', flush=True)

            board.push(chosen_move)
            print(board, flush=True)

            current_material = MaterialCounter.material_in_board(board)
            if current_material.black_kings == 0:
                # Black king has been captured. White wins.
                game_outcome = 1.0
                break
            elif current_material.white_kings == 0:
                # White king has been captured. Black wins.
                game_outcome = -1.0
                break
            elif board.is_fifty_moves() or board.is_repetition():
                # Here, we use the ground state to check for fifty-move rule and repetition for draws.
                game_outcome = 0.0
                break

        print(f'Game outcome: {game_outcome}', flush=True)

        if game_outcome == 0.0:
            game_convnet_value = [0.0] * len(game_convnet_data)
        else:
            game_convnet_value = [
                game_outcome if (b[13] == 0).all().item() else -game_outcome
                for b in game_convnet_data
            ]

        convnet_data.extend(game_convnet_data)
        convnet_policy_label.extend(game_convnet_label)
        convnet_value_label.extend(game_convnet_value)
        defogger_data.extend(game_defogger_data)
        defogger_label.extend(game_defogger_label)

    # Loss functions.
    convnet_loss_fn = ConvNetLossFn()
    opt = torch.optim.Adam(convnet_model.parameters(), lr=convnet_lr, weight_decay=convnet_weight_decay)

    convnet_model.train()
    s = time.perf_counter()
    for epoch in range(convnet_epochs):
        avg_loss = 0
        for data, policy, value in zip(convnet_data, convnet_policy_label, convnet_value_label):
            policy = torch.reshape(policy, torch.Size([1, 4672]))
            value = torch.tensor([value])
            pred_value, pred_policy = convnet_model(torch.reshape(data, (1, *data.shape)))
            loss = convnet_loss_fn(policy, value, pred_policy, pred_value)
            avg_loss += loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(f'ConvNet epoch {epoch + 1}, average loss: {avg_loss / len(convnet_data)}', flush=True)
    e = time.perf_counter()
    print(f'ConvNet training time: {e-s} seconds', flush=True)
    convnet_model.eval()

    defogger_loss_fn = DefoggerLossFn()
    opt = torch.optim.Adam(defogger_model.parameters(), lr=defogger_lr, weight_decay=defogger_weight_decay)

    defogger_model.train()
    s = time.perf_counter()
    for epoch in range(vae_epochs):
        avg_loss = 0
        for data, label in zip(defogger_data, defogger_label):
            board, mu, logvar = defogger_model(data)
            loss = defogger_loss_fn(label, board, mu, logvar)
            avg_loss += loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(f'VAE epoch {epoch + 1}, average loss: {avg_loss / len(defogger_data)}', flush=True)
    e = time.perf_counter()
    print(f'VAE training time: {e-s} seconds', flush=True)
    defogger_model.eval()

    return convnet_model, defogger_model

