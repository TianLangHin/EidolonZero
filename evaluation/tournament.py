from architectures import ConvNet, VAE, PuctConfig, most_likely_predicted_state, p_uct
from boards import FoggedBoard, FullState, MaterialCounter, \
    fogged_board_to_tensor, move_gen_to_tensor, move_policy_to_tensor, tensor_to_move_gen
from evaluation import RandomBaselinePlayer

from collections import namedtuple
from time import perf_counter
import torch
from typing import List, Optional, Tuple

PlayConfig = namedtuple('PlayConfig', ['puct_config', 'possibilities', 'simulations'])

def game_status(game_state: FullState) -> Optional[str]:
    material_count = MaterialCounter.material_in_board(game_state)
    if material_count.white_kings == 0:
        return -1 # Black win
    elif material_count.black_kings == 0:
        return 1 # White win
    elif game_state.is_fifty_moves() or game_state.is_repetition():
        return 0 # Draw
    else:
        return None

def against_baseline(
        random_baseline: RandomBaselinePlayer, model: Tuple[ConvNet, VAE],
        *, play_config: PlayConfig, random_as_white: bool,
        verbose: bool = True) -> Tuple[List[str], int]:

    convnet_model, defogger_model = model
    puct_config, possibilities, simulations = play_config

    game_state = FullState()
    game_history = []
    ply_number = 1

    if verbose:
        print('Game Start')

    while (status := game_status(game_state)) is None:

        # In both cases, we need to determine all the legal moves available.
        legal_move_list = list(FoggedBoard.generate_fow_chess_moves(game_state))

        # If it is random baseline turn to play, choose the move.
        if (ply_number % 2 == 1) == random_as_white:
            move = random_baseline.choose_move(legal_move_list)
            if verbose:
                print(f'Ply {ply_number}, Baseline: {move.uci()}')
        else:
            # For the bot to work, it needs to know the legal moves available as a tensor.
            legal_move_tensor = move_gen_to_tensor((move for move in legal_move_list), game_state.turn)
            # We then derive the board for it to play from.
            fogged_board = FoggedBoard.derived_from_full_state(game_state)
            fogged_tensor = fogged_board_to_tensor(fogged_board)

            s = perf_counter()

            summed_move_policy = torch.zeros(torch.Size([73, 8, 8]))
            # Iterate over all possibilities
            for _ in range(possibilities):
                # Conduct prediction from VAE.
                defogged_state = most_likely_predicted_state(
                    defogger_model(fogged_tensor[:13,:,:])[0],
                    fogged_tensor,
                    fogged_board.hidden_material)

                # Now conduct the P-UCT search using the ConvNet.
                search_result = p_uct(defogged_state, legal_move_tensor,
                    simulations, convnet_model, puct_config)

                # Extract the visit count of each move, then make the distribution.
                total_visit_count = sum(stats.visit_count for stats in search_result.values())
                move_policy_generator = (
                    (action, stats.visit_count / total_visit_count)
                    for action, stats in search_result.items() if action in legal_move_list)
                move_policy_tensor = move_policy_to_tensor(move_policy_generator, game_state.turn)
                if move_policy_tensor.sum() == 0:
                    move_policy_tensor += move_gen_to_tensor((move for move in [legal_move_list[0]]), game_state.turn)

                # Add this to the policy considering all possibilities.
                summed_move_policy += move_policy_tensor

            e = perf_counter()

            # Choose the move most explored.
            move = next(tensor_to_move_gen(
                (summed_move_policy == summed_move_policy.max()).int(),
                position=game_state
            ))

            if verbose:
                print(f'Ply {ply_number}, Model: {move.uci()}')
                print(f'Time elapsed: {e-s} seconds')

        # Update the overall game state.
        game_state.push(move)
        game_history.append(move.uci())
        ply_number += 1

        if verbose:
            print(game_state)

    return game_history, status

def head_to_head(
        model_white: Tuple[ConvNet, VAE], model_black: Tuple[ConvNet, VAE],
        play_config_white: PlayConfig, play_config_black: PlayConfig) -> Tuple[List[str], int]:

    game_state = FullState()
    game_history = []
    ply_number = 1

    while (status := game_status(game_state)) is None:

        playing_model = model_white if ply_number % 2 == 1 else model_black
        convnet, defogger = playing_model

        legal_move_list = list(FoggedBoard.generate_fow_chess_moves(game_state))
        legal_move_tensor = move_gen_to_tensor((move for move in legal_move_list), game_state.turn)
        fogged_board = FoggedBoard.derived_from_full_state(game_state)
        fogged_tensor = fogged_board_to_tensor(fogged_board)

        summed_move_policy = torch.zeros(torch.Size([73, 8, 8]))
        # Iterate over all possibilities
        for _ in range(possibilities):
            # Conduct prediction from VAE.
            defogged_state = most_likely_predicted_state(
                defogger(fogged_tensor[:13,:,:])[0],
                fogged_tensor,
                fogged_board.hidden_material)

            # Now conduct the P-UCT search using the ConvNet.
            search_result = p_uct(defogged_state, legal_move_tensor,
                simulations, convnet, puct_config)

            # Extract the visit count of each move, then make the distribution.
            total_visit_count = sum(stats.visit_count for stats in search_result.values())
            move_policy_generator = (
                (action, stats.visit_count / total_visit_count)
                for action, stats in search_result.items() if action in legal_move_list)
            move_policy_tensor = move_policy_to_tensor(move_policy_generator, game_state.turn)
            if move_policy_tensor.sum() == 0:
                move_policy_tensor += move_gen_to_tensor((move for move in [legal_move_list[0]]), game_state.turn)

            # Add this to the policy considering all possibilities.
            summed_move_policy += move_policy_tensor

        move = next(tensor_to_move_gen(
            (summed_move_policy == summed_move_policy.max()).int(),
            position=game_state
        ))

        try:
            game_state.push(move)
        except AssertionError:
            game_state.push(move := legal_move_list[0])
        game_history.append(move.uci())
        ply_number += 1

    return game_history, status
