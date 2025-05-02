from architectures import VAE, most_likely_predicted_state
from boards import FoggedBoard, MaterialCounter, fogged_board_to_tensor
from evaluation import RandomBaselinePlayer, piece_type_accuracy, piece_location_iou_accuracy

import chess
import os
import random
import torch
from typing import List, Tuple

def average(data: List[float]) -> float:
    length = len(data)
    return sum(element / length for element in data)

def test_vae_accuracy(vae_model: VAE, num_games: int, seed: int) -> Tuple[float, float]:
    vae_type_accuracy = []
    vae_iou_accuracy = []
    move_selector = RandomBaselinePlayer(seed)
    for game in range(num_games):
        board = chess.Board()
        while True:
            move_list = list(FoggedBoard.generate_fow_chess_moves(board))
            selected_move = move_selector.choose_move(move_list)

            fogged_board = FoggedBoard.derived_from_full_state(board)
            fogged_tensor = fogged_board_to_tensor(fogged_board)

            defogged_state = most_likely_predicted_state(
                vae_model(fogged_tensor[:13,:,:])[0],
                fogged_tensor,
                fogged_board.hidden_material
            )
            vae_type_accuracy.append(
                piece_type_accuracy(defogged_state, board, fogged_board))
            vae_iou_accuracy.append(
                piece_location_iou_accuracy(defogged_state, board, fogged_board))

            board.push(selected_move)

            # Check for conclusion (White win, Black win, draw)
            current_material = MaterialCounter.material_in_board(board)
            if current_material.black_kings == 0:
                break
            elif current_material.white_kings == 0:
                break
            elif board.is_fifty_moves() or board.is_repetition():
                break
    return average(vae_type_accuracy), average(vae_iou_accuracy)

if __name__ == '__main__':
    vae = VAE(512)
    vae.load_state_dict(torch.load(
        os.path.join(os.getcwd(), 'models', 'vae-dirichletalpha0.15-12.pt'),
        weights_only=True
    ))
    print(test_vae_accuracy(vae, 10, 314159))
