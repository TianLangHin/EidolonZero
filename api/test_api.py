import chess
import torch
from itertools import product
from typing import Dict, List, Optional

from boards import FoggedBoard, move_gen_to_tensor, position_to_tensor, tensor_to_position

# We define some test API handles to show that the web server API works together
# with the packages we plan on using

def legal_chess_moves_from_fen(fen: str) -> Optional[List[chess.Move]]:
    try:
        board = chess.Board(fen)
        return [m.uci() for m in board.legal_moves]
    except ValueError:
        return None

def fow_chess_move_tensor(fen: str) -> Optional[Dict[str, int]]:
    try:
        board = chess.Board(fen)
        move_tensor = move_gen_to_tensor(
            FoggedBoard.generate_fow_chess_moves(board),
            board.turn)
        return {
            f'{stack}-{r}-{f}': move_tensor[stack][r][f].item()
            for stack, r, f in product(range(73), range(8), range(8))
        }
    except ValueError:
        return None

def sanity_check_position_tensor(fen: str) -> Optional[bool]:
    try:
        board = chess.Board(fen)
        reconstructed = tensor_to_position(position_to_tensor(board))
        return set(board.legal_moves) == set(reconstructed.legal_moves)
    except ValueError:
        return None
