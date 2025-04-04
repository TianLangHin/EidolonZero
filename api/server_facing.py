import chess
from itertools import product
from typing import Dict, Optional

from boards import FoggedBoard, MaterialCounter

def piece_string(board: chess.Board, square: int) -> Optional[str]:
    colour = board.color_at(square)
    match board.piece_type_at(square):
        case chess.PAWN:
            return 'P' if colour else 'p'
        case chess.KNIGHT:
            return 'N' if colour else 'n'
        case chess.BISHOP:
            return 'B' if colour else 'b'
        case chess.ROOK:
            return 'R' if colour else 'r'
        case chess.QUEEN:
            return 'Q' if colour else 'q'
        case chess.KING:
            return 'K' if colour else 'k'
        case _:
            return None

def stub_getfoggedstate(fen: str) -> Optional[Dict]:
    try:
        board = chess.Board(fen)
        fogged_board = FoggedBoard.derived_from_full_state(board)
        fogged = fogged_board.fogged_board_state
        hidden_material = fogged_board.hidden_material
        vision = FoggedBoard.get_visible_squares(board,
            list(FoggedBoard.generate_fow_chess_moves(board)))
        files, ranks = 'abcdefgh', '12345678'
        args_translate = {
            'white_pawns': 'wp',
            'black_pawns': 'bp',
            'white_knights': 'wn',
            'black_knights': 'bn',
            'white_bishops': 'wb',
            'black_bishops': 'bb',
            'white_rooks': 'wr',
            'black_rooks': 'br',
            'white_queens': 'wq',
            'black_queens': 'bq',
            'white_kings': 'wk',
            'black_kings': 'bk'
        }
        return {
            'fen': fogged.fen(),
            'visible': {
                file_str + rank_str: (vision >> sq) & 1 != 0
                for sq, (rank_str, file_str) in enumerate(product(ranks, files))
            },
            'material': {
                args_translate[attr]: getattr(hidden_material, attr)
                for attr in args_translate.keys()
            }
        }
    except ValueError:
        return None

def stub_inference(fen: str, material: MaterialCounter) -> Optional[Dict]:
    try:
        board = chess.Board(fen)
        try:
            move = next(board.generate_legal_moves())
        except StopIteration:
            move = chess.Move(chess.E2, chess.E4)
        # Makes some model call here
        files, ranks = 'abcdefgh', '12345678'
        return {
            'predicted_board': {
                # Normally there is some inference here,
                # not just outputting the original.
                'fen': board.fen(),
                'squares': {
                    file_str + rank_str: piece_string(board, sq)
                    for sq, (rank_str, file_str) in enumerate(product(ranks, files))
                }
            },
            'move': move.uci()
        }
    except ValueError:
        return None

def stub_makemove(fen: str, move: str) -> Optional[Dict]:
    try:
        board = chess.Board(fen)
        board.push_uci(move)
        files, ranks = 'abcdefgh', '12345678'
        return {
            'new_board': {
                'fen': board.fen(),
                'squares': {
                    file_str + rank_str: piece_string(board, sq)
                    for sq, (rank_str, file_str) in enumerate(product(ranks, files))
                }
            }
        }
    except ValueError:
        return None
