from architectures import ConvNet, VAE, most_likely_predicted_state,p_uct
from boards import FoggedBoard, MaterialCounter, \
    fogged_board_to_tensor, move_gen_to_tensor, move_policy_to_tensor, tensor_to_move_gen
from evaluation import PlayConfig

import chess
from itertools import product
import torch
from typing import Dict, Optional, Tuple

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

def getfoggedstate(fen: str, invert: bool) -> Optional[Dict]:
    try:
        board = chess.Board(fen)
        derived_board = FoggedBoard.derived_from_full_state(board)
        true_fen = derived_board.fogged_board_state.fen()
        if invert:
            board.turn = not board.turn
        fogged_board = FoggedBoard.derived_from_full_state(board)
        fogged = fogged_board.fogged_board_state
        hidden_material = derived_board.hidden_material
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
            'fen': true_fen,
            'visible': {
                file_str + rank_str: (vision >> sq) & 1 != 0
                for sq, (rank_str, file_str) in enumerate(product(ranks, files))
            },
            'squares': {
                file_str + rank_str: piece_string(board, sq)
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

def inference(
    fen: str, material: MaterialCounter,
    play_config: PlayConfig, model: Tuple[ConvNet, VAE]) -> Optional[Dict]:

    try:
        convnet, vae = model
        convnet.eval()
        vae.eval()

        board = chess.Board(fen)
        files, ranks = 'abcdefgh', '12345678'

        legal_move_list = list(FoggedBoard.generate_fow_chess_moves(board))
        legal_move_tensor = move_gen_to_tensor((move for move in legal_move_list), board)

        fogged_board = FoggedBoard(board, material)
        fogged_board_tensor = fogged_board_to_tensor(fogged_board)

        summed_move_policy = torch.zeros(torch.Size([73, 8, 8]))
        predicted_board = None
        for _ in range(play_config.possibilities):
            defogged_state = most_likely_predicted_state(
                vae(fogged_board_tensor[:13,:,:])[0],
                fogged_board_tensor,
                material
            )
            if predicted_board is None:
                predicted_board = defogged_state

            search_result = p_uct(defogged_state, legal_move_tensor,
                play_config.simulations, convnet, play_config.puct_config)

            total_visit_count = sum(stats.visit_count for stats in search_result.values())
            move_policy_generator = ((action, stats.visit_count / total_visit_count)
                for action, stats in search_result.items() if action in legal_move_list)
            move_policy_tensor = move_policy_to_tensor(move_policy_generator, board.turn)
            if move_policy_tensor.sum() == 0:
                move_policy_tensor += move_gen_to_tensor(
                    (move for move in [legal_move_list[0]]), board.turn)
            summed_move_policy += move_policy_tensor

        move = next(tensor_to_move_gen(
            (summed_move_policy == summed_move_policy.max()).int(),
            position=board
        ))

        return {
            'predicted_board': {
                'fen': predicted_board.fen(),
                'squares': {
                    file_str + rank_str: piece_string(predicted_board, sq)
                    for sq, (rank_str, file_str) in enumerate(product(ranks, files))
                }
            },
            'move': move.uci()
        }
    except ValueError:
        return None

def makemove(fen: str, move: str) -> Optional[Dict]:
    try:
        board = chess.Board(fen)
        board.push(chess.Move.from_uci(move))
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
