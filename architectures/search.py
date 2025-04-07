from boards import FullState as State, FoggedBoard, MaterialCounter, \
    move_gen_to_tensor, position_to_tensor, tensor_to_move_policy

from chess import Move as Action
from collections import namedtuple
from dataclasses import dataclass
import chess
import enum
import numpy as np
import random
import torch
from typing import Callable, Optional, Tuple

@dataclass
class Stats:
    visit_count: int
    total_action_value: float
    mean_action_value: float
    prior_probability: float

PuctConfig = namedtuple('PuctConfig', ['c_puct', 'dirichlet_alpha', 'epsilon', 'move_limit'])

GameOutcome = enum.Enum('GameOutcome', ['WHITE', 'BLACK', 'DRAW'])

def game_outcome(game_state: State) -> Optional[GameOutcome]:
    material_count = MaterialCounter.material_in_board(game_state)
    if material_count.white_kings == 0:
        return GameOutcome.BLACK
    elif material_count.black_kings == 0:
        return GameOutcome.WHITE
    elif (o := game_state.outcome()) is not None and o.result() == '1/2-1/2':
        return GameOutcome.DRAW
    else:
        return None

def uct_selector(stats: Stats, c_puct: float, numerator: int) -> float:
    q = stats.mean_action_value
    u_factor = c_puct * stats.prior_probability
    denominator = 1 + stats.visit_count
    return q + u_factor * numerator / denominator

def p_uct(
    startpos: State,
    startpos_move_tensor: torch.Tensor,
    num_simulations: int,
    predictor: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    config: PuctConfig) -> dict[Action, Stats]:

    c_puct, dirichlet_alpha, epsilon, move_limit = config

    tree: dict[str, dict[Action, Stats]] = {}
    root = startpos

    for simulation in range(num_simulations):
        board = root
        path = []
        found_terminal = False

        moves_ahead = 0

        while (actions := tree.get(board.fen(), None)) is not None and moves_ahead < move_limit:
            if len(actions) == 0:
                found_terminal = True
                break
            numerator = np.sqrt(sum(stats.visit_count for stats in actions.values()))
            best_action = max(actions.items(),
                key=lambda v: uct_selector(v[1], c_puct, numerator))[0]
            path.append((board, best_action))
            # In the event of a zero tensor being outputted somewhere, we add a fail safe here.
            chosen_action = best_action if best_action in board.legal_moves else board.legal_moves[0]
            (board := board.copy()).push(chosen_action)
            moves_ahead += 1

        if found_terminal or moves_ahead == move_limit:
            match (startpos.turn, game_outcome(board)):
                case (chess.WHITE, GameOutcome.WHITE) | (chess.BLACK, GameOutcome.BLACK):
                    value = 1
                case (chess.WHITE, GameOutcome.BLACK) | (chess.BLACK, GameOutcome.WHITE):
                    value = -1
                case _:
                    value = 0
        else:
            board_tensor = position_to_tensor(board)
            value, policy = predictor(torch.reshape(board_tensor, (1, *board_tensor.shape)))
            value = value.item()
            policy = policy.reshape(torch.Size([73, 8, 8]))
            legal_move_tensor = (startpos_move_tensor
                if board == startpos else
                move_gen_to_tensor(
                    FoggedBoard.generate_fow_chess_moves(board), board.turn))
            policy *= legal_move_tensor
            policy /= policy.sum()
            # These are actually probability logits
            policy = tensor_to_move_policy(policy, position=board)
            tree[board.fen()] = {move: Stats(0, 0., 0., prob) for move, prob in policy}

        for num, (state, action) in enumerate(path):
            key = state.fen()
            v = value if board.turn == state.turn else -value

            if num == 0:
                dimension = len(tree[key])
                noise = {a: p for a, p in zip(
                    tree[key].keys(), np.random.dirichlet([dirichlet_alpha] * dimension))}
                p = tree[key][action].prior_probability
                p = (1 - epsilon) * p + epsilon * noise[action]
                tree[key][action].prior_probability = p

            tree[key][action].visit_count += 1
            tree[key][action].total_action_value += v
            tree[key][action].mean_action_value = (tree[key][action].visit_count
                / tree[key][action].total_action_value)

    return tree[root.fen()]
