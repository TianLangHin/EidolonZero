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

PuctConfig = namedtuple('PuctConfig', ['c_puct', 'dirichlet_alpha', 'epsilon'])

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
    num_simulations: int,
    predictor: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    config: PuctConfig) -> dict[Action, Stats]:

    c_puct, dirichlet_alpha, epsilon = config

    tree: dict[str, dict[Action, Stats]] = {}
    root = startpos

    for _ in range(num_simulations):
        board = root
        path = []
        found_terminal = False

        moves_ahead = 0

        while (actions := tree.get(board.fen(), None)) is not None and moves_ahead < 256:
            if len(actions) == 0:
                found_terminal = True
                break
            numerator = sum(stats.visit_count for stats in actions.values())
            best_action = max(actions.items(),
                key=lambda v: uct_selector(v[1], c_puct, numerator))[0]
            path.append((board, best_action))
            (board := board.copy()).push(best_action)
            moves_ahead += 1

        if found_terminal or moves_ahead == 256:
            match (startpos.turn, game_outcome(board)):
                case (chess.WHITE, GameOutcome.WHITE) | (chess.BLACK, GameOutcome.BLACK):
                    value = 1
                case (chess.WHITE, GameOutcome.BLACK) | (chess.BLACK, GameOutcome.WHITE):
                    value = -1
                case _:
                    value = 0
        else:
            value, policy = predictor(position_to_tensor(board))
            value = value.item()
            policy = policy.reshape(torch.Size([73, 8, 8]))
            legal_move_tensor = move_gen_to_tensor(
                FoggedBoard.generate_fow_chess_moves(board), board.turn)
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
                    tree[key].keys(), np.random.dirichlet([0.3] * dimension))}
                epsilon = 0.25
                p = tree[key][action].prior_probability
                p = (1 - epsilon) * p + epsilon * noise[action]
                tree[key][action].prior_probability = p

            tree[key][action].visit_count += 1
            tree[key][action].total_action_value += v
            tree[key][action].mean_action_value = (tree[key][action].visit_count
                / tree[key][action].total_action_value)

    return tree[root.fen()]
