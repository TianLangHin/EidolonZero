import chess
import torch

def position_to_tensor(board: chess.Board) -> torch.Tensor:
    position = torch.zeros(torch.Size([20, 8, 8]))
    return position
