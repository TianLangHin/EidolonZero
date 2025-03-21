import chess
import torch

from utils import flip_square

def position_to_tensor(board: chess.Board) -> torch.Tensor:
    position = torch.zeros(torch.Size([20, 8, 8]))
    # Should probably extract `.fogged_board` here.
    # A lot depends on the `.turn` too.
    return position

if __name__ == '__main__':
    print(flip_square(63, chess.BLACK))
