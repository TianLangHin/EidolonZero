import chess
import torch

from boards.utils import flip_square

PIECE_STACK_INDEX = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

def position_to_tensor(board: chess.Board) -> torch.Tensor:
    turn = board.turn
    position = torch.zeros(torch.Size([18, 8, 8]))
    for square in range(64):
        piece = board.piece_type_at(square)
        if piece is None:
            continue
        player = board.color_at(square) == turn
        stack_index = 6 * player + PIECE_STACK_INDEX.index(piece)
        square_rank, square_file = divmod(flip_square(square, turn), 8)
        position[stack_index, square_rank, square_file] = 1
        position[12, square_rank, square_file] = 1
    for move in board.legal_moves:
        destination_square = flip_square(move.to_square)
        square_rank, square_file = divmod(destination_square, 8)
        position[12, square_rank, square_file] = 1
    position[13] = 0 if turn == chess.WHITE else 1
    player_1, player_2 = ((chess.WHITE, chess.BLACK)
        if turn == chess.WHITE else (chess.BLACK, chess.WHITE))
    position[14] = int(board.has_kingside_castling_rights(player_1))
    position[15] = int(board.has_queenside_castling_rights(player_1))
    position[16] = int(board.has_kingside_castling_rights(player_2))
    position[17] = int(board.has_queenside_castling_rights(player_2))
    return position

if __name__ == '__main__':
    print(flip_square(63, chess.BLACK))
