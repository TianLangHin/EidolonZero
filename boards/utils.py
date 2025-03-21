import chess

# Turns the corresponding square index for White to Black and vice versa.
def flip_square(square: int, turn: bool) -> int:
    if turn == chess.WHITE:
        return square
    square_rank, square_file = divmod(square, 8)
    return 8 * (7 - square_rank) + square_file

