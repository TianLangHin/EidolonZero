import chess
import torch
from enum import Enum
from typing import Generator, Optional, Tuple

Direction = Enum('Direction', [
    ('N', 0),
    ('NE', 1),
    ('E', 2),
    ('SE', 3),
    ('S', 4),
    ('SW', 5),
    ('W', 6),
    ('NW', 7),
])

KNIGHT_MOVE_MAP = {
    15: 0,
    17: 1,
    6: 2,
    10: 3,
    -6: 4,
    -10: 5,
    -15: 6,
    -17: 7,
}

UNDERPROMOTION_PIECE_MAP = {
    chess.ROOK: 0,
    chess.BISHOP: 1,
    chess.KNIGHT: 2,
}

UNDERPROMOTION_DIRECTION_MAP = {
    Direction.NW: 2,
    Direction.N: 0,
    Direction.NE: 1,
}

# Converts the movement from one square to another
# (each encoded as an integer from 0 to 63)
# to an enum that shows which direction that movement is in,
# as well as finding the distance travelled in this direction.
# If it is not one of the eight cardinal directions, `None` is returned.
def direction_and_magnitude(origin: int, destination: int) -> Optional[Tuple[Direction, int]]:
    origin_rank, origin_file = origin >> 3, origin & 7
    destination_rank, destination_file = destination >> 3, destination & 7
    if origin_file == destination_file:
        # If you stay on the same file, the movement is vertical.
        distance = abs(destination_rank - origin_rank)
        if destination_rank > origin_rank:
            return Direction.N, distance
        elif destination_rank < origin_rank:
            return Direction.S, distance
        else:
            return None
    elif origin_rank == destination_rank:
        # If you stay on the same rank, the movement is horizontal.
        distance = abs(destination_file - origin_file)
        if destination_file > origin_file:
            return Direction.E, distance
        elif destination_file < origin_file:
            return Direction.W, distance
        else:
            return None
    elif origin_rank - origin_file == destination_rank - destination_file:
        # Tests whether these squares lie on the same major diagonal
        # (south-west to north-east)
        distance = abs(destination_rank - origin_rank)
        if destination_rank > origin_rank:
            return Direction.NE, distance
        elif destination_rank < origin_rank:
            return Direction.SW, distance
        else:
            return None
    elif origin_rank + origin_file == destination_rank + destination_file:
        # Tests whether these squares lie on the same minor diagonal
        # (north-west to south-east)
        distance = abs(destination_rank - origin_rank)
        if destination_rank > origin_rank:
            return Direction.NW, distance
        elif destination_rank < origin_rank:
            return Direction.SE, distance
        else:
            return None
    else:
        return None

# The output dimension of this `torch.Tensor` will be 8 x 8 x 73,
# as per the move distribution representation used by the AlphaZero team.
# This function assumes all moves given to it are valid.
def move_gen_to_tensor(move_gen: Generator[chess.Move, None, None]) -> torch.Tensor:
    '''
    The representation of a move distribution will be as used in chess for AlphaZero:
    Marking a `1` in a particular cell [i][j] in the 8x8 plane of a particular stack
    represents moving a piece from that square.
    The plane within the stack this appears in represents which kind of movement is made.

    In the first 56 stacks, there are 8 groups of 7 planes:
      * Within each group, the relative index 0 to 6 represents moving 1 to 7 squares.
      * Each group represents a particular direction of movement,
        mapping their relative index of 0 to 7 to {N, NE, E, SE, S, SW, W, NW}.

    In the next 8 planes, each knight move direction is encoded.
    The chosen order of encoding is:
        (x-1,y+2), (x+1,y+2), (x-2,y+1), (x+2,y+1),
        (x-2,y-1), (x+2,y-1), (x-1,y-2), (x+1,y-2).

    In the final 9 planes, each possible movement with underpromotion is encoded.
    There are 3 groups of 3 planes:
      * Within each group, the relative index 0 to 2 represents moving in {N, NE, NW}.
      * Each group represents a particular underpromotion piece: {Rook, Bishop, Knight}.
    '''
    move_dist = torch.zeros(torch.Size([73, 8, 8]))
    for move in move_gen:
        # Relies on the little endian encoding of
        # `Chess.A1 = 0`, `Chess.B1 = 1`, etc. until `Chess.H8 = 63`.
        origin, destination, promote = move.from_square, move.to_square, move.promotion
        square_index = (origin >> 3, origin & 7)
        match promote:
            case None | chess.QUEEN:
                match direction_and_magnitude(origin, destination):
                    case None:
                        # Knight moves
                        # We assume there are no null moves being made.
                        square_difference = destination - origin
                        plane_index = 56 + KNIGHT_MOVE_MAP[square_difference]
                        move_dist[(plane_index, *square_index)] = 1
                    case direction, magnitude:
                        plane_index = direction.value * 7 + magnitude - 1
                        move_dist[(plane_index, *square_index)] = 1
            case chess.ROOK | chess.BISHOP | chess.KNIGHT:
                piece_index = UNDERPROMOTION_PIECE_MAP[promote]
                direction, _ = direction_and_magnitude(origin, destination)
                # The line below will trigger a `ValueError`
                # if it is not a valid pawn promotion move.
                direction_index = UNDERPROMOTION_DIRECTION_MAP[direction]
                plane_index = 64 + 3 * piece_index + direction_index
                move_dist[(plane_index, *square_index)] = 1

    return move_dist







