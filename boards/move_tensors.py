import chess
import torch
from enum import Enum
from typing import Generator, Optional, Tuple

from boards.utils import flip_square

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

DIRECTION_LOOKUP_MAP = [
    Direction.N,
    Direction.NE,
    Direction.E,
    Direction.SE,
    Direction.S,
    Direction.SW,
    Direction.W,
    Direction.NW,
]

DIRECTION_OFFSET_MAP = {
    Direction.N: 8,
    Direction.NE: 9,
    Direction.E: 1,
    Direction.SE: -7,
    Direction.S: -8,
    Direction.SW: -9,
    Direction.W: -1,
    Direction.NW: 7,
}

KNIGHT_REVERSE_MAP = [15, 17, 6, 10, -6, -10, -15, -17]
KNIGHT_MOVE_MAP = {offset: i for i, offset in enumerate(KNIGHT_REVERSE_MAP)}

UNDERPROMOTION_PIECE_REVERSE_MAP = [chess.ROOK, chess.BISHOP, chess.KNIGHT]
UNDERPROMOTION_PIECE_MAP = {
    piece: i for i, piece in enumerate(UNDERPROMOTION_PIECE_REVERSE_MAP)
}

UNDERPROMOTION_DIRECTION_REVERSE_MAP = [Direction.N, Direction.NE, Direction.NW]
UNDERPROMOTION_DIRECTION_MAP = {
    direction: i for i, direction in enumerate(UNDERPROMOTION_DIRECTION_REVERSE_MAP)
}

# Converts the movement from one square to another
# (each encoded as an integer from 0 to 63)
# to an enum that shows which direction that movement is in,
# as well as finding the distance travelled in this direction.
# If it is not one of the eight cardinal directions, `None` is returned.
def direction_and_magnitude(origin: int, destination: int) -> Optional[Tuple[Direction, int]]:
    origin_rank, origin_file = divmod(origin, 8)
    destination_rank, destination_file = divmod(destination, 8)
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
def move_gen_to_tensor(move_gen: Generator[chess.Move, None, None], turn: bool) -> torch.Tensor:
    '''
    The dimension of the tensor outputted by the function is `torch.Size([73, 8, 8])`.

    The representation of a move distribution will be as used in chess for AlphaZero:
    Marking a `1` in a particular cell [i][j] in the 8x8 plane of a particular stack
    represents moving a piece from that square.
    The plane within the stack this appears in represents which kind of movement is made.

    In the first 56 stacks, there are 8 groups of 7 planes:
      * Within each group, the relative index 0 to 6 represents
        moving 1 to 7 squares in that direction.
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
        origin, destination, promote = (flip_square(move.from_square, turn),
            flip_square(move.to_square, turn), move.promotion)
        square_index = divmod(origin, 8)
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

# Takes in a Tensor of size 8 x 8 x 73, and yields each of the moves
# when decoded from this tensor representation.
# The order of the move yielding is based purely on where in the stack
# its respective representation is placed.
def tensor_to_move_gen(move_dist: torch.Tensor, *, position: chess.Board) -> Generator[chess.Move, None, None]:
    turn = position.turn
    for stack_index in range(56):
        # The first lot of stacks is 8 groups of 7 stacks, with each group representing a movement direction
        # and the position within each group of 8 representing the number of squares moved.
        direction_index, magnitude = divmod(stack_index, 7)
        # The magnitude represented by the 0-th offsetted stack within a group is a movement of one square.
        magnitude += 1
        for square in range(64):
            # For each square within the 8 x 8 grid,
            # we use the little endian encoding to match with the convention of the `chess` package.
            square_rank, square_file = divmod(square, 8)
            from_square = flip_square(square, turn)
            if move_dist[stack_index][square_rank][square_file]:
                from_square = flip_square(square, turn)
                # The direction of the movement will be encoded as if the current player is White.
                # We determine the final destination based on this,
                # and only flip the origin and destination squares based on the player colour
                # at the very final conversion to a `chess.Move` instance.
                direction = DIRECTION_LOOKUP_MAP[direction_index]
                match direction:
                    # These are the only three movement directions that can cause a promotion move
                    # since these are the only three ways a pawn can move towards the eighth rank.
                    case Direction.N | Direction.NW | Direction.NE:
                        destination = square + magnitude * DIRECTION_OFFSET_MAP[direction]
                        to_square = flip_square(destination, turn)
                        # Since only underpromotions are explicitly given different stacks,
                        # pawn moves to the eighth rank are implicitly encoded here as a queen promotion.
                        # However, if the moving piece is not a pawn, then no promotion is marked at all.
                        if position.piece_type_at(from_square) == chess.PAWN and (destination >> 3) == 7:
                            yield chess.Move(from_square, to_square, chess.QUEEN)
                        else:
                            yield chess.Move(from_square, to_square)
                    case _:
                        # If it is any other direction, it cannot be a pawn move.
                        # (Technically, we have the ability to check this, but this function will
                        # assume that the move distribution encoded in the tensor is valid,
                        # to make it as fast as possible.)
                        to_square = flip_square(square + magnitude * DIRECTION_OFFSET_MAP[direction], turn)
                        yield chess.Move(from_square, to_square)
    # The next 8 stacks encode certain knight movement directions.
    for knight_index in range(8):
        movement_offset = KNIGHT_REVERSE_MAP[knight_index]
        for square in range(64):
            square_rank, square_file = divmod(square, 8)
            if move_dist[56 + knight_index][square_rank][square_file]:
                # We again do not do validation on whether the given movement offset is possible,
                # since we assume the tensor representation is valid.
                # Hence, we can just add the offset and assume
                # that it will not wrap incorrectly off a side of the board.
                destination = square + movement_offset
                yield chess.Move(flip_square(square, turn), flip_square(destination, turn))
    # Here, we assume that moves with an underpromotion piece marked is a pawn move.
    for stack_index in range(9):
        # The tensor representation here is 3 groups of 3 stacks,
        # with the stack within a group determining the direction of movement (N, NW, NE)
        # and the positioning of the group itself determines which piece is being underpromoted to.
        piece_index, direction_index = divmod(stack_index, 3)
        piece_type = UNDERPROMOTION_PIECE_REVERSE_MAP[piece_index]
        direction = UNDERPROMOTION_DIRECTION_REVERSE_MAP[direction_index]
        for square in range(64):
            square_rank, square_file = divmod(square, 8)
            if move_dist[64 + stack_index][square_rank][square_file]:
                from_square = flip_square(square, turn)
                match direction:
                    case Direction.N:
                        to_square = flip_square(square + 8)
                    case Direction.NW:
                        to_square = flip_square(square + 7)
                    case Direction.NE:
                        to_square = flip_square(square + 9)
                yield chess.Move(from_square, to_square, piece_type)

if __name__ == '__main__':
    from chessboard import FoggedBoard

    def back_and_forth_test(fen: str):
        board = chess.Board()
        original_move_set = set(FoggedBoard.generate_fow_chess_moves(board))
        move_tensor = move_gen_to_tensor(FoggedBoard.generate_fow_chess_moves(board), board.turn)
        move_set = set(tensor_to_move_gen(move_tensor, position=board))
        print('Test FEN', fen, ':', original_move_set == move_set)

    fens = [
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3',
        'r1bqk2r/ppppbppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 6 5',
        'r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4',
        'r1bqk2r/ppppbppp/2n2n2/1B2p3/4P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 5',
        'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
        '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
        'r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1',
        'rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8',
        'r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10',
        '2kn1b2/4P3/7p/ppPr3P/P3N3/8/8/R3K1NR w KQ - 0 1',
        'r3k1nr/8/8/p3n3/PPpR3p/7P/4p3/2KN1B2 b kq - 0 1',
    ]

    for fen in fens:
        back_and_forth_test(fen)
