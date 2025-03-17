from chess import Board as FullState
from collections import namedtuple
from itertools import product
import chess

# Attribute names for each kind of piece and colour
PIECE_NAMES = [
    'white_pawns',
    'black_pawns',
    'white_knights',
    'black_knights',
    'white_bishops',
    'black_bishops',
    'white_rooks',
    'black_rooks',
    'white_queens',
    'black_queens',
    'white_kings',
    'black_kings',
]

class MaterialCounter(namedtuple('MaterialCounter', PIECE_NAMES)):
    '''
    The `MaterialCounter` type keeps track of how much material is on the board.
    This allows determining how much material is hidden under the fog.

    The attributes are initialised based on the starting position of chess.
    '''

    @classmethod
    def default(cls) -> 'MaterialCounter':
        '''
        The MaterialCounter corresponding to the starting position
        is available as a custom initialiser with no arguments.
        '''
        return cls.material_in_board(chess.Board())

    @classmethod
    def material_in_board(cls, board: FullState) -> 'MaterialCounter':
        '''
        The number of pieces belonging to each type of chess piece and colour
        is taken directly from the `SquareSet`s in the board and counted,
        passed to the default initialiser via a dictionary of arguments.

        Takes in a `chess.Board` (a.k.a. `FullState`)
        since a `PartialState` will always contain one such instance internally.
        '''
        return cls(**{
            attr: len(board.pieces(pt, c))
            for (pt, c), attr in zip(product(chess.PIECE_TYPES, chess.COLORS), PIECE_NAMES)
        })

    # We also define element-wise addition and subtraction on this type
    # for convenient determination of material counts hidden by the fog-of-war.
    def __add__(self, other: 'MaterialCounter') -> 'MaterialCounter':
        return MaterialCounter(*map(lambda x, y: x+y, self, other))

    def __sub__(self, other: 'MaterialCounter') -> 'MaterialCounter':
        return MaterialCounter(*map(lambda x, y: x-y, self, other))

# Driver code for small amounts of testing.
if __name__ == '__main__':
    startpos_counter = MaterialCounter.default()
    example_fen = 'rn2k2r/pppb1ppp/4pn2/8/1b2P3/2N5/PP3PPP/R1BQK1NR w KQkq - 0 9'
    example_counter = MaterialCounter.material_in_board(chess.Board(example_fen))
    print(startpos_counter)
    print(example_counter)
    print(startpos_counter - example_counter)
