from chess import Board as FullState, Move
from collections import namedtuple
from functools import reduce
from itertools import product
from typing import Generator, List, Optional
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
        since a `FoggedBoard` will always contain one such instance internally.
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

class FoggedBoard:
    '''
    This class represents a board state as it would be observed by the player to move.
    Internally, a `chess.Board` instance is used to represent the state,
    but this state itself will have the invisible pieces removed.
    The following pieces of information are recorded together with this perceived board state:
     * Hidden material count, since this information is technically known to the player at all times
       by counting the number of captures of opponent pieces were made.
     * Legal moves in the current position, since this information is used to derive
       the perceived board state.
     * Visible squares, which is also directly used to determine the perceived board state.
    '''
    def __init__(
            self,
            board: chess.Board,
            material: MaterialCounter,
            *,
            moves: Optional[List[Move]] = None):
        # The optional `moves` parameter is made available to
        # prevent redundant computation of legal moves.
        self.fogged_board_state = board
        self.hidden_material = material
        self.legal_moves = moves or list(FoggedBoard.generate_fow_chess_moves(board))
        self.visible_squares = FoggedBoard.get_visible_squares(board, self.legal_moves)

    @staticmethod
    def generate_fow_chess_moves(board: FullState) -> Generator[Move, None, None]:
        '''
        Generates all possible legal moves that can be made by the player to move.
        Since this is within a fog-of-war chess setting, this will ignore any checks or pins,
        and will also allow castling out of, through and into check.
        '''
        yield from board.generate_pseudo_legal_moves()

        (king, rook_a, rook_h, dest_ooo, dest_oo, backrank) = (
            (chess.E1, chess.A1, chess.H1, chess.C1, chess.G1, chess.BB_RANK_1)
            if board.turn else
            (chess.E8, chess.A8, chess.H8, chess.C8, chess.G8, chess.BB_RANK_8))

        mask_ooo = backrank & (chess.BB_FILE_B | chess.BB_FILE_C | chess.BB_FILE_D)
        mask_oo = backrank & (chess.BB_FILE_F | chess.BB_FILE_G)

        if board.has_queenside_castling_rights(board.turn) and not (board.occupied & mask_ooo):
            yield chess.Move(king, dest_ooo)
        if board.has_kingside_castling_rights(board.turn) and not (board.occupied & mask_oo):
            yield chess.Move(king, dest_oo)

    @staticmethod
    def get_visible_squares(board: FullState, moves: List[Move]) -> chess.SquareSet:
        '''
        For a given position and its legal moves,
        the set of squares visible to the current player is the union of
        the set of squares any of the pieces can move to and
        the set of squares where the current player has a piece.
        '''
        square_set = reduce(
            lambda squares, piece: squares | board.pieces(piece, board.turn),
            chess.PIECE_TYPES,
            chess.SquareSet())
        for move in moves:
            square_set |= 1 << move.from_square
            square_set |= 1 << move.to_square
        return square_set

    @classmethod
    def derived_from_full_state(cls, board: FullState, *, copy_board=True) -> 'FoggedBoard':
        '''
        The fogged chessboard state is always calculated from the perspective of
        the player who is going to make a move from this position
        (which is given by the `turn` attribute).

        We also provide an option to copy the passed board or not,
        to prevent unpredictable behaviour due to the usage of the `.remove_piece_at` method.
        '''

        total_material = MaterialCounter.material_in_board(board)

        if copy_board:
            board = board.copy()

        legal_moves = list(FoggedBoard.generate_fow_chess_moves(board))
        visible_squares = FoggedBoard.get_visible_squares(board, legal_moves)
        for square in range(chess.A1, chess.H8 + 1):
            if not (visible_squares >> square) & 1:
                board.remove_piece_at(square)

        visible_material = MaterialCounter.material_in_board(board)

        return cls(board, total_material - visible_material, moves=legal_moves)

# Driver code for small amounts of testing.
if __name__ == '__main__':
    print('Driver code for MaterialCounter')
    startpos_counter = MaterialCounter.default()
    example_fen = 'rn2k2r/pppb1ppp/4pn2/8/1b2P3/2N5/PP3PPP/R1BQK1NR w KQkq - 0 9'
    example_counter = MaterialCounter.material_in_board(chess.Board(example_fen))
    print(startpos_counter)
    print(example_counter)
    print(startpos_counter - example_counter)

    print('Driver code for FoggedBoard')
    example_fen = 'rn1qkb1r/p1p1pppp/1p1p1n2/8/3PP3/3b1N2/PPP2PPP/RNB1K2R w KQkq - 0 6'
    startpos = FoggedBoard.derived_from_full_state(chess.Board(example_fen), copy_board=False)
    print(startpos.fogged_board_state)
    print(startpos.hidden_material)
    print(startpos.legal_moves)
    print(startpos.visible_squares)
