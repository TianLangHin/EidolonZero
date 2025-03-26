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

# The output dimension of this `torch.Tensor` will be 8 x 8 x 18,
# as a simplified and adapated version of the representation used by the AlphaZero team.
def position_to_tensor(board: chess.Board) -> torch.Tensor:
    '''
    The dimension of the tensor outputted by the function is `torch.Size([18, 8, 8])`.

    The representation of a position is inspired from AlphaZero.
    Marking a `1` in a particular cell [i][j] in the 8x8 of a particular stack
    that represents piece positionings means a piece is present at that square.
    Other planes are either completely `0` or `1`.

    The orientation of this tensor is always adjusted to make the movement of pawns
    upwards in the board, in line with the perspective of the player.

    In the first 6 stacks (index 0-5), each plane represents the presence of a piece
    belonging to the current player to move (P1) of that respective type at that cell.
    The order of representation is (pawn, knight, bishop, rook, queen, king).

    In the next 6 stacks (index 6-11), each plane represents the presence of a piece
    belonging to the current player to move (P2) of that respective type at that cell.
    The order of representation is (pawn, knight, bishop, rook, queen, king).

    The next plane (index 12) encodes a 1 value at a square that is visible
    to the player to move. This is also adjusted to account for the flipped
    encoding for Black.

    The next plane represents whether the current player is White or Black.
    White is encoded as a full plane of 0, and Black is encoded as a full plane of 1.

    The next four planes represent castling rights, which can be thought of as
    two groups of two planes.
      * Within each group, the first plane indicates kingside castling rights,
        and the second plane indicates queenside castling rights.
      * The first group indicates castling rights for the player to move (P1),
        and the second group indicates castling rights for P2.
    '''
    turn = board.turn
    position = torch.zeros(torch.Size([18, 8, 8]))
    for square in range(64):
        # For each square in the position, we check what the type of piece is.
        piece = board.piece_type_at(square)
        if piece is None:
            continue
        # We also check which player it belongs to.
        player = board.color_at(square) != turn
        # The correct stack is thus found, and we update the correct cell.
        stack_index = 6 * player + PIECE_STACK_INDEX.index(piece)
        square_rank, square_file = divmod(flip_square(square, turn), 8)
        position[stack_index, square_rank, square_file] = 1
        # We also add this to the vision field of P1 if this piece belongs to us.
        if board.color_at(square) == turn:
            position[12, square_rank, square_file] = 1
    # We also mark any legal destination as indicated by a move as visible.
    for move in board.legal_moves:
        destination_square = flip_square(move.to_square, turn)
        square_rank, square_file = divmod(destination_square, 8)
        position[12, square_rank, square_file] = 1
    # Indicates whether the current player is White or Black.
    position[13] = 0 if turn == chess.WHITE else 1
    # Castling rights with respect to P1.
    player_1, player_2 = ((chess.WHITE, chess.BLACK)
        if turn == chess.WHITE else (chess.BLACK, chess.WHITE))
    position[14] = int(board.has_kingside_castling_rights(player_1))
    position[15] = int(board.has_queenside_castling_rights(player_1))
    position[16] = int(board.has_kingside_castling_rights(player_2))
    position[17] = int(board.has_queenside_castling_rights(player_2))
    return position

def tensor_to_position(pos_tensor: torch.Tensor) -> chess.Board:
    # Initially, an empty chessboard is made.
    board = chess.Board.empty()
    # Firstly, we update whether it is White or Black to play.
    board.turn = chess.WHITE if (pos_tensor[13] == 0).all().item() else chess.BLACK
    # This is to change the interpretation of the 2 groups of 6 stacks
    # depending on the board perspective.
    player_order = ((chess.WHITE, chess.BLACK)
        if board.turn == chess.WHITE else (chess.BLACK, chess.WHITE))
    for player_index, player_turn in enumerate(player_order):
        for piece_index, piece_type in enumerate(PIECE_STACK_INDEX):
            for square in range(64):
                square_rank, square_file = divmod(square, 8)
                if pos_tensor[player_index * 6 + piece_index][square_rank][square_file]:
                    # If a certain piece does exist at a given cell,
                    # we construct that piece using piece type and player turn (P1/P2),
                    # then set it at the square that may be flipped depending on perspective.
                    board.set_piece_at(
                        flip_square(square, board.turn),
                        chess.Piece(piece_type, player_turn))
    # We then extract the castling rights also taking into account
    # the perspective change.
    index_white, index_black = (14, 16) if board.turn == chess.WHITE else (16, 14)
    castling_rights = (
        'K' if (pos_tensor[index_white] == 1).all().item() else '',
        'Q' if (pos_tensor[index_white + 1] == 1).all().item() else '',
        'k' if (pos_tensor[index_black] == 1).all().item() else '',
        'q' if (pos_tensor[index_black + 1] == 1).all().item() else '',
    )
    # Castling rights must be set via the `set_castling_fen` method.
    board.set_castling_fen(''.join(castling_rights))
    return board

if __name__ == '__main__':
    from chessboard import FoggedBoard

    def back_and_forth_test(fen: str):
        board = chess.Board(fen)
        board_tensor = position_to_tensor(board)
        reconstructed_board = tensor_to_position(board_tensor)

        fogged_moves = set(FoggedBoard.generate_fow_chess_moves(board))
        legal_moves = set(board.generate_legal_moves())

        reconstructed_fogged_moves = set(
            FoggedBoard.generate_fow_chess_moves(reconstructed_board))
        reconstructed_legal_moves = set(reconstructed_board.generate_legal_moves())

        print('Test FEN', fen, ':',
            fogged_moves == reconstructed_fogged_moves,
            legal_moves == reconstructed_legal_moves)

    fens = [
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1',
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
