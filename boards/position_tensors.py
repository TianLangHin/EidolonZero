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
        player = board.color_at(square) != turn
        stack_index = 6 * player + PIECE_STACK_INDEX.index(piece)
        square_rank, square_file = divmod(flip_square(square, turn), 8)
        position[stack_index, square_rank, square_file] = 1
        position[12, square_rank, square_file] = 1
    for move in board.legal_moves:
        destination_square = flip_square(move.to_square, turn)
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

def tensor_to_position(pos_tensor: torch.Tensor) -> chess.Board:
    board = chess.Board.empty()
    board.turn = chess.WHITE if (pos_tensor[13] == 0).all().item() else chess.BLACK
    player_order = ((chess.WHITE, chess.BLACK)
        if board.turn == chess.WHITE else (chess.BLACK, chess.WHITE))
    for player_index, player_turn in enumerate(player_order):
        for piece_index, piece_type in enumerate(PIECE_STACK_INDEX):
            for square in range(64):
                square_rank, square_file = divmod(square, 8)
                if pos_tensor[player_index * 6 + piece_index][square_rank][square_file]:
                    board.set_piece_at(
                        flip_square(square, board.turn),
                        chess.Piece(piece_type, player_turn))
    index_white, index_black = (14, 16) if board.turn == chess.WHITE else (16, 14)
    castling_rights = (
        'K' if (pos_tensor[index_white] == 1).all().item() else '',
        'Q' if (pos_tensor[index_white + 1] == 1).all().item() else '',
        'k' if (pos_tensor[index_black] == 1).all().item() else '',
        'q' if (pos_tensor[index_black + 1] == 1).all().item() else '',
    )
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
