import chess
from typing import List, Optional

# We define some test API handles to show that the web server API works together
# with the packages we plan on using

def legal_chess_moves_from_fen(fen: str) -> Optional[List[chess.Move]]:
    try:
        board = chess.Board(fen)
        return [m.uci() for m in board.legal_moves]
    except ValueError:
        return None
