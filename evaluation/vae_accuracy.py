from boards import FoggedBoard

import chess

# Correct piece type predictions
def piece_type_accuracy(
        pred: chess.Board, true: chess.Board, fogged: FoggedBoard) -> float:

    hidden_piece_count = sum(fogged.hidden_material)

    known_pieces_map = {
        sq for sq in range(64)
        if fogged.fogged_board_state.piece_at(sq) is not None}

    pred_pieces_map = {
        sq: pred.piece_at(sq)
        for sq in range(64)
        if sq not in known_pieces_map and pred.piece_at(sq) is not None}

    correct_piece_type_predictions = sum(
        true.piece_at(sq) == pt for sq, pt in pred_pieces_map.items())

    return correct_piece_type_predictions / hidden_piece_count

# Intersection over union (piece locations)
def piece_location_iou_accuracy(
        pred: chess.Board, true: chess.Board, fogged: FoggedBoard) -> float:

    known_pieces_map = {
        sq for sq in range(64)
        if fogged.fogged_board_state.piece_at(sq) is not None}

    pred_set = {
        sq for sq in range(64)
        if sq not in known_pieces_map and pred.piece_at(sq) is not None}

    true_set = {
        sq for sq in range(64)
        if sq not in known_pieces_map and true.piece_at(sq) is not None}

    return len(pred_set & true_set) / len(pred_set | true_set)
