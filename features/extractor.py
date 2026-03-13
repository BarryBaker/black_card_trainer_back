from typing import Dict, Iterable, List, Optional, Sequence, Mapping, Any, Tuple
import sys
from pathlib import Path

# Add parent directory to path so we can import db.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from db import PokiDatabase
from features.board import Board
from features.encoding import encode_hands
from features.feature_ops import (
    compute_counts_by_rank,
    feature_overpair,
    feature_top_pair,
    feature_quads,
    feature_full_house,
    feature_trips,
    feature_two_pair,
    feature_board_pair,
    feature_flush_strength,
    feature_flush_draw_strength,
    feature_flush_blocker,
    feature_double_flush_draw,
    feature_rr,
    feature_drr,
    feature_straight_draw,
    straight_helper,
    feature_backdoor_flushdraw,
)

import time


def extract_features(
    board,
    data,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract poker hand features from board and action data.

    Args:
        board_and_data: Dictionary with 'board' (str) and 'data' (dict) keys,
                       typically from db.get_poki_nth_board_and_data(n)

    Returns:
        Tuple of (feature_table, actions_df) dataframes
    """
    feature_names: Optional[List[str]] = None

    data_items: List[Tuple[Any, Mapping[str, Any]]] = list(data.items())
    row_keys: List[Any] = [k for k, _ in data_items]
    action_dicts: Iterable[Mapping[str, Any]] = [v for _, v in data_items]

    # Encode
    board = Board(board)
    ranks, suits = encode_hands(data)
    board_ranks = board.ranks
    board_suits = board.suits

    time1 = time.time()

    counts_by_rank = compute_counts_by_rank(ranks)

    feats: Dict[str, np.ndarray] = {}

    feats["Quads"] = feature_quads(counts_by_rank, board)
    fh_mask, fh_best = feature_full_house(counts_by_rank, board)
    feats["Full"] = fh_mask
    feats["Nut Full"] = fh_best
    trips_mask, trips1_mask, trips_top_mask = feature_trips(counts_by_rank, board)
    feats["Trips"] = trips_mask
    feats["Nut Trips"] = trips1_mask
    feats["Top Trips"] = trips_top_mask
    two_pair_mask, top_two_pair_mask = feature_two_pair(counts_by_rank, board)
    feats["Two Pair"] = two_pair_mask
    feats["Top Two Pair"] = top_two_pair_mask
    board_rank_counts = np.bincount(board_ranks, minlength=15)
    feats["Overpair"] = feature_overpair(counts_by_rank, board_rank_counts)
    board_pair_mask, mid_pair_mask, low_pair_mask = feature_board_pair(
        counts_by_rank, board
    )
    feats["Board Pair"] = board_pair_mask
    feats["Mid Pair"] = mid_pair_mask
    feats["Low Pair"] = low_pair_mask

    # Flush features
    flush_mask_ord = feature_flush_strength(ranks, suits, board)
    feats["Flush"] = (flush_mask_ord > 0).astype(np.int8)
    feats["Nut Flush"] = (flush_mask_ord == 1).astype(np.int8)
    feats["2nd Nut Flush"] = (flush_mask_ord == 2).astype(np.int8)
    # Top pair features
    top_pair_mask, highest_not_on_board = feature_top_pair(counts_by_rank, board)
    feats["Top Pair"] = top_pair_mask
    # Top kicker for top pair: must both have top pair and also hold the highest rank not on board
    board_present = board.rank_counts[2:15] > 0
    all_ranks = np.arange(2, 15, dtype=np.int8)
    not_on_board_mask = ~board_present
    if not_on_board_mask.any():
        highest_not_on_board_value = int(all_ranks[not_on_board_mask].max())
    else:
        highest_not_on_board_value = 0
    has_best_kicker = (highest_not_on_board == highest_not_on_board_value).astype(
        np.int8
    )
    feats["Top Pair TK"] = (top_pair_mask & has_best_kicker).astype(np.int8)

    # Flush draws
    fdraw_mask_ord = feature_flush_draw_strength(ranks, suits, board)
    feats["Flushdraw"] = (fdraw_mask_ord > 0).astype(np.int8)
    feats["Nut Flushdraw"] = (fdraw_mask_ord == 1).astype(np.int8)
    feats["Nut Flushdraw"] = (fdraw_mask_ord == 2).astype(np.int8)
    bdfd, double_bdfd = feature_backdoor_flushdraw(suits, board)
    feats["Backdoor FD"] = bdfd
    feats["D Backdoor FD"] = double_bdfd
    # Flush blockers
    blk, blk1, blkh = feature_flush_blocker(ranks, suits, board)
    feats["Flushblocker"] = blk
    feats["Nut Flushblocker"] = blk1
    feats["High Flushblocker"] = blkh
    feats["RR"] = feature_rr(counts_by_rank, board)
    feats["D RR"] = feature_drr(counts_by_rank, board)

    # Double flush draw (turn only)
    feats["D Flushdraw"] = feature_double_flush_draw(suits, board)

    # Precompute straight helper once; reuse for straight, nut_straight, and draws
    (made_high, hand_sum_used) = straight_helper(counts_by_rank, board_ranks)
    # feats["straight_made_high"] = made_high
    # feats["straight_hand_sum_used"] = hand_sum_used
    feats["Straight"] = (made_high >= 1).astype(np.int8)
    feats["Nut Straight"] = (made_high == 1).astype(np.int8)

    outs_count, draw_made_high = feature_straight_draw(
        counts_by_rank, board_ranks, made_high, hand_sum_used
    )
    # Straight draw feature definitions
    feats["Gutshot"] = (outs_count == 1).astype(np.int8)
    feats["Nut Gutshot"] = ((outs_count == 1) & (draw_made_high == 1)).astype(np.int8)
    feats["Oesd"] = (outs_count == 2).astype(np.int8)
    feats["Nut Oesd"] = ((outs_count == 2) & (draw_made_high == 1)).astype(np.int8)
    feats["Wrap"] = (outs_count > 2).astype(np.int8)
    feats["Nut Wrap"] = ((outs_count > 2) & (draw_made_high == 1)).astype(np.int8)

    # Rank-based features
    rank_chars = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    cols_single: Dict[str, np.ndarray] = {}
    cols_double: Dict[str, np.ndarray] = {}
    for offset, rc in enumerate(rank_chars, start=2):
        cols_single[rc] = (counts_by_rank[:, offset] == 1).astype(np.int8)
        cols_double[rc * 2] = (counts_by_rank[:, offset] >= 2).astype(np.int8)

    df_single = pd.DataFrame(cols_single, index=list(row_keys))
    df_double = pd.DataFrame(cols_double, index=list(row_keys))
    rank_df = pd.concat([df_single, df_double], axis=1)

    col_order = feature_names if feature_names else list(feats.keys())
    time2 = time.time()
    X = np.column_stack([feats[name] for name in col_order])

    features_df = pd.DataFrame(X, columns=col_order, index=row_keys)
    if action_dicts is not None:
        action_list = list(action_dicts)
        if len(action_list) != features_df.shape[0]:
            raise ValueError("Length of action_dicts must match number of hands")
        action_keys = list(action_list[0].keys()) if action_list else []

        if action_keys:
            actions_matrix = np.array(
                [[float(d.get(k, 0)) for k in action_keys] for d in action_list],
                dtype=np.float32,
            )
            actions_df = pd.DataFrame(
                actions_matrix,
                columns=[str(k) for k in action_keys],
                index=features_df.index,
            )
        else:
            actions_df = pd.DataFrame(index=features_df.index)
    else:
        actions_df = pd.DataFrame(index=features_df.index)

    # Combine all dataframes
    feature_table = pd.concat([features_df, rank_df], axis=1)

    print(f"Feature computation took {time2 - time1:.6f} seconds")

    return feature_table, actions_df


if __name__ == "__main__":
    # Initialize the database class
    db = PokiDatabase()

    # Get and extract features for the nth row
    n = 1
    board_and_data = db.get_poki_nth_board_and_data(n)
    feature_table, actions_df = extract_features(board_and_data)

    print(feature_table)
    print(actions_df)
