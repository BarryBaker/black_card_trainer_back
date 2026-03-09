from __future__ import annotations

import numpy as np
from typing import Iterable, List, Tuple

RANK_TO_INT = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

SUIT_TO_INT = {"c": 0, "d": 1, "h": 2, "s": 3}


def _parse_card(card: str) -> Tuple[int, int]:
    rank_char, suit_char = card[0], card[1]
    return RANK_TO_INT[rank_char], SUIT_TO_INT[suit_char]


def encode_hands(hand_keys: dict[str, dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode strategy dicts of PLO hands into two numpy arrays:
      - ranks: shape (N, 4), dtype int16, values in [2..14]
      - suits: shape (N, 4), dtype int8, values in [0..3]

    hand format examples: "AhAdKdKh" (two-char per card, 8 chars total) or "AhAdKhKd".
    Order is as provided; downstream ops should be permutation-invariant via sorting.
    """
    hand_list: List[str] = list(hand_keys.keys())
    n = len(hand_list)
    ranks = np.empty((n, 4), dtype=np.int16)
    suits = np.empty((n, 4), dtype=np.int8)

    for i, hand in enumerate(hand_list):
        # Expect 8 characters, card1card2card3card4
        cards = [hand[j : j + 2] for j in range(0, len(hand), 2)]
        if len(cards) != 4:
            raise ValueError(f"Invalid hand string '{hand}'")
        rs = []
        ss = []
        for c in cards:
            r, s = _parse_card(c)
            rs.append(r)
            ss.append(s)
        ranks[i] = rs
        suits[i] = ss

    # Sort ranks within each hand for permutation-invariant comparisons
    # ranks.sort(axis=1)
    return ranks, suits


# def encode_hands_unsorted(hand_keys: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Encode hands without sorting, preserving original card order so ranks and suits align by position.
#     Returns (ranks, suits) with shape (N,4).
#     """
#     hand_list: List[str] = list(hand_keys)
#     n = len(hand_list)
#     ranks = np.empty((n, 4), dtype=np.int16)
#     suits = np.empty((n, 4), dtype=np.int8)
#     for i, hand in enumerate(hand_list):
#         cards = [hand[j : j + 2] for j in range(0, len(hand), 2)]
#         if len(cards) != 4:
#             raise ValueError(f"Invalid hand string '{hand}'")
#         rs = []
#         ss = []
#         for c in cards:
#             r, s = _parse_card(c)
#             rs.append(r)
#             ss.append(s)
#         ranks[i] = rs
#         suits[i] = ss
#     return ranks, suits


def encode_board(board: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a board string like '5h4h4d' or 'AsKdTd' (2-5 cards allowed) into
    rank and suit arrays of shape (B,) where B is number of board cards.
    """
    cards = [board[j : j + 2] for j in range(0, len(board), 2)]
    br = np.empty(len(cards), dtype=np.int16)
    bs = np.empty(len(cards), dtype=np.int8)
    for i, c in enumerate(cards):
        r, s = _parse_card(c)
        br[i] = r
        bs[i] = s
    return br, bs
