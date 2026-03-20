from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

"""
python -m pytest -v

"""

# Allow running this file directly: `python tests/test_feature_ops.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from features.board import Board
from features.encoding import encode_hands
from features.feature_ops import (
    feature_backdoor_flushdraw,
    compute_counts_by_rank,
    feature_overpair,
    feature_top_pair,
    feature_board_pair,
    feature_double_flush_draw,
    feature_flush_blocker,
    feature_flush_draw_strength,
    feature_flush_strength,
    feature_full_house,
    feature_quads,
    feature_rr,
    feature_rr_high_low,
    feature_straight_draw,
    feature_trips,
    feature_two_pair,
    straight_helper,
)


def test_top_pair_unpaired_board():
    # Board: AsKd7h (unpaired, highest=A)
    board = Board("AsKd7h")
    hands = {
        "AhAdKhKd": {},  # has Ace → top pair
        "QsQdJhJd": {},  # no Ace → not top pair
        "Ac2d3h4s": {},  # has Ace → top pair
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    top_pair, kicker = feature_top_pair(counts, board)
    assert top_pair.tolist() == [1, 0, 1]
    # Highest not on board per hand: exclude A,K,7
    # AhAdKhKd -> none (0); QsQdJhJd -> Q (12); Ac2d3h4s -> 4
    assert kicker.tolist() == [0, 12, 4]


def test_top_pair_paired_board_singleton_high():
    # Board: AhAdKc (paired Aces, singleton highest=K)
    board = Board("AhAdKc")
    hands = {
        "KhKdQsQd": {},  # has K → top pair
        "QhQdJhJd": {},  # no K → not top pair
        "Ks2d3h4s": {},  # has K → top pair
        "As2d3h4s": {},  # no K → nottop pair
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    top_pair, kicker = feature_top_pair(counts, board)
    assert top_pair.tolist() == [1, 0, 1, 0]
    # Not-on-board exclude A and K; KhKdQsQd -> Q (12); QhQdJhJd -> Q (12); Ks2d3h4s -> 4
    assert kicker.tolist() == [12, 12, 4, 4]


def test_top_pair_double_paired_board_singleton_high():
    # Board: AhAdKcKd (double paired Aces and Kings, no singletons -> top pair impossible)
    board = Board("AhAdKcKd")
    hands = {
        "KhKdQsQd": {},
        "QhQdJhJd": {},
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    top_pair, kicker = feature_top_pair(counts, board)
    # No singleton ranks on board → top pair mask all zeros
    assert top_pair.tolist() == [0, 0]
    # Not-on-board highest excludes A and K; first hand Q (12), second Q (12)
    assert kicker.tolist() == [12, 12]


def test_top_pair_trips_board_singleton_high():
    # Board: AhAdAsKc (trips Aces, singleton highest=K)
    board = Board("AhAdAsKc")
    hands = {
        "KhKdQsQd": {},  # has K → top pair
        "QhQdJhJd": {},  # no K → not top pair
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    top_pair, kicker = feature_top_pair(counts, board)
    assert top_pair.tolist() == [1, 0]
    # Not-on-board highest excludes A and K; highest is Q (12)
    assert kicker.tolist() == [12, 12]


def test_overpair_vs_board_highest():
    # Overpair: pocket pair higher than highest rank on board
    # Board: Qh9d4s (highest=Q)
    board = Board("Qh9d4s")
    hands = {
        "KhKc2d3s": {},  # KK overpair (13 > 12)
        "QhQcAhKs": {},  # QQ equals highest → not overpair
        "JhJcAsKs": {},  # JJ (11 < 12) → not overpair
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    overpair = feature_overpair(counts, board.rank_counts)
    assert overpair.tolist() == [1, 0, 0]


def test_quads_unpaired_board():
    # Unpaired board: no quads possible
    board = Board("AsKd7h")
    hands = {
        "AhAdKhKd": {},  # pocket pairs but unpaired board → no quads
        "QsQdJhJd": {},
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    quads = feature_quads(counts, board)
    assert quads.tolist() == [0, 0]


def test_quads_single_paired_board():
    # Board paired: AhAdKc → need pocket pair of A to make quads
    board = Board("AhAdKc")
    hands = {
        "AhAcQsQd": {},  # AA + board AA → quads
        "KhKcQsQd": {},  # KK only → no quads
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    quads = feature_quads(counts, board)
    assert quads.tolist() == [1, 0]


def test_quads_double_paired_board():
    # Board double-paired: AhAdKcKd → need AA or KK in hand
    board = Board("AhAdKcKd")
    hands = {
        "AhAcQsQd": {},  # AA → quads of A
        "KhKcQsQd": {},  # KK → quads of K
        "QhQcJhJd": {},  # QQ → no quads
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    quads = feature_quads(counts, board)
    assert quads.tolist() == [1, 1, 0]


def test_quads_trips_on_board():
    # Trips on board: AhAdAsKc → need at least one A in hand
    board = Board("AhAdAsKc")
    hands = {
        "AhAcQsQd": {},  # has A → quads
        "KhKcQsQd": {},  # no A → no quads
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    quads = feature_quads(counts, board)
    assert quads.tolist() == [1, 0]


def test_quads_quads_on_board_blocked():
    # Quads on board: AhAdAsAc → no quads possible from hand
    board = Board("AhAdAsAc")
    hands = {
        "AhAcQsQd": {},
        "KhKcQsQd": {},
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    quads = feature_quads(counts, board)
    assert quads.tolist() == [0, 0]


def test_full_house_unpaired_board():
    board = Board("AsKd7h")
    hands = {
        "AhAdKhKd": {},
        "QsQdJhJd": {},
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    fh, fh1 = feature_full_house(counts, board)
    assert fh.tolist() == [0, 0]
    assert fh1.tolist() == [0, 0]


def test_full_house_single_paired_board():
    # Board: AhAdKc (pair of A, singleton K)
    board = Board("AhAdKc")
    hands = {
        "KhKcQsQd": {},  # pair of singleton rank K → full house
        "AhKcQsQd": {},  # one A (pair rank) + one singleton K → full house
        "QhQcJhJd": {},  # pair of non-board singleton rank → not full house
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    fh, fh1 = feature_full_house(counts, board)
    assert fh.tolist() == [1, 1, 0]
    # Best when no higher singleton: need one from pair (A) and one from highest singleton (K)
    assert fh1.tolist() == [0, 1, 0]


def test_full_house_double_paired_board():
    # Board: AhAdKcKd (double paired). No singletons on board
    board = Board("AhAdKcKd")
    hands = {
        "AhKcQsQd": {},  # one from each pair → full house
        "AhQcQsQd": {},  # one from pair A + one singleton Q → full house not applicable (no singleton on board), should be 0
        "QhQcJhJd": {},  # pair of singleton ranks (none on board) → not full house
        "AhAdQsQd": {},  # two from A (pocket pair) + board double paired still need one from K? Rule says one from each is enough; pocket pair only is not enough → 0
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    fh, fh1 = feature_full_house(counts, board)
    assert fh.tolist() == [1, 0, 0, 0]
    assert fh1.tolist() == [0, 0, 0, 0]


def test_full_house_trips_board_higher_singleton_pair_required():
    # Board: AhAdAsKc (trips A, singleton K). Need pocket pair of singleton higher than trips (K> A? No). None qualifies.
    board = Board("AhAdAsKc")
    hands = {
        "KhKcQsQd": {},  # KK pair; K (13) not higher than trips A (14) → not full house per rule
        "QhQcJhJd": {},  # QQ pair; Q (12) < A (14) → not
        "AhKcQsQd": {},  # has A,K but not pair of K/Q → not
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    fh, fh1 = feature_full_house(counts, board)
    assert fh.tolist() == [0, 0, 0]
    assert fh1.tolist() == [0, 0, 0]

    board = Board("8h8d8sKc")
    hands = {
        "KhKcQsQd": {},  # full house per rule
        "QhQcJhJd": {},  # QQ pair; Q (12) < A (14) → not
        "AhKcQsQd": {},  # has A,K but not pair of K/Q → not
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    fh, fh1 = feature_full_house(counts, board)
    assert fh.tolist() == [1, 0, 0]
    # Best requires pair of highest singleton (7)
    assert fh1.tolist() == [1, 0, 0]

    board = Board("8h8d8s2c")
    hands = {
        "KhKcQsQd": {},  # not full house per rule
        "QhQcJhJd": {},  # QQ pair; Q (12) < A (14) → not
        "AhKc2s2d": {},  # has A,K but not pair of K/Q → not
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    fh, fh1 = feature_full_house(counts, board)
    assert fh.tolist() == [0, 0, 0]
    assert fh1.tolist() == [0, 0, 0]


def test_full_house_trips_board_higher_singleton_pair_exists():
    # Board: 6h6d6s7c (trips 6, singleton 7). Need pocket pair of any singleton higher than 6; 7s7d qualifies.
    board = Board("6h6d6s7c")
    hands = {
        "7h7dQsQd": {},  # 7s pair; 7 > 6 → full house
        "5h5dQsQd": {},  # 5 pair; 5 <= 6 → not
        "AhKcQsQd": {},  # no pair of higher singleton → not
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    fh, fh1 = feature_full_house(counts, board)
    assert fh.tolist() == [1, 0, 0]
    assert fh1.tolist() == [1, 0, 0]


def test_full_house_quads_on_board_none():
    board = Board("AhAdAsAc")
    hands = {
        "QsQdJhJd": {},
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    fh, fh1 = feature_full_house(counts, board)
    assert fh.tolist() == [0]
    assert fh1.tolist() == [0]


def test_trips_unpaired_board():
    # Board: AsKd7h → need pocket pair of board ranks; trips_1 requires AA
    board = Board("AsKd7h")
    hands = {
        "AhAdKhKd": {},  # AA and KK → trips (both), trips_1 true due to AA
        "KhKcQsQd": {},  # KK only → trips true, trips_1 false
        "QhQcJhJd": {},  # no board pairs → no trips
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    trips, trips1, trips_top = feature_trips(counts, board)
    assert trips.tolist() == [1, 1, 0]
    assert trips1.tolist() == [1, 0, 0]
    assert trips_top.tolist() == [0, 0, 0]


def test_trips_single_paired_board():
    # Board: 9h9d6c; trips if holds a 9; trips_1 requires best kicker not on board (A if not present)
    board = Board("9h9d6c")
    hands = {
        "Ah9dKhKd": {},  # has 9 and A kicker → trips, trips_1
        "Kc9sQsQd": {},  # has 9 but no Ace kicker → trips only
        "AhKcQsQd": {},  # no 9 → no trips
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    trips, trips1, trips_top = feature_trips(counts, board)
    assert trips.tolist() == [1, 1, 0]
    assert trips1.tolist() == [1, 0, 0]
    assert trips_top.tolist() == [0, 0, 0]


def test_trips_double_paired_board():
    # Board: 9h9d6c6d; trips if holds either 9 or 6; trips_1 requires 9 and best kicker not on board (A)
    board = Board("9h9d6c6d")
    hands = {
        "Ah9dKhKd": {},  # has 9 and A → trips, trips_1
        "Kc6sQsQd": {},  # has 6 but no Ace → trips only
        "AhKcQsQd": {},  # none of 9/6 → no trips
        "Kh9s4s3d": {},  # has 9 but no Ace → trips only, trips_1 false, trips_top true since 9 is higher of the two paired ranks on board
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    trips, trips1, trips_top = feature_trips(counts, board)
    assert trips.tolist() == [1, 1, 0, 1]
    assert trips1.tolist() == [1, 0, 0, 0]
    # New return: marks hands that contain the higher of the two paired board ranks.
    assert trips_top.tolist() == [1, 0, 0, 1]


def test_trips_board_with_trips_or_quads_none():
    # Trips or quads on board → no trips
    for b in [Board("9h9d9c"), Board("9h9d9c9s")]:
        hands = {
            "Ah9dKhKd": {},
            "Kc6sQsQd": {},
        }
        ranks, _ = encode_hands(hands)
        counts = compute_counts_by_rank(ranks)
        trips, trips1, trips_top = feature_trips(counts, b)
        assert trips.tolist() == [0, 0]
        assert trips1.tolist() == [0, 0]
        assert trips_top.tolist() == [0, 0]


def test_two_pair_unpaired_board():
    # Board: AsKd7h → two_pair if holds any two of {A,K,7}; top_two_pair if holds A and K
    board = Board("AsKd7h")
    hands = {
        "AhKdQsQd": {},  # A and K → two_pair and top_two_pair
        "Ah7dQsQd": {},  # A and 7 → two_pair only
        "QhQcJhJd": {},  # none → no two_pair
        "QhTs7s7h": [],  # pair of 7s but only one 7 on board → no two_pair
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    tp, ttp = feature_two_pair(counts, board)
    assert tp.tolist() == [1, 1, 0, 0]
    assert ttp.tolist() == [1, 0, 0, 0]


def test_two_pair_single_paired_board():
    # Board: 9h9dA cK d (pair 9, singletons A,K; both higher than 9)
    board = Board("9h9dAcKd")
    hands = {
        "AhKdQsQd": {},  # A and K → two_pair and top_two_pair
        "AhQdQsQd": {},  # only A → not two_pair
        "KdQdQsQd": {},  # only K → not two_pair
        "AhJdKdQd": {},  # A and K mixed → two_pair and top_two_pair
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    tp, ttp = feature_two_pair(counts, board)
    assert tp.tolist() == [1, 0, 0, 1]
    assert ttp.tolist() == [1, 0, 0, 1]


def test_two_pair_single_paired_board_insufficient_singletons():
    # Board: JhJd9c9d (double paired) → always 0
    board = Board("JhJd9c9d")
    hands = {
        "AhKdQsQd": {},
        "AhQdQsQd": {},
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    tp, ttp = feature_two_pair(counts, board)
    assert tp.tolist() == [0, 0]
    assert ttp.tolist() == [0, 0]


def test_two_pair_board_with_trips_or_quads_none():
    for b in [Board("9h9d9c"), Board("9h9d9c9s")]:
        hands = {
            "AhKdQsQd": {},
            "AhQdQsQd": {},
        }
        ranks, _ = encode_hands(hands)
        counts = compute_counts_by_rank(ranks)
        tp, ttp = feature_two_pair(counts, b)
        assert tp.tolist() == [0, 0]
        assert ttp.tolist() == [0, 0]


def test_board_pair_unpaired_board():
    board = Board("AsKd7h")
    hands = {
        "AhQdQsQd": {},  # has A (singleton) → board_pair
        "Kh7dQsQd": {},  # has K or 7 → board_pair
        "QhQcJhJd": {},  # none of A,K,7 → no
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    bp, mp, lp = feature_board_pair(counts, board)
    assert bp.tolist() == [1, 1, 0]
    # 3 singletons (A,K,7): mid=K, low=7
    assert mp.tolist() == [0, 1, 0]
    assert lp.tolist() == [0, 1, 0]


def test_board_pair_single_paired_board():
    # Board: 9h9dAcKd → singletons A,K
    board = Board("9h9dAcKd")
    hands = {
        "AhQdQsQd": {},  # has A
        "Kh7dQsQd": {},  # has K
        "QhQcJhJd": {},  # neither
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    bp, mp, lp = feature_board_pair(counts, board)
    assert bp.tolist() == [1, 1, 0]
    # 2 singletons (A,K): mid=none, low=lower of A,K which is K
    assert mp.tolist() == [0, 0, 0]
    assert lp.tolist() == [0, 1, 0]


def test_board_pair_double_paired_or_trips():
    for b in [Board("9h9d6c6d"), Board("9h9d9c")]:
        hands = {
            "AhQdQsQd": {},
            "Kh7dQsQd": {},
        }
        ranks, _ = encode_hands(hands)
        counts = compute_counts_by_rank(ranks)
        bp, mp, lp = feature_board_pair(counts, b)
        # double paired has no singletons → all 0; trips has one singleton (if any) → may be >0 for 9h9d9c (none)
        if (b.rank_counts == 1).any():
            # rare case if trips + a singleton exists on 4th card
            expected_bp = [
                (counts[i, np.flatnonzero(b.rank_counts == 1)[0]] >= 1).astype(int)
                for i in range(len(hands))
            ]
            assert bp.tolist() == expected_bp
            # With exactly 1 singleton, mid/low should be zero
            assert mp.tolist() == [0, 0]
            assert lp.tolist() == [0, 0]
        else:
            assert bp.tolist() == [0, 0]
            assert mp.tolist() == [0, 0]
            assert lp.tolist() == [0, 0]


def test_flush_strength_and_features():
    # Board: AsTs3s2c (three spades on board)
    board = Board("AsTs3s2c")
    hands = {
        "Ks9sQhJh": {},  # top spade K: best → ord 1
        "Qs9sKhJh": {},  # top spade Q: second-best → ord 2
        "Js9sKhJh": {},  # top spade J: other flush → ord 3
        "KdQhJhTh": {},  # no two spades → 0
    }
    ranks, suits = encode_hands(hands)
    ord_mask = feature_flush_strength(ranks, suits, board)
    assert ord_mask.tolist() == [1, 2, 3, 0]

    board = Board("AsQsTs")
    hands = {
        "Kh7h5s3s": {},  # just flush
        "Ks8h6s6d": {},
    }

    ranks, suits = encode_hands(hands)

    ord_mask = feature_flush_strength(ranks, suits, board)
    assert ord_mask.tolist() == [3, 1]

    # Flush draw tests
    board = Board("AsQs9d2c")  # exactly two spades on board
    hands = {
        "Ks9sQhJh": {},  # K-high spade draw → best
        "Js9sKhJh": {},  # J-high spade draw → 2nd
        "Ts9sKhJh": {},  # Q-high spade draw → 3rd
        "KdQhJhTh": {},  # no draw → 0
    }
    ranks, suits = encode_hands(hands)
    dmask = feature_flush_draw_strength(ranks, suits, board)
    assert dmask.tolist() == [1, 2, 3, 0]


def test_flush_draw_strength_second_nut_and_multisuit_regression():
    # Single draw suit: second nut must remain category 2 (never 1)
    board = Board("AsQs9d2c")
    hands = {
        "Js8sKhJh": {},  # spades J-high, with K available as nut -> second nut (2)
        "Ks8sKhJh": {},  # spades K-high -> nut (1)
        "Ts8sKhJh": {},  # spades T-high -> other draw (3)
    }
    ranks, suits = encode_hands(hands)
    dmask = feature_flush_draw_strength(ranks, suits, board)
    assert dmask.tolist() == [2, 1, 3]

    # Two draw suits on turn (2-2 board split): keep best category across suits.
    # Regression: do not downgrade from 1 to 2 when a later suit is second nut.
    board = Board("As7s5d")  # clubs=2, spades=2
    hands = {
        "KsQsTc2c": {},  # clubs K-high draw (1), spades Q-high draw (2) → overall 1 (best)
    }
    ranks, suits = encode_hands(hands)
    dmask = feature_flush_draw_strength(ranks, suits, board)
    assert dmask.tolist() == [1]


def test_flush_blocker_features():
    # Board with three spades (flush present) and also valid for blocker (>=2)
    board = Board("AsTs3s2c")
    hands = {
        "KsQhJhTd": {},  # one spade K → blocker and blocker_1 (highest non-board spade is K)
        "QsQhJhTd": {},  # one spade Q → blocker_high (2nd highest below K with A,T,3 on board)
        "JsQhJhTd": {},  # one spade J → blocker_high (3rd)
        "9sQhJhTd": {},  # one spade 9 → blocker_high (4th or 5th depending on board)
        "KhQhJhTd": {},  # no spade → no blocker
        "KsQsJhTd": {},  # two spades → not a blocker (requires exactly one)
        "KhQhJh4s": {},  # blocker but not high
    }
    ranks, suits = encode_hands(hands)
    blk, blk1, blkh = feature_flush_blocker(ranks, suits, board)
    assert blk.tolist() == [1, 1, 1, 1, 0, 0, 1]
    assert blk1.tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert blkh.tolist() == [0, 1, 1, 1, 0, 0, 0]


def test_double_flush_draw():
    # Turn board with 2-2 suit split: AsQsKhJh -> spades 2, hearts 2
    board = Board("AsQsKhJh")
    hands = {
        "Ts9s8h7h": {},  # 2 spades and 2 hearts → double flush draw
        "AhAsKdKs": {},  # 2 spades, hearts 1 → no
        "AdQdKdJs": {},  # mixed → no
    }
    ranks, suits = encode_hands(hands)
    mask = feature_double_flush_draw(suits, board)
    assert mask.tolist() == [1, 0, 0]

    # Not on turn (river) → always 0
    board = Board("AsQsKh")
    mask = feature_double_flush_draw(suits, board)
    assert mask.tolist() == [0, 0, 0]


def test_backdoor_flushdraw_flop_only():
    board = Board("AsKd7h")  # rainbow flop: spade, diamond, heart singletons
    hands = {
        "QsJs2d3d": {},  # 2 spades + 2 diamonds → backdoor AND double backdoor
        "QcJc2d3d": {},  # 2 clubs (not singleton) + 2 diamonds → backdoor only
        "QhJh2s3c": {},  # 2 hearts + mixed → backdoor only
        "QsJd2h3c": {},  # no suit appears twice → no backdoor
        "QcJc2s3s": {},  # 2 clubs (not singleton) + 2 spades → backdoor only
        "QsJs2h3h": {},  # 2 spades + 2 hearts → backdoor AND double backdoor
    }
    _, suits = encode_hands(hands)
    mask, double_mask = feature_backdoor_flushdraw(suits, board)
    assert mask.tolist() == [1, 1, 1, 0, 1, 1]
    assert double_mask.tolist() == [1, 0, 0, 0, 0, 1]

    # Board with 2 of same suit (non-singleton) + 2 other singletons
    board = Board(
        "AsKs7h"
    )  # spades 2, hearts 1 → only h is singleton (need >=2 singletons for double)
    hands = {
        "QhJh2d3d": {},  # 2 hearts (singleton) + 2 diamonds (not on board) → backdoor only (only 1 singleton)
        "QsJs2h3h": {},  # 2 spades (not singleton) + 2 hearts → backdoor only
    }
    _, suits = encode_hands(hands)
    mask, double_mask = feature_backdoor_flushdraw(suits, board)
    assert mask.tolist() == [1, 1]
    assert double_mask.tolist() == [0, 0]  # Need at least 2 singleton suits on board

    board = Board("AsQs9d2c")  # turn, never a backdoor flush draw
    _, suits = encode_hands({"QsJs2d3d": {}})
    mask, double_mask = feature_backdoor_flushdraw(suits, board)
    assert mask.tolist() == [0]
    assert double_mask.tolist() == [0]


def test_drr_feature():
    # Board: AsKd7h; drr requires TWO pocket pairs not on board (no restrictions)
    board = Board("AsKd7h")
    hands = {
        "QhQsJhJd": {},  # QQ, JJ both not on board → drr
        "AhAdKhKd": {},  # AA on board, KK on board → not drr
        "QhQs2h2d": {},  # QQ, 22 both not on board → drr
        "7h7s2d3c": {},  # 77 on board, no second pair → not drr
        "QhJs2h3d": {},  # no pocket pairs → not drr
        "AhAdQhQd": {},  # AA on board, QQ not on board → only one valid pair, not drr
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    from features.feature_ops import feature_drr

    drr = feature_drr(counts, board)
    assert drr.tolist() == [1, 0, 1, 0, 0, 0]

    # Board with paired rank: AhAdKc
    board = Board("AhAdKc")
    hands = {
        "QhQsJhJd": {},  # QQ, JJ both not on board → drr
        "KhKdQhQd": {},  # KK on board, QQ not on board → only one valid pair
        "9h9s8h8d": {},  # 99, 88 both not on board → drr
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    drr = feature_drr(counts, board)
    assert drr.tolist() == [1, 0, 1]


def test_rr_feature():
    # Board: AsKd7h (highest=A); rr are pocket pairs below A and not on board
    board = Board("AsKd7h")
    hands = {
        "AhAdKhKd": {},  # AA, KK on board ranks → not rr
        "QhQsJhJd": {},  # QQ pair below A and not on board → rr
        "7h7s2d3c": {},  # 77 equals board rank 7 on board → not rr
        "2h2s3d4c": {},  # 22 below A and not on board → rr
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    rr = feature_rr(counts, board)
    assert rr.tolist() == [0, 1, 0, 1]
    board = Board("TsTd7h")
    hands = {
        "AhAdKhKd": {},  # AA, KK on board ranks → not rr
        "QhJsTh9d": {},  # QQ pair below A and not on board → rr
        "6h6s2d3c": {},  # rr
        "2h2s3d4c": {},  # 22 below A and not on board → rr
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)
    rr = feature_rr(counts, board)
    assert rr.tolist() == [0, 0, 1, 1]


def test_rr_high_low_feature():
    # Board: AsKd7h -> highest=A(14)
    # rr-eligible ranks below A and not on board: 2,3,4,5,6,8,9,T,J,Q
    # Split ascending with odd-middle-to-high rule:
    #   low: 2,3,4,5,6
    #   high: 8,9,T,J,Q
    board = Board("AsKd7h")
    hands = {
        "QhQsJhJd": {},  # high_rr
        "6h6s2d3c": {},  # low_rr
        "8h8s4d4c": {},  # both
        "AhAdKhKd": {},  # none (board ranks)
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)

    high_rr, low_rr = feature_rr_high_low(counts, board)
    assert high_rr.tolist() == [1, 0, 1, 0]
    assert low_rr.tolist() == [0, 1, 1, 0]

    # Even-count split example
    # Board: TsTd7h -> highest=T(10)
    # rr-eligible: 2,3,4,5,6,8,9
    # low: 2,3,4
    # high: 5,6,8,9
    board = Board("TsTd7h")
    hands = {
        "9h9s2d3c": {},  # high only
        "4h4sAhKc": {},  # low only
        "6h6s3d3c": {},  # both
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)

    high_rr, low_rr = feature_rr_high_low(counts, board)
    assert high_rr.tolist() == [1, 0, 1]
    assert low_rr.tolist() == [0, 1, 1]


def test_straight_feature_flags_from_helper():
    board = Board("6h7d9s")
    hands = {
        "5c8cTdAd": {},  # highest possible made straight
        "5c8cJdAd": {},  # second-best made straight
        "AcKcQd2d": {},  # no straight
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)

    made_high, hand_sum_used = straight_helper(counts, board.ranks)

    straight = (made_high >= 1).astype(np.int8)
    nut_straight = (made_high == 1).astype(np.int8)

    assert made_high.tolist() == [1, 2, 0]
    assert hand_sum_used.tolist() == [18, 13, 0]
    assert straight.tolist() == [1, 1, 0]
    assert nut_straight.tolist() == [1, 0, 0]


def test_straight_draw_feature_flags_from_helper():
    board = Board("7h8dKs")
    hands = {
        "5c9cAsQd": {},  # gutshot
        "5c6cAsQd": {},  # oesd
        "5c9cTcAd": {},  # oesd_1
        "5c6c9hAd": {},  # wrap
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)

    made_high, hand_sum_used = straight_helper(counts, board.ranks)
    outs_count, draw_made_high = feature_straight_draw(
        counts, board.ranks, made_high, hand_sum_used
    )

    gutshot = (outs_count == 1).astype(np.int8)
    gutshot_1 = ((outs_count == 1) & (draw_made_high == 1)).astype(np.int8)
    oesd = (outs_count == 2).astype(np.int8)
    oesd_1 = ((outs_count == 2) & (draw_made_high == 1)).astype(np.int8)
    wrap = (outs_count > 2).astype(np.int8)
    wrap_1 = ((outs_count > 2) & (draw_made_high == 1)).astype(np.int8)

    assert made_high.tolist() == [0, 0, 0, 0]
    assert hand_sum_used.tolist() == [0, 0, 0, 0]
    assert outs_count.tolist() == [1, 2, 2, 5]
    assert draw_made_high.tolist() == [2, 3, 1, 3]
    assert gutshot.tolist() == [1, 0, 0, 0]
    assert gutshot_1.tolist() == [0, 0, 0, 0]
    assert oesd.tolist() == [0, 1, 1, 0]
    assert oesd_1.tolist() == [0, 0, 1, 0]
    assert wrap.tolist() == [0, 0, 0, 1]
    assert wrap_1.tolist() == [0, 0, 0, 0]

    board = Board("7h8d9s")
    hands = {
        "5cJcAsQd": {},  # gutshot_1
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)

    made_high, hand_sum_used = straight_helper(counts, board.ranks)
    outs_count, draw_made_high = feature_straight_draw(
        counts, board.ranks, made_high, hand_sum_used
    )

    gutshot = (outs_count == 1).astype(np.int8)
    gutshot_1 = ((outs_count == 1) & (draw_made_high == 1)).astype(np.int8)

    assert made_high.tolist() == [0]
    assert hand_sum_used.tolist() == [0]
    assert outs_count.tolist() == [1]
    assert draw_made_high.tolist() == [1]
    assert gutshot.tolist() == [1]
    assert gutshot_1.tolist() == [1]

    board = Board("5h6d9s")
    hands = {
        "7c8cTdAd": {},  # wrap_1 while already straight
    }
    ranks, _ = encode_hands(hands)
    counts = compute_counts_by_rank(ranks)

    made_high, hand_sum_used = straight_helper(counts, board.ranks)
    outs_count, draw_made_high = feature_straight_draw(
        counts, board.ranks, made_high, hand_sum_used
    )

    wrap = (outs_count > 2).astype(np.int8)
    wrap_1 = ((outs_count > 2) & (draw_made_high == 1)).astype(np.int8)

    assert made_high.tolist() == [1]
    assert hand_sum_used.tolist() == [15]
    assert outs_count.tolist() == [3]
    assert draw_made_high.tolist() == [1]
    assert wrap.tolist() == [1]
    assert wrap_1.tolist() == [1]
