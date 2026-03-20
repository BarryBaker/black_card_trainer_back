from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


def compute_counts_by_rank(ranks: np.ndarray) -> np.ndarray:
    """
    Count occurrences of ranks 2..14 per hand.
    ranks: (N,4) int array
    returns: (N, 15) int counts (index by rank value), indices 0/1 unused
    """
    n = ranks.shape[0]
    counts = np.zeros((n, 15), dtype=np.int8)
    # ranks sorted; use vectorized add via advanced indexing
    for pos in range(4):
        counts[np.arange(n), ranks[:, pos]] += 1

    return counts


def feature_overpair(
    counts_by_rank: np.ndarray, board_rank_counts: np.ndarray
) -> np.ndarray:
    """
    Overpair: player holds a pocket pair whose rank is strictly higher than the
    highest rank on the board.
    """
    # Highest pocket pair rank per hand (0 if none). Consider ranks 2..14.
    pair_mask = counts_by_rank[:, 2:15] >= 2
    ranks = np.arange(2, 15, dtype=np.int16)[None, :]
    pair_ranks = (pair_mask.astype(np.int16) * ranks).max(axis=1).astype(np.int16)

    # Highest board rank (0 if board empty). Consider ranks 2..14.
    board_present = board_rank_counts[2:15] > 0
    if board_present.any():
        board_highest_rank = int(
            (board_present.astype(np.int16) * ranks.squeeze(0)).max()
        )
    else:
        board_highest_rank = 0

    return ((pair_ranks > board_highest_rank) & (pair_ranks > 0)).astype(np.int8)


from .board import Board


def feature_top_pair(
    counts_by_rank: np.ndarray, board: Board
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-pair mask and each hand's highest kicker not on board.

    Rules:
      - If board is not paired: top pair means holding the board's highest rank.
      - If board is paired/double-paired/trips: top pair means holding the highest
        board rank that appears exactly once on the board (singleton). If none,
        top pair is impossible.

    Returns:
      - top_pair_mask: (N,) int8
      - highest_not_on_board_rank: (N,) int8, 0 if none
    """
    # Hand has rank presence 2..14
    hand_has = counts_by_rank[:, 2:15] >= 1
    ranks = np.arange(2, 15, dtype=np.int16)[None, :]

    # Board counts and singleton logic
    bcounts = board.rank_counts  # length 15
    board_present = bcounts[2:15] > 0
    board_singletons = bcounts[2:15] == 1

    if (
        not board.is_paired
        and not board.has_trips_on_board
        and not board.is_double_paired
        and not board.has_quads_on_board
    ):
        # No duplicates on board: top pair is board.highest_rank
        top_rank = int(board.highest_rank)
        top_pair_mask = (
            (counts_by_rank[:, top_rank] >= 1).astype(np.int8)
            if top_rank > 0
            else np.zeros(counts_by_rank.shape[0], dtype=np.int8)
        )
    else:
        # Duplicates present: use highest singleton rank, if any
        if board_singletons.any():
            # Compute highest singleton rank value
            singleton_ranks_vals = (
                (board_singletons.astype(np.int16)[None, :] * ranks)
                .max(axis=1)
                .astype(np.int16)
            )
            top_rank = int(singleton_ranks_vals[0])
            top_pair_mask = (
                (counts_by_rank[:, top_rank] >= 1).astype(np.int8)
                if top_rank > 0
                else np.zeros(counts_by_rank.shape[0], dtype=np.int8)
            )
        else:
            top_rank = 0
            top_pair_mask = np.zeros(counts_by_rank.shape[0], dtype=np.int8)

    # Highest rank not present on board, per hand (kicker candidate)
    not_on_board = (~board_present)[None, :]
    kicker_mask = hand_has & not_on_board
    highest_not_on_board = (
        (kicker_mask.astype(np.int16) * ranks).max(axis=1).astype(np.int16)
    )
    highest_not_on_board = highest_not_on_board.astype(np.int8)

    return top_pair_mask, highest_not_on_board


def feature_quads(counts_by_rank: np.ndarray, board: Board) -> np.ndarray:
    """
    1 if the hand makes quads given the board, per rules:
      - If board is unpaired (no duplicates) → no quads
      - If single paired board → need pocket pair of that paired rank (2 from hand)
      - If double paired board → need pocket pair of either paired rank
      - If trips on board (including full house) → need at least one card of the trips rank
      - If quads on board → no quads (board locks quads)
    """
    bcounts = board.rank_counts  # (15,)
    # Early exits
    if board.has_quads_on_board:
        return np.zeros(counts_by_rank.shape[0], dtype=np.int8)

    has_duplicates = (bcounts >= 2).any()
    if not has_duplicates:
        return np.zeros(counts_by_rank.shape[0], dtype=np.int8)

    # Trips case (covers full houses as well)
    trips_ranks = np.where(bcounts == 3)[0]
    if trips_ranks.size > 0:
        r = int(trips_ranks[0])
        return (counts_by_rank[:, r] >= 1).astype(np.int8)

    # Double-paired or single-paired
    paired_ranks = np.where(bcounts == 2)[0]
    if paired_ranks.size == 0:
        return np.zeros(counts_by_rank.shape[0], dtype=np.int8)

    # Need pocket pair of any paired rank
    hand_pairs_mask = counts_by_rank[:, paired_ranks] >= 2
    has_quads = hand_pairs_mask.any(axis=1)
    return has_quads.astype(np.int8)


def feature_has_rank(counts_by_rank: np.ndarray, rank_value: int) -> np.ndarray:
    return (counts_by_rank[:, rank_value] >= 1).astype(np.int8)


def feature_rr(counts_by_rank: np.ndarray, board: Board) -> np.ndarray:
    """
    rr: player has a pocket pair (>=2 of same rank in hand) of a rank that:
      - does NOT appear on the board, and
      - is strictly lower than the highest rank on the board.
    Returns (N,) int8 mask.
    """
    n = counts_by_rank.shape[0]
    # Board presence mask for ranks 2..14
    present_on_board = board.rank_counts[2:15] > 0
    highest_board = int(board.highest_rank) if board.highest_rank else 0
    # Candidate hand pocket pair ranks: 2..14
    hand_pair_mask = counts_by_rank[:, 2:15] >= 2
    # Mask out ranks present on board
    not_on_board = ~present_on_board
    if highest_board > 0:
        lower_than_high = np.arange(2, 15) < highest_board
    else:
        lower_than_high = np.zeros(13, dtype=bool)
    allowed_ranks = not_on_board & lower_than_high
    if not allowed_ranks.any():
        return np.zeros(n, dtype=np.int8)
    rr_mask = (hand_pair_mask[:, allowed_ranks].any(axis=1)).astype(np.int8)
    return rr_mask


def _rr_allowed_rank_values(board: Board) -> np.ndarray:
    """
    Return rank values (2..14) that qualify for rr on this board:
      - strictly lower than board highest rank
      - not present on board
    """
    highest_board = int(board.highest_rank) if board.highest_rank else 0
    if highest_board <= 0:
        return np.empty(0, dtype=np.int16)

    rank_values = np.arange(2, 15, dtype=np.int16)
    present_on_board = board.rank_counts[2:15] > 0
    lower_than_high = rank_values < highest_board
    allowed_mask = (~present_on_board) & lower_than_high
    return rank_values[allowed_mask]


def feature_rr_high_low(
    counts_by_rank: np.ndarray, board: Board
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split rr-eligible ranks into high and low buckets by rank value.

    Steps:
      - Build rr-eligible ranks: below highest board rank and not on board.
      - Sort ascending and split equally into two groups.
      - If odd count, the middle rank is assigned to high_rr.

    Returns:
      - high_rr: (N,) int8 mask, has a pocket pair of any high-group rank
      - low_rr: (N,) int8 mask, has a pocket pair of any low-group rank
    """
    n = counts_by_rank.shape[0]
    allowed_ranks = _rr_allowed_rank_values(board)
    if allowed_ranks.size == 0:
        z = np.zeros(n, dtype=np.int8)
        return z, z

    # Ascending by value; high gets the larger half, including middle if odd.
    allowed_sorted = np.sort(allowed_ranks)
    split_idx = allowed_sorted.size // 2
    low_group = allowed_sorted[:split_idx]
    high_group = allowed_sorted[split_idx:]

    high_rr = (
        (counts_by_rank[:, high_group] >= 2).any(axis=1).astype(np.int8)
        if high_group.size
        else np.zeros(n, dtype=np.int8)
    )
    low_rr = (
        (counts_by_rank[:, low_group] >= 2).any(axis=1).astype(np.int8)
        if low_group.size
        else np.zeros(n, dtype=np.int8)
    )
    return high_rr, low_rr


def feature_drr(counts_by_rank: np.ndarray, board: Board) -> np.ndarray:
    """
    drr (double rr): player has TWO pocket pairs (>=2 of same rank in hand) of ranks that:
      - do NOT appear on the board
    Returns (N,) int8 mask.
    """
    n = counts_by_rank.shape[0]
    # Board presence mask for ranks 2..14
    present_on_board = board.rank_counts[2:15] > 0
    # Candidate hand pocket pair ranks: 2..14
    hand_pair_mask = counts_by_rank[:, 2:15] >= 2
    # Mask out ranks present on board
    not_on_board = ~present_on_board
    allowed_ranks = not_on_board
    if not allowed_ranks.any():
        return np.zeros(n, dtype=np.int8)
    # Count how many valid pocket pairs each hand has
    valid_pairs_count = (hand_pair_mask[:, allowed_ranks]).sum(axis=1)
    drr_mask = (valid_pairs_count >= 2).astype(np.int8)
    return drr_mask


def feature_full_house(
    counts_by_rank: np.ndarray, board: Board
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full house detection per rules:
      - If board unpaired (no duplicates) → no full house
      - If exactly one pair on board → full house if:
          * pocket pair of any singleton board rank, OR
          * one card of the board pair rank AND one card of any singleton board rank
      - If double paired → full house if:
          * one card of each pair rank, OR
          * one card of a pair rank AND one card of any singleton board rank (if any), OR
          * pocket pair of any singleton board rank (if any)
      - If trips on board → full house only if:
          * there exists at least one singleton board rank higher than the trips rank, AND
          * player holds a pocket pair of any of those higher singleton ranks
      - If quads on board → no full house
    """
    b = board.rank_counts
    n = counts_by_rank.shape[0]
    if board.has_quads_on_board:
        z = np.zeros(n, dtype=np.int8)
        return z, z

    has_duplicates = (b >= 2).any()
    if not has_duplicates:
        z = np.zeros(n, dtype=np.int8)
        return z, z

    paired_ranks = np.flatnonzero(b == 2)
    trips_ranks = np.flatnonzero(b == 3)
    singleton_ranks = np.flatnonzero(b == 1)

    # Helper: vectorized presence checks
    def hand_has_any(ranks_list: np.ndarray) -> np.ndarray:
        if ranks_list.size == 0:
            return np.zeros(n, dtype=bool)
        return (counts_by_rank[:, ranks_list] >= 1).any(axis=1)

    def hand_pair_any(ranks_list: np.ndarray) -> np.ndarray:
        if ranks_list.size == 0:
            return np.zeros(n, dtype=bool)
        return (counts_by_rank[:, ranks_list] >= 2).any(axis=1)

    # Trips on board case
    if trips_ranks.size > 0:
        t = int(trips_ranks[0])
        higher_singletons = singleton_ranks[singleton_ranks > t]
        if higher_singletons.size == 0:
            z = np.zeros(n, dtype=np.int8)
            return z, z
        base = hand_pair_any(higher_singletons).astype(np.int8)
        highest_higher = int(higher_singletons.max())
        best = (counts_by_rank[:, highest_higher] >= 2).astype(np.int8)
        return base, best

    # Exactly one pair on board
    if paired_ranks.size == 1:
        pr = int(paired_ranks[0])
        pair_of_singleton = hand_pair_any(singleton_ranks)
        one_pair_and_one_singleton = (counts_by_rank[:, pr] >= 1) & hand_has_any(
            singleton_ranks
        )
        base = (pair_of_singleton | one_pair_and_one_singleton).astype(np.int8)
        # Best selection
        higher_singletons = singleton_ranks[singleton_ranks > pr]
        if higher_singletons.size >= 1:
            highest_singleton = int(higher_singletons.max())
            best = (counts_by_rank[:, highest_singleton] >= 2).astype(np.int8)
        else:
            # No singleton above pair; best is having one from pair and one from highest singleton (lower)
            highest_singleton = (
                int(singleton_ranks.max()) if singleton_ranks.size else 0
            )
            if highest_singleton > 0:
                best = (
                    (counts_by_rank[:, pr] >= 1)
                    & (counts_by_rank[:, highest_singleton] >= 1)
                ).astype(np.int8)
            else:
                best = np.zeros(n, dtype=np.int8)
        return base, best

    # Double paired board (two pairs); singleton_ranks may be empty on 4-card board
    if paired_ranks.size >= 2:
        r1, r2 = int(paired_ranks[0]), int(paired_ranks[1])
        has_r1 = counts_by_rank[:, r1] >= 1
        has_r2 = counts_by_rank[:, r2] >= 1
        any_singleton = hand_has_any(singleton_ranks)
        pair_of_singleton = hand_pair_any(singleton_ranks)
        base = (
            (has_r1 & has_r2)
            | (has_r1 & any_singleton)
            | (has_r2 & any_singleton)
            | pair_of_singleton
        )
        # No explicit rule provided for best on double-paired; set to 0 by definition
        best = np.zeros(n, dtype=np.int8)
        return base.astype(np.int8), best

    # Fallback
    z = np.zeros(n, dtype=np.int8)
    return z, z


def feature_trips(
    counts_by_rank: np.ndarray, board: Board
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Compute three masks:
      - trips: has trips under rules
      - trips_1: best trips among trips per rules
            - trips_top: on double-paired boards, hand contains the higher paired board rank

    Rules:
      - If trips/quads on board → no trips
      - Unpaired board (no duplicates): need pocket pair of any board rank; trips_1 when pair is the highest board rank
      - Paired board (exactly one pair): trips if holds at least one of the paired rank;
        trips_1 if also holds the highest possible rank not on board
      - Double paired: trips if holds at least one card of either paired rank;
        trips_1 if holds the higher paired rank AND the highest rank not on board
    """
    n = counts_by_rank.shape[0]
    b = board.rank_counts
    # No trips when trips/quads are already on board
    if board.has_trips_on_board or board.has_quads_on_board:
        z = np.zeros(n, dtype=np.int8)
        return z, z, z

    # Board structure
    duplicates_present = (b >= 2).any()
    singleton_ranks = np.flatnonzero(b == 1)
    paired_ranks = np.flatnonzero(b == 2)

    # Highest not-on-board rank value
    board_present = b[2:15] > 0
    not_on_board_mask = ~board_present
    if not_on_board_mask.any():
        highest_not_on_board_value = int(
            np.arange(2, 15, dtype=np.int16)[not_on_board_mask].max()
        )
    else:
        highest_not_on_board_value = 0

    trips_mask = np.zeros(n, dtype=bool)
    trips1_mask = np.zeros(n, dtype=bool)
    has_higher_pair = np.zeros(n, dtype=bool)

    if not duplicates_present:
        # Unpaired board: need pocket pair of any board rank; trips_1 requires the pair at highest board rank
        pair_any_board_rank = (
            (counts_by_rank[:, singleton_ranks] >= 2).any(axis=1)
            if singleton_ranks.size
            else np.zeros(n, dtype=bool)
        )
        highest_board_rank = int(board.highest_rank)
        pair_at_highest = (
            (counts_by_rank[:, highest_board_rank] >= 2)
            if highest_board_rank > 0
            else np.zeros(n, dtype=bool)
        )
        trips_mask = pair_any_board_rank
        trips1_mask = pair_at_highest
    elif paired_ranks.size == 1:
        # Exactly one pair: trips if holds a card of the paired rank
        pr = int(paired_ranks[0])
        has_pair_rank = counts_by_rank[:, pr] >= 1
        has_best_kicker = (
            (counts_by_rank[:, highest_not_on_board_value] >= 1)
            if highest_not_on_board_value > 0
            else np.zeros(n, dtype=bool)
        )
        trips_mask = has_pair_rank
        trips1_mask = has_pair_rank & has_best_kicker
    elif paired_ranks.size >= 2:
        # Double paired: trips if holds at least one of either paired rank
        r1, r2 = int(paired_ranks[0]), int(paired_ranks[1])
        has_r1 = counts_by_rank[:, r1] >= 1
        has_r2 = counts_by_rank[:, r2] >= 1
        trips_mask = has_r1 | has_r2
        # trips_1 requires holding the higher paired rank and the best kicker
        higher_pair = max(r1, r2)
        has_higher_pair = counts_by_rank[:, higher_pair] >= 1
        has_best_kicker = (
            (counts_by_rank[:, highest_not_on_board_value] >= 1)
            if highest_not_on_board_value > 0
            else np.zeros(n, dtype=bool)
        )
        trips1_mask = has_higher_pair & has_best_kicker

    return (
        trips_mask.astype(np.int8),
        trips1_mask.astype(np.int8),
        has_higher_pair.astype(np.int8),
    )


def feature_two_pair(
    counts_by_rank: np.ndarray, board: Board
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two_pair and top_two_pair per rules:
      - Unpaired board: two_pair if holds at least two distinct board ranks;
        top_two_pair if holds both highest and second-highest board ranks.
      - Single paired board: consider only singleton board ranks higher than the pair rank.
        If there are at least two such singletons, two_pair if holds any two of them;
        top_two_pair if holds both of the two highest among them.
      - Double paired / trips / quads on board: none.
    """
    n = counts_by_rank.shape[0]
    b = board.rank_counts
    # No two-pair when board has trips/quads or double-paired per definition
    if board.has_trips_on_board or board.has_quads_on_board or board.is_double_paired:
        z = np.zeros(n, dtype=np.int8)
        return z, z

    # Helper: count distinct ranks from a set that appear in the hand
    def count_distinct_from(ranks_list: np.ndarray) -> np.ndarray:
        if ranks_list.size == 0:
            return np.zeros(n, dtype=np.int16)
        present = counts_by_rank[:, ranks_list] >= 1
        return present.sum(axis=1).astype(np.int16)

    # Unpaired: all board ranks are singletons
    if not (b >= 2).any():
        singletons = np.flatnonzero(b == 1)
        num_hits = count_distinct_from(singletons)
        two_pair = num_hits >= 2
        # Top two are highest and second-highest board ranks
        top = int(board.highest_rank)
        second = int(board.second_highest_rank)
        has_top = (counts_by_rank[:, top] >= 1) if top > 0 else np.zeros(n, dtype=bool)
        has_second = (
            (counts_by_rank[:, second] >= 1) if second > 0 else np.zeros(n, dtype=bool)
        )
        top_two_pair = has_top & has_second
        return two_pair.astype(np.int8), top_two_pair.astype(np.int8)

    # Single paired: exactly one pair
    if (
        (b == 2).sum() == 1
        and not board.has_trips_on_board
        and not board.has_quads_on_board
    ):
        pair_rank = int(np.flatnonzero(b == 2)[0])
        # singletons strictly higher than the pair rank
        higher_singletons = np.array(
            [r for r in np.flatnonzero(b == 1) if r > pair_rank], dtype=np.int16
        )
        if higher_singletons.size < 2:
            z = np.zeros(n, dtype=np.int8)
            return z, z
        num_hits = count_distinct_from(higher_singletons)
        two_pair = num_hits >= 2
        # Top two singletons among higher_singletons
        top2_sorted = np.sort(higher_singletons)[-2:]
        t1, t2 = int(top2_sorted[-1]), int(top2_sorted[-2])
        has_t1 = counts_by_rank[:, t1] >= 1
        has_t2 = counts_by_rank[:, t2] >= 1
        top_two_pair = has_t1 & has_t2
        return two_pair.astype(np.int8), top_two_pair.astype(np.int8)

    # Fallback: no two-pair by this definition
    z = np.zeros(n, dtype=np.int8)
    return z, z


def feature_board_pair(
    counts_by_rank: np.ndarray, board: Board
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns three masks (N,) int8:
      - board_pair: hand contains at least one rank that matches any singleton on board
      - mid_pair: hand contains a rank matching the board's mid singleton(s)
      - low_pair: hand contains a rank matching the board's low singleton(s)

    Mapping rules by number of singletons on board:
      - 0 or 1 singleton: mid_pair = low_pair = 0
      - 2 singletons: low_pair = lower (second-highest) singleton
      - 3 singletons: mid_pair = second-highest; low_pair = third-highest
      - 4 singletons: mid_pair = second and third-highest; low_pair = fourth-highest
      - 5 singletons: mid_pair = second and third-highest; low_pair = fourth and fifth-highest
    """
    n = counts_by_rank.shape[0]
    b = board.rank_counts
    singleton_ranks = np.flatnonzero(b == 1)
    if singleton_ranks.size == 0:
        z = np.zeros(n, dtype=np.int8)
        return z, z, z

    # board_pair: any singleton
    board_pair_mask = (
        (counts_by_rank[:, singleton_ranks] >= 1).any(axis=1).astype(np.int8)
    )

    # Determine mid/low sets by sorted singleton ranks (descending)
    s_sorted = np.sort(singleton_ranks)[::-1]  # high → low
    k = s_sorted.size
    mid_set: np.ndarray
    low_set: np.ndarray
    if k <= 1:
        mid_set = np.array([], dtype=int)
        low_set = np.array([], dtype=int)
    elif k == 2:
        mid_set = np.array([], dtype=int)
        low_set = s_sorted[-1:]
    elif k == 3:
        mid_set = s_sorted[1:2]
        low_set = s_sorted[2:3]
    elif k == 4:
        mid_set = s_sorted[1:3]
        low_set = s_sorted[3:4]
    else:
        # k >= 5
        mid_set = s_sorted[1:3]
        low_set = s_sorted[3:5]

    if mid_set.size:
        mid_pair_mask = (counts_by_rank[:, mid_set] >= 1).any(axis=1).astype(np.int8)
    else:
        mid_pair_mask = np.zeros(n, dtype=np.int8)
    if low_set.size:
        low_pair_mask = (counts_by_rank[:, low_set] >= 1).any(axis=1).astype(np.int8)
    else:
        low_pair_mask = np.zeros(n, dtype=np.int8)

    return board_pair_mask, mid_pair_mask, low_pair_mask


def feature_flush_strength(
    hand_ranks: np.ndarray, hand_suits: np.ndarray, board: Board
) -> np.ndarray:
    """
    Returns per-hand ordinal mask for flush strength on current board:
      - 0: no flush (board has <3 of any suit, or hand has <2 of that suit)
      - 1: player's highest rank in the flush suit equals the highest rank not on board for that suit
      - 2: player's highest rank is the second-highest possible not on board

    Notes:
      - Assumes Omaha 4-card hands: `hand_ranks` shape (N,4) with values 2..14; `hand_suits` shape (N,4) with values 0..3
      - Board suits/ranks from Board
    """
    n = hand_ranks.shape[0]
    # Quick board suit counts
    suit_counts = board.suit_counts  # length 4
    ranks_board = board.ranks  # board ranks array
    suits_board = board.suits  # board suits array

    # Determine flush candidate suits on board (>=3 on board)
    candidate_suits = np.flatnonzero(suit_counts >= 3)
    if candidate_suits.size == 0:
        return np.zeros(n, dtype=np.int8)

    # For each candidate suit, compute available non-board ranks (descending)
    # Build a mask for ranks 2..14 in that suit not present on board
    result = np.zeros(n, dtype=np.int8)

    for suit in candidate_suits.tolist():
        # Board ranks in this suit
        board_suit_ranks = ranks_board[suits_board == suit]
        board_rank_present = np.zeros(15, dtype=bool)
        board_rank_present[board_suit_ranks] = True

        # Hand has at least two of this suit?
        hand_suit_counts = (hand_suits == suit).sum(axis=1)
        eligible = hand_suit_counts >= 2
        if not eligible.any():
            continue

        # Highest available non-board ranks for this suit (ascending list of available ranks)
        available = [r for r in range(2, 15) if not board_rank_present[r]]
        if len(available) == 0:
            continue
        highest_avail = int(available[-1])
        second_avail = int(available[-2]) if len(available) >= 2 else 0

        # Compute per-hand highest rank of this suit
        suit_ranks = np.where(hand_suits == suit, hand_ranks, -1)
        hand_top = suit_ranks.max(axis=1)

        is_best = eligible & (hand_top == highest_avail)
        is_second = eligible & (second_avail > 0) & (hand_top == second_avail)
        # Default ordinal for other flushes: 3
        result = np.where(is_best, 1, result)
        result = np.where(~is_best & is_second, 2, result)
        other_flush = eligible & ~(is_best | is_second)
        result = np.where(other_flush & (result == 0), 3, result)

    return result.astype(np.int8)


def feature_flush_draw_strength(
    hand_ranks: np.ndarray, hand_suits: np.ndarray, board: Board
) -> np.ndarray:
    """
    Returns per-hand ordinal mask for flush draw strength on current board:
      - 0: no flush draw (board has <2 of a suit, or hand has <2 of that suit)
      - 1: player's highest rank in the draw suit equals the highest rank not on board for that suit
      - 2: player's highest rank equals the second-highest rank not on board for that suit
      - 3: other flush draw (has two in suit but not top-2 highest available)
    A flush draw requires at least two suited on board and two in hand (same suit).
    """
    n = hand_ranks.shape[0]
    suit_counts = board.suit_counts
    ranks_board = board.ranks
    suits_board = board.suits

    # Flush draws consider only suits that appear exactly twice on the board
    candidate_suits = np.flatnonzero(suit_counts == 2)
    if candidate_suits.size == 0:
        return np.zeros(n, dtype=np.int8)

    result = np.zeros(n, dtype=np.int8)
    for suit in candidate_suits.tolist():
        board_suit_ranks = ranks_board[suits_board == suit]
        board_rank_present = np.zeros(15, dtype=bool)
        board_rank_present[board_suit_ranks] = True

        hand_suit_counts = (hand_suits == suit).sum(axis=1)
        eligible = hand_suit_counts >= 2
        if not eligible.any():
            continue

        available = [r for r in range(2, 15) if not board_rank_present[r]]
        if len(available) == 0:
            continue
        highest_avail = int(available[-1])
        second_avail = int(available[-2]) if len(available) >= 2 else 0

        suit_ranks = np.where(hand_suits == suit, hand_ranks, -1)
        hand_top = suit_ranks.max(axis=1)

        is_best = eligible & (hand_top == highest_avail)
        is_second = eligible & (second_avail > 0) & (hand_top == second_avail)
        other = eligible & ~(is_best | is_second)

        suit_result = np.zeros(n, dtype=np.int8)
        suit_result = np.where(is_best, 1, suit_result)
        suit_result = np.where(~is_best & is_second, 2, suit_result)
        suit_result = np.where(other, 3, suit_result)

        # If multiple draw suits exist (e.g., 2-2 turn board), keep the strongest
        # category per hand: 1 (best) < 2 < 3, while 0 means no draw for that suit.
        result = np.where(
            result == 0,
            suit_result,
            np.where(suit_result == 0, result, np.minimum(result, suit_result)),
        )

    return result.astype(np.int8)


def feature_backdoor_flushdraw(
    hand_suits: np.ndarray, board: Board
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backdoor flush draw (flop only):
      - Board must have exactly 3 cards
      - Consider suits that appear exactly once on the board
      - First mask: hand has at least two cards of any singleton board suit
      - Second mask (double backdoor): hand has exactly 2 cards from one singleton suit
        AND exactly 2 cards from a different singleton suit (2-2 split)
    Returns: (backdoor_mask, double_backdoor_mask) as (N,) int8 masks.
    """
    n = hand_suits.shape[0]
    if board.num_cards != 3:
        z = np.zeros(n, dtype=np.int8)
        return z, z

    singleton_suits = np.flatnonzero(board.suit_counts == 1)
    if singleton_suits.size == 0:
        z = np.zeros(n, dtype=np.int8)
        return z, z

    # hand_counts[i, j] = number of cards in hand i that match singleton suit j
    hand_counts = (hand_suits[:, :, None] == singleton_suits[None, None, :]).sum(axis=1)

    # Regular backdoor: at least 2 cards of any singleton suit
    backdoor_mask = (hand_counts >= 2).any(axis=1).astype(np.int8)

    # Double backdoor: exactly 2 cards from one singleton AND exactly 2 from another
    # Need at least 2 singleton suits on board
    if singleton_suits.size < 2:
        return backdoor_mask, np.zeros(n, dtype=np.int8)

    # Count how many singleton suits have exactly 2 cards per hand
    exactly_two_mask = hand_counts == 2
    num_with_exactly_two = exactly_two_mask.sum(axis=1)
    double_backdoor_mask = (num_with_exactly_two >= 2).astype(np.int8)

    return backdoor_mask, double_backdoor_mask


def feature_flush_blocker(
    hand_ranks: np.ndarray, hand_suits: np.ndarray, board: Board
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute flush blocker features:
      - blocker: 1 if board has at least two of a suit and hand has exactly one of that suit
      - blocker_1: 1 if blocker and the held card is the highest non-board rank of that suit
      - blocker_high: 1 if blocker and the held card is the 2nd/3rd/4th/5th highest non-board rank of that suit
    """
    n = hand_ranks.shape[0]
    suit_counts = board.suit_counts
    ranks_board = board.ranks
    suits_board = board.suits

    blocker = np.zeros(n, dtype=np.int8)
    blocker_1 = np.zeros(n, dtype=np.int8)
    blocker_high = np.zeros(n, dtype=np.int8)

    candidate_suits = np.flatnonzero(suit_counts >= 2)
    if candidate_suits.size == 0:
        return blocker, blocker_1, blocker_high

    for suit in candidate_suits.tolist():
        board_suit_ranks = ranks_board[suits_board == suit]
        board_rank_present = np.zeros(15, dtype=bool)
        board_rank_present[board_suit_ranks] = True

        # Exactly one card of this suit in hand
        hand_suit_counts = (hand_suits == suit).sum(axis=1)
        exactly_one = hand_suit_counts == 1
        if not exactly_one.any():
            continue

        # Determine the held rank for this suit per hand (use max since only one card has this suit)
        suit_ranks = np.where(hand_suits == suit, hand_ranks, -1)
        held_rank = suit_ranks.max(axis=1)

        # Compute available non-board ranks descending
        available = [r for r in range(2, 15) if not board_rank_present[r]]
        if len(available) == 0:
            continue
        # Highest non-board rank
        highest_avail = int(available[-1])
        # Next four ranks if available
        next_ranks = available[:-1][-3:][
            ::-1
        ]  # take up to 4 below highest, in descending order

        is_blocker = exactly_one
        is_best = is_blocker & (held_rank == highest_avail)
        is_high = is_blocker & np.isin(held_rank, next_ranks)

        blocker = np.where(is_blocker, 1, blocker)
        blocker_1 = np.where(is_best, 1, blocker_1)
        blocker_high = np.where(is_high, 1, blocker_high)

    return blocker, blocker_1, blocker_high


def feature_double_flush_draw(hand_suits: np.ndarray, board: Board) -> np.ndarray:
    """
    Double flush draw (turn only):
      - Board must have exactly 4 cards
      - Board must contain exactly two suits, each appearing twice (2-2 split)
      - Hand must contain exactly two cards of each of those two suits (2 and 2)
    Returns (N,) int8 mask
    """
    n = hand_suits.shape[0]
    if board.num_cards != 4:
        return np.zeros(n, dtype=np.int8)

    suit_counts = board.suit_counts  # length 4
    pair_suits = np.flatnonzero(suit_counts == 2)
    # Require exactly two suits with count==2
    if pair_suits.size != 2:
        return np.zeros(n, dtype=np.int8)

    s1, s2 = int(pair_suits[0]), int(pair_suits[1])

    hand_counts_s1 = (hand_suits == s1).sum(axis=1)
    hand_counts_s2 = (hand_suits == s2).sum(axis=1)
    mask = (hand_counts_s1 == 2) & (hand_counts_s2 == 2)
    return mask.astype(np.int8)


def _build_run_masks() -> np.ndarray:
    # 10 windows: starts 1..10 (1 represents Ace-low), highs are 5..14
    run_masks = np.empty(10, dtype=np.uint32)
    for i, start in enumerate(range(1, 11)):
        m = np.uint32(0)
        for k in range(start, start + 5):
            m |= np.uint32(1) << np.uint32(k)
        run_masks[i] = m
    return run_masks


def _popcount_u32(arr_u32: np.ndarray) -> np.ndarray:
    # Byte popcount via LUT; supports arrays of any shape
    lut256 = (
        np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
        .sum(axis=1)
        .astype(np.uint8)
    )
    bytes_view = arr_u32.view(np.uint8)
    reshaped = bytes_view.reshape(arr_u32.shape + (4,))
    return lut256[reshaped].sum(axis=-1).astype(np.uint8)


def straight_helper(
    counts_by_rank: np.ndarray, board_ranks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper for Omaha straight logic.

    Returns:
      - made_high: (N,) int8 -> ordinal among board possibilities: 1 = highest, 2 = second, ...; 0 if none
      - hand_sum_used: (N,) int16 -> sum of the two hand rank values used in the
        selected straight window (Ace counts as 1 when used low). 0 if no straight.
    """
    n = counts_by_rank.shape[0]
    if board_ranks.shape[0] < 3:
        return (np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int16))

    # Board unique-rank bitmask; include Ace-low bit at 1 if Ace present
    board_rank_counts = np.bincount(board_ranks, minlength=15)
    present_board_ranks = np.flatnonzero(board_rank_counts > 0)
    present_board_ranks = present_board_ranks[
        (present_board_ranks >= 2) & (present_board_ranks <= 14)
    ]
    if present_board_ranks.size:
        board_bits = np.bitwise_or.reduce(
            (np.uint32(1) << present_board_ranks.astype(np.uint32))
        ).astype(np.uint32)
    else:
        board_bits = np.uint32(0)
    if board_rank_counts[14] > 0:
        board_bits |= np.uint32(1) << np.uint32(1)

    # Hand unique-rank bitmasks; include Ace-low bit at 1 where applicable
    hand_present = counts_by_rank > 0
    bit_weights = np.uint32(1) << np.arange(15, dtype=np.uint32)
    hand_bits = (
        (hand_present.astype(np.uint32) * bit_weights).sum(axis=1).astype(np.uint32)
    )
    has_ace_in_hand = hand_present[:, 14]
    hand_bits |= has_ace_in_hand.astype(np.uint32) << np.uint32(1)

    # Windows and high ranks
    run_masks = _build_run_masks()  # (10,)
    run_highs = np.arange(10, dtype=np.int16) + 5  # 5..14

    # Board-only coverage counts per run
    board_in_run = np.uint32(board_bits) & run_masks
    board_counts = _popcount_u32(board_in_run)  # (10,)
    board_possible_mask = board_counts >= 3
    board_possible_highs = run_highs[board_possible_mask].astype(np.int8)
    board_highest_possible = (
        int(board_possible_highs.max()) if board_possible_highs.size else 0
    )

    # Per-hand windows that form a valid made straight
    hand_bits_2d = hand_bits[:, None]  # (N,1)
    run_masks_2d = run_masks[None, :]  # (1,10)
    union_covers = ((hand_bits_2d | board_bits) & run_masks_2d) == run_masks_2d
    hand_in_run = hand_bits_2d & run_masks_2d

    hand_counts = _popcount_u32(hand_in_run)
    # reuse board_counts with broadcast
    ok_windows = union_covers & (hand_counts >= 2) & (board_counts[None, :] >= 3)

    made_mask = ok_windows.any(axis=1)
    # Compute actual made high-rank per hand (rank value 5..14, else 0)
    candidate_highs = ok_windows.astype(np.int16) * run_highs[None, :]
    made_high_rank = candidate_highs.max(axis=1).astype(np.int8)
    made_high_rank[made_mask == 0] = 0

    # Map rank to ordinal vs board_possible_highs (desc)
    if board_possible_highs.size:
        highs_desc = np.sort(board_possible_highs)[::-1]
        rank_to_ord = np.zeros(15, dtype=np.int8)
        for idx, r in enumerate(highs_desc):
            rank_to_ord[int(r)] = np.int8(idx + 1)
        made_high = rank_to_ord[made_high_rank]
        made_high[made_high_rank == 0] = 0
    else:
        made_high = np.zeros_like(made_high_rank, dtype=np.int8)

    # print(f"made_mask: {made_mask}")
    # print(f"made_high: {made_high}")
    # print(f"board_possible_highs: {board_possible_highs}")
    # print(f"board_highest_possible: {board_highest_possible}")

    # Compute sum of the two hand ranks used in the selected window
    # Choose the window that achieved made_high_rank (highest run high)
    selected_window_idx = candidate_highs.argmax(axis=1)
    selected_hand_bits = hand_in_run[np.arange(n), selected_window_idx]
    selected_hand_bits = np.where(made_high_rank == 0, np.uint32(0), selected_hand_bits)

    # Extract rank positions 1..14 present in selected_hand_bits and sum the top 2
    positions = np.arange(1, 15, dtype=np.int16)
    bits_matrix = ((selected_hand_bits[:, None] >> positions) & 1).astype(np.int8)
    values = bits_matrix.astype(np.int16) * positions[None, :]
    if values.shape[1] >= 2:
        top2_idx = np.argpartition(values, -2, axis=1)[:, -2:]
        top2_vals = np.take_along_axis(values, top2_idx, axis=1)

        hand_sum_used = top2_vals.sum(axis=1).astype(np.int16)
    else:
        hand_sum_used = np.zeros(n, dtype=np.int16)
    hand_sum_used = np.where(made_high_rank == 0, 0, hand_sum_used).astype(np.int16)

    return (made_high, hand_sum_used)


def feature_straight_draw(
    counts_by_rank: np.ndarray,
    board_ranks: np.ndarray,
    base_made_high: np.ndarray | None = None,
    base_hand_sum_used: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Straight draw stats per hand using next-street simulations.
    Returns:
      - outs_count: how many next ranks make a straight (if base_made_high==0), or
        how many make another straight using higher hand ranks (if base_made_high>0)
      - draw_made_high: made-high ordinal summary per hand based on the same rule:
        base==0 → max over next_made_high_ord; base>0 → max over next_made_high_ord only where
        next_hand_sum_used > base_hand_sum_used; 0 if none
    """
    if base_made_high is None or base_hand_sum_used is None:
        base_made_high, base_hand_sum_used = straight_helper(
            counts_by_rank, board_ranks
        )
    n = counts_by_rank.shape[0]

    board_counts = np.bincount(board_ranks, minlength=15)
    candidate_ranks_list = [r for r in range(1, 15) if board_counts[r] == 0]
    candidate_count = len(candidate_ranks_list)
    if candidate_count == 0:
        return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)

    next_made_high_ord = np.zeros((n, candidate_count), dtype=np.int8)
    next_hand_sum_used = np.zeros((n, candidate_count), dtype=np.int16)

    for j, r in enumerate(candidate_ranks_list):
        next_board = np.concatenate(
            [board_ranks, np.array([r], dtype=board_ranks.dtype)]
        )
        next_made_high, next_hand_sum = straight_helper(counts_by_rank, next_board)
        next_made_high_ord[:, j] = next_made_high.astype(np.int8)
        next_hand_sum_used[:, j] = next_hand_sum.astype(np.int16)

    base_zero_mask = base_made_high == 0
    base_pos_mask = ~base_zero_mask

    outs_count = np.zeros(n, dtype=np.int8)
    draw_made_high = np.zeros(n, dtype=np.int8)

    if base_zero_mask.any():
        outs_zero = (next_made_high_ord[base_zero_mask] > 0).sum(axis=1).astype(np.int8)
        made_high_zero = next_made_high_ord[base_zero_mask].max(axis=1).astype(np.int8)
        outs_count[base_zero_mask] = outs_zero
        draw_made_high[base_zero_mask] = made_high_zero

    if base_pos_mask.any():
        # Improvements are where next uses higher hand rank sums than base
        # Also ensure next_made_high_ord > 0
        improve_mask = (
            next_hand_sum_used[base_pos_mask]
            > base_hand_sum_used[base_pos_mask][:, None]
        ) & (next_made_high_ord[base_pos_mask] > 0)
        outs_pos = improve_mask.sum(axis=1).astype(np.int8)
        # Max of next_made_high_ord only at improve positions; zero if none
        masked_highs = np.where(improve_mask, next_made_high_ord[base_pos_mask], 0)
        made_high_pos = masked_highs.max(axis=1).astype(np.int8)
        outs_count[base_pos_mask] = outs_pos
        draw_made_high[base_pos_mask] = made_high_pos

    return outs_count, draw_made_high


"""trips top
lrr,hrr,
"""
