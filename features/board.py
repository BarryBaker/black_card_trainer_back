from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .encoding import encode_board


Street = Literal["preflop", "flop", "turn", "river"]


@dataclass
class Board:
    raw: str

    def __str__(self) -> str:
        return self.raw

    def __repr__(self) -> str:
        cards = self.raw.replace(" ", "")
        return " ".join(cards[i : i + 2] for i in range(0, len(cards), 2))

    def __post_init__(self):
        ranks, suits = encode_board(self.raw)
        self.ranks: np.ndarray = ranks
        self.suits: np.ndarray = suits
        self.num_cards: int = int(ranks.shape[0])
        self.street: Street = self._infer_street(self.num_cards)

        # Counts
        self.rank_counts: np.ndarray = np.bincount(self.ranks, minlength=15).astype(
            np.int8
        )
        self.suit_counts: np.ndarray = np.bincount(self.suits, minlength=4).astype(
            np.int8
        )

        # Rank structure
        self.is_unpaired: bool = bool((self.rank_counts == 1).any())
        self.is_paired: bool = bool(
            (self.rank_counts == 2).sum() == 1 and not (self.rank_counts >= 3).any()
        )
        self.is_double_paired: bool = bool(
            (self.rank_counts == 2).sum() == 2 and not (self.rank_counts >= 3).any()
        )
        self.has_trips_on_board: bool = bool((self.rank_counts == 3).any())
        self.has_quads_on_board: bool = bool((self.rank_counts == 4).any())

        # Suit structure
        self.has_two_suited: bool = bool((self.suit_counts >= 2).any())
        self.has_three_suited: bool = bool((self.suit_counts >= 3).any())
        self.four_flush_board: bool = bool((self.suit_counts >= 4).any())

        self.is_monotone: bool = bool(
            (self.suit_counts >= 3).any() and (self.suit_counts > 0).sum() == 1
        )
        self.is_two_tone: bool = bool(
            (self.suit_counts > 0).sum() == 2 and not self.is_monotone
        )

        # Convenience
        self.highest_rank: int = int(self.ranks.max()) if self.num_cards > 0 else 0
        self.second_highest_rank: int = (
            int(np.partition(self.ranks, -2)[-2]) if self.num_cards >= 2 else 0
        )

    @staticmethod
    def _infer_street(num_board_cards: int) -> Street:
        if num_board_cards <= 0:
            return "preflop"
        if num_board_cards == 3:
            return "flop"
        if num_board_cards == 4:
            return "turn"
        if num_board_cards >= 5:
            return "river"
        return "preflop"
