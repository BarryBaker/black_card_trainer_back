from db import PokiDatabase

from features.board import Board
from features.encoding import encode_hands
from features.extractor import extract_features
from trees.best_feature import get_best_single_feature
from typing import Dict, Iterable, List, Optional, Sequence, Mapping, Any, Tuple

import numpy as np
import pandas as pd

db = PokiDatabase()
board, data, stack, pos, pot, line = db.get_poki_nth_board_and_data(20)
print("Board", board, "Stack", stack, "Pos", pos, "Pot", pot, "Line", line)

features, actions = extract_features(board, data)
best_feature = get_best_single_feature(features, actions)
print("Best feature:", best_feature.feature_name)
