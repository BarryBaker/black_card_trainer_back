import sys
from pathlib import Path

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from db import PokiDatabase
import time
from features.board import Board
from features.encoding import encode_hands
from features.extractor import extract_features
from trees import best_feature
from trees.best_feature import (
    get_best_single_feature,
    get_overall_action_ratios,
    rank_single_features,
)
from typing import Dict, Iterable, List, Optional, Sequence, Mapping, Any, Tuple

import numpy as np
import pandas as pd
from pprint import pprint

total_mass = 0
result = []

# BOARD_DUMP_PATH = Path(__file__).resolve().parent / "board_dump.txt"

# with BOARD_DUMP_PATH.open("w", encoding="utf-8") as board_dump_file:
#     for i in range(1, 3000):
#         board, data, stack, pos, pot, line = db.get_poki_nth_board_and_data(i)
#         print(i, board, stack, pos, pot, line, file=board_dump_file)

# the_tree = {}


def split(features, actions, depth=0):

    features = features.loc[:, (features.nunique() > 1)]
    mass = features.shape[0] / total_mass
    overall_action_ratios = get_overall_action_ratios(actions)
    if features.empty:
        # print("Features dataframe is empty, stopping split.")

        try:
            taskhandhasit = features.loc["KhKc5s4s", :]

        except KeyError:
            taskhandhasit = None
        # if taskhandhasit is not None:
        #     print("Task hand has it in empty features case.")
        return {
            "mass": mass,
            "actions": overall_action_ratios,
            "split": False,
            "best_feature": None,
            "cards": actions,
            "taskhand": taskhandhasit is not None,
        }
    (
        best_score,
        true_ratios_str,
        false_ratios_str,
    ) = get_best_single_feature(features, actions)
    mass = features.shape[0] / total_mass

    best_overall_action = max(overall_action_ratios, key=overall_action_ratios.get)
    best_true_action = max(true_ratios_str, key=true_ratios_str.get)
    best_false_action = max(false_ratios_str, key=false_ratios_str.get)

    actions_split = (
        best_overall_action != best_true_action
        or best_true_action != best_false_action
        or best_overall_action != best_false_action
    )

    if mass < 0.05 and not actions_split:
        # print("Mass below threshold, stopping split.")

        try:
            taskhandhasit = features.loc["KhKc5s4s", :]

        except KeyError:
            taskhandhasit = None
        # if taskhandhasit is not None:
        #     print("Task hand has it in low mass case.")
        return {
            "mass": mass,
            "actions": overall_action_ratios,
            "split": False,
            "best_feature": None,
            "cards": actions,
            "taskhand": taskhandhasit is not None,
        }

    # Split both dataframes with the same boolean mask from the best feature column.
    best_feature_mask = features[best_score.feature_name].astype(bool)

    try:
        taskhandhasit = features.loc["KhKc5s4s", :]  # [best_score.feature_name]
        # print(
        #     "Task hand has best feature:",
        #     taskhandhasit,
        #     best_score.feature_name,
        # )
    except KeyError:
        taskhandhasit = None
    # if taskhandhasit:
    #     print("Task hand has it in best feature case:", best_score.feature_name)
    features_true = features.loc[best_feature_mask].copy()
    features_false = features.loc[~best_feature_mask].copy()

    actions_true = actions.loc[best_feature_mask].copy()
    actions_false = actions.loc[~best_feature_mask].copy()
    if features_true.shape[0] == 0 or features_false.shape[0] == 0:
        return {
            "mass": mass,
            "actions": overall_action_ratios,
            "split": False,
            "best_feature": None,
            "cards": actions,
            "taskhand": taskhandhasit is not None,
        }
    # if best_score.feature_name == "two_pair":
    #     try:
    #         taskhandhasit = bool(features.loc["KhKc5s4s", :][best_score.feature_name])
    #         print(
    #             "Task hand has best feature:",
    #             taskhandhasit,
    #             best_score.feature_name,
    #         )
    #     except KeyError:
    #         print("Task hand does not have best feature:", best_score.feature_name)
    #     print(
    #         "tttwpppp",
    #         {
    #             "mass": mass,
    #             "actions": overall_action_ratios,
    #             "best_feature": best_score.feature_name,
    #             "split": actions_split,
    #             "taskhand": taskhandhasit,
    #             "cards": "KhKc5s4s" in features.index.tolist(),
    #         },
    #     )
    return {
        "mass": mass,
        "actions": overall_action_ratios,
        "best_feature": best_score.feature_name,
        "split": actions_split,
        "true": split(
            features_true,
            actions_true,
        ),
        "false": split(
            features_false,
            actions_false,
        ),
        "cards": actions,
        "taskhand": taskhandhasit is not None,
    }


def find_last_false_subtree(subTree):
    if subTree["false"]["best_feature"] is None:

        return subTree["false"]
    return find_last_false_subtree(subTree["false"])


def is_tree_splits(subTree):
    if subTree["split"] is False:
        return (
            "true" in subTree
            and is_tree_splits(subTree["true"])
            or "false" in subTree
            and is_tree_splits(subTree["false"])
        )
    return True


# def build_result(tree):
#     def print_tree(subTree, indentation=0):
#         if not is_tree_splits(subTree):
#             return
#         true_not_splits = subTree["true"]["best_feature"] is None
#         if true_not_splits:
#             action_to_print = subTree["true"]["actions"]

#             cards = (
#                 subTree["true"]["cards"]
#                 .sample(n=min(7, len(subTree["true"]["cards"])))
#                 .to_dict("index")
#             )
#             taskhand = subTree["true"]["taskhand"]
#         else:
#             last_false = find_last_false_subtree(subTree["true"])
#             action_to_print = last_false["actions"]
#             cards = (
#                 last_false["cards"]
#                 .sample(n=min(7, len(last_false["cards"])))
#                 .to_dict("index")
#             )
#             taskhand = last_false["taskhand"]
#         result.append(
#             {
#                 "indentation": indentation,
#                 "best_feature": subTree["best_feature"],
#                 "action": action_to_print,
#                 "cards": cards,
#                 "taskhand": taskhand,
#             }
#         )

#         if subTree["true"]["best_feature"] is not None:
#             print_tree(subTree["true"], indentation + 1)
#         if subTree["false"]["best_feature"] is not None:
#             print_tree(subTree["false"], indentation)

#     print_tree(tree)
#     last_false = find_last_false_subtree(tree["false"])
#     action_to_print = last_false["actions"]
#     cards = (
#         last_false["cards"].sample(n=min(7, len(last_false["cards"]))).to_dict("index")
#     )
#     result.append(
#         {
#             "indentation": 0,
#             "best_feature": "rest",
#             "action": action_to_print,
#             "cards": cards,
#             "taskhand": last_false["taskhand"],
#         }
#     )


def build_result(subTree):

    subresult = []
    if not is_tree_splits(subTree):
        return []
    true_splits = is_tree_splits(subTree["true"])
    false_splits = is_tree_splits(subTree["false"])

    subresult.append(
        {
            "action": subTree["true"]["actions"],
            "cards": (
                (subTree["true"]["cards"].head(3).index.tolist())
                if not true_splits
                else []
            ),
            "taskhand": subTree["true"]["taskhand"] if not true_splits else "",
            "best_feature": subTree["best_feature"],
            "mass": subTree["true"]["mass"],
            "subtree": (
                build_result(subTree["true"])
                if subTree["true"]["best_feature"] is not None
                else []
            ),
        }
    )

    if not false_splits:

        subresult.append(
            {
                "action": subTree["false"]["actions"],
                "cards": (subTree["false"]["cards"].head(3).index.tolist()),
                "taskhand": subTree["false"]["taskhand"],
                "best_feature": "rest",
                "mass": subTree["false"]["mass"],
                "subtree": [],
            }
        )

    if subTree["false"]["best_feature"] is not None:
        subresult = subresult + build_result(subTree["false"])
        # subresult.append(
        #     {
        #         "action": subTree["false"]["actions"],
        #         "cards": (
        #             subTree["false"]["cards"]
        #             .sample(n=min(7, len(subTree["false"]["cards"])))
        #             .index.tolist()
        #         ),
        #         "taskhand": subTree["false"]["taskhand"],
        #         "best_feature": subTree["false"]["best_feature"],
        #         "mass": subTree["false"]["mass"],
        #         "subtree": (
        #             build_result(subTree["false"])
        #             if subTree["false"]["best_feature"] is not None
        #             else []
        #         ),
        #     }
        # )
    return subresult


# last_false = find_last_false_subtree(tree["false"])
# action_to_print = max(last_false["actions"], key=last_false["actions"].get)
# cards = (
#     last_false["cards"].sample(n=min(7, len(last_false["cards"]))).index.tolist()
# )
# tasnkhand = last_false["taskhand"]
# print(
#     ("rest" + ": " + f"\033[91m{action_to_print}\033[0m" + f"{tasnkhand}").ljust(55)
#     + f" {cards}"
# )


def main():
    global total_mass

    db = PokiDatabase()
    time1 = time.time()
    board, data, stack, pos, pot, line = db.get_poki_nth_board_and_data(70199)

    print()
    print(
        "------ Board",
        board,
        "| Stack",
        stack,
        "| Position",
        pos,
        "| Pot",
        pot,
        "| Line",
        line,
    )

    features, actions = extract_features(board, data)

    total_mass = actions.shape[0]

    tree = split(features, actions)
    print(tree["actions"])
    # pprint(tree)

    # build_result(tree)

    time2 = time.time()
    print(f"Result built in {time2 - time1:.2f} seconds")
    result = build_result(tree)
    for item in result:
        print(item)
        print()


if __name__ == "__main__":
    main()
