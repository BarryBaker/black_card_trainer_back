from __future__ import annotations

import pandas as pd

from trees.best_feature import (
    normalize_actions,
    rank_single_features,
    score_single_feature,
)


def test_normalize_actions_scales_each_row_independently():
    actions = pd.DataFrame(
        {
            "C": [51.0, 0.0, 20.0],
            "R50": [0.0, 0.0, 80.0],
        },
        index=["a", "b", "c"],
    )

    normalized, weights = normalize_actions(actions)

    assert normalized.loc["a"].tolist() == [1.0, 0.0]
    assert normalized.loc["b"].tolist() == [0.0, 0.0]
    assert normalized.loc["c"].tolist() == [0.2, 0.8]
    assert weights.loc["a"] == 0.51
    assert weights.loc["b"] == 0.0
    assert weights.loc["c"] == 1.0


def test_score_single_feature_prefers_explanatory_split():
    features = pd.DataFrame(
        {
            "value": [1, 1, 0, 0],
            "blocker": [1, 0, 1, 0],
        },
        index=["h1", "h2", "h3", "h4"],
    )
    actions = pd.DataFrame(
        {
            "C": [10.0, 20.0, 90.0, 80.0],
            "R50": [90.0, 80.0, 10.0, 20.0],
        },
        index=features.index,
    )

    value_score = score_single_feature(features, actions, "value")
    blocker_score = score_single_feature(features, actions, "blocker")

    assert value_score.improvement > blocker_score.improvement
    assert {k: round(v, 6) for k, v in value_score.true_prediction.items()} == {
        "C": 0.15,
        "R50": 0.85,
    }
    assert {k: round(v, 6) for k, v in value_score.false_prediction.items()} == {
        "C": 0.85,
        "R50": 0.15,
    }
    assert value_score.true_action_totals == {"C": 30.0, "R50": 170.0}
    assert value_score.false_action_totals == {"C": 170.0, "R50": 30.0}
    assert round(value_score.true_total_mass, 6) == 200.0
    assert round(value_score.false_total_mass, 6) == 200.0


def test_rank_single_features_returns_best_feature_first():
    features = pd.DataFrame(
        {
            "made_hand": [1, 1, 0, 0, 0, 0],
            "backdoor": [1, 0, 1, 0, 1, 0],
            "always_on": [1, 1, 1, 1, 1, 1],
        },
        index=[f"h{i}" for i in range(6)],
    )
    actions = pd.DataFrame(
        {
            "C": [5.0, 10.0, 85.0, 90.0, 80.0, 95.0],
            "R50": [95.0, 90.0, 15.0, 10.0, 20.0, 5.0],
        },
        index=features.index,
    )

    scores = rank_single_features(features, actions)

    assert scores[0].feature_name == "made_hand"
    assert scores[-1].feature_name == "always_on"
    assert scores[-1].improvement == 0.0


def test_leaf_argmax_builds_deterministic_leaf_policy():
    features = pd.DataFrame(
        {"aggressive": [1, 1, 0, 0]}, index=["h1", "h2", "h3", "h4"]
    )
    actions = pd.DataFrame(
        {
            "C": [20.0, 35.0, 80.0, 90.0],
            "R50": [80.0, 65.0, 20.0, 10.0],
        },
        index=features.index,
    )

    score = score_single_feature(
        features,
        actions,
        "aggressive",
        prediction_mode="leaf_argmax",
        loss_metric="l1",
    )

    assert score.true_prediction == {"C": 0.0, "R50": 1.0}
    assert score.false_prediction == {"C": 1.0, "R50": 0.0}
    assert score.improvement > 0.0


def test_partial_frequency_rows_have_lower_weight_in_loss():
    features = pd.DataFrame({"feature": [1, 1, 0, 0]}, index=["h1", "h2", "h3", "h4"])
    actions = pd.DataFrame(
        {
            "C": [100.0, 0.0, 100.0, 30.0],
            "R50": [0.0, 100.0, 0.0, 20.0],
        },
        index=features.index,
    )

    weighted_score = score_single_feature(features, actions, "feature")
    unweighted_score = score_single_feature(
        features,
        actions,
        "feature",
        normalize_action_rows=False,
    )

    assert round(weighted_score.false_prediction["C"], 6) == round(13.0 / 15.0, 6)
    assert round(weighted_score.false_prediction["R50"], 6) == round(2.0 / 15.0, 6)
    assert weighted_score.loss < unweighted_score.loss
