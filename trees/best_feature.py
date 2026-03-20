from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Literal, Sequence
import time
import numpy as np
import pandas as pd

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db import PokiDatabase
from features.extractor import extract_features

LossMetric = Literal["l1", "l2"]
PredictionMode = Literal["leaf_average", "leaf_argmax"]


@dataclass(frozen=True)
class FeatureScore:
    feature_name: str
    loss: float
    baseline_loss: float
    improvement: float
    true_count: int
    false_count: int
    true_total_mass: float
    false_total_mass: float
    true_action_totals: Dict[str, float]
    false_action_totals: Dict[str, float]
    true_prediction: Dict[str, float]
    false_prediction: Dict[str, float]


def _group_action_summary(
    raw_actions: pd.DataFrame,
) -> tuple[float, Dict[str, float], Dict[str, float]]:
    action_totals = raw_actions.astype(np.float64).clip(lower=0.0).sum(axis=0)
    total_mass = float(action_totals.sum())
    totals_dict = {column: float(value) for column, value in action_totals.items()}
    if total_mass > 0:
        ratios_dict = {
            column: float(value / total_mass) for column, value in action_totals.items()
        }
    else:
        ratios_dict = {column: 0.0 for column in raw_actions.columns}
    return total_mass, totals_dict, ratios_dict


def normalize_actions(actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Normalize each row and return row weights from the original totals.

    The normalized rows describe the action mix conditional on reaching the spot.
    The returned weights keep the original row totals meaningful, so a row with
    total 50 counts half as much as a row with total 100 in both leaf fitting and
    loss calculation.
    """
    if actions.empty:
        raise ValueError("actions dataframe must contain at least one action column")

    action_values = actions.astype(np.float64).clip(lower=0.0)
    row_sums = action_values.sum(axis=1)
    normalized = action_values.copy()
    nonzero_rows = row_sums > 0
    normalized.loc[nonzero_rows] = normalized.loc[nonzero_rows].div(
        row_sums.loc[nonzero_rows], axis=0
    )
    row_weights = row_sums / 100.0
    return normalized, row_weights


def _validate_inputs(
    features: pd.DataFrame, actions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty:
        raise ValueError("features dataframe must not be empty")
    if actions.empty:
        raise ValueError("actions dataframe must not be empty")

    aligned_features, aligned_actions = features.align(actions, join="inner", axis=0)
    if aligned_features.empty:
        raise ValueError("features and actions do not share any index values")

    return aligned_features, aligned_actions


def _prediction_for_group(
    action_group: np.ndarray,
    row_weights: np.ndarray,
    prediction_mode: PredictionMode,
) -> np.ndarray:
    if action_group.shape[0] == 0:
        raise ValueError("cannot build a prediction from an empty group")
    if row_weights.shape[0] != action_group.shape[0]:
        raise ValueError("row_weights length must match action_group length")

    total_weight = float(row_weights.sum())
    if total_weight > 0:
        mean_prediction = np.average(action_group, axis=0, weights=row_weights)
    else:
        mean_prediction = action_group.mean(axis=0)
    if prediction_mode == "leaf_average":
        return mean_prediction
    if prediction_mode == "leaf_argmax":
        hard_prediction = np.zeros_like(mean_prediction)
        hard_prediction[int(mean_prediction.argmax())] = 1.0
        return hard_prediction
    raise ValueError(f"unsupported prediction_mode: {prediction_mode}")


def _loss(
    actual: np.ndarray,
    prediction: np.ndarray,
    row_weights: np.ndarray,
    loss_metric: LossMetric,
) -> float:
    diffs = actual - prediction
    if loss_metric == "l2":
        row_losses = np.square(diffs).sum(axis=1)
        return float((row_losses * row_weights).sum())
    if loss_metric == "l1":
        row_losses = np.abs(diffs).sum(axis=1)
        return float((row_losses * row_weights).sum())
    raise ValueError(f"unsupported loss_metric: {loss_metric}")


def score_single_feature(
    features: pd.DataFrame,
    actions: pd.DataFrame,
    feature_name: str,
    *,
    loss_metric: LossMetric = "l2",
    prediction_mode: PredictionMode = "leaf_average",
    normalize_action_rows: bool = True,
) -> FeatureScore:

    aligned_features, aligned_actions = _validate_inputs(features, actions)
    if feature_name not in aligned_features.columns:
        raise KeyError(f"unknown feature column: {feature_name}")

    raw_actions = aligned_actions.astype(np.float64).clip(lower=0.0)

    if normalize_action_rows:
        prepared_actions, row_weights = normalize_actions(aligned_actions)
    else:
        prepared_actions = aligned_actions.astype(np.float64)
        row_weights = pd.Series(1.0, index=prepared_actions.index, dtype=np.float64)

    action_matrix = prepared_actions.to_numpy(dtype=np.float64)
    row_weight_array = row_weights.to_numpy(dtype=np.float64)
    feature_mask = aligned_features[feature_name].fillna(0).astype(bool).to_numpy()

    baseline_prediction = _prediction_for_group(
        action_matrix, row_weight_array, prediction_mode
    )
    baseline_loss = _loss(
        action_matrix, baseline_prediction, row_weight_array, loss_metric
    )

    true_count = int(feature_mask.sum())
    false_count = int((~feature_mask).sum())
    true_total_mass, true_action_totals, true_action_ratios = _group_action_summary(
        raw_actions.loc[feature_mask]
    )
    false_total_mass, false_action_totals, false_action_ratios = _group_action_summary(
        raw_actions.loc[~feature_mask]
    )

    if true_count == 0 or false_count == 0:
        constant_prediction = {
            column: float(value)
            for column, value in zip(prepared_actions.columns, baseline_prediction)
        }
        return FeatureScore(
            feature_name=feature_name,
            loss=baseline_loss,
            baseline_loss=baseline_loss,
            improvement=0.0,
            true_count=true_count,
            false_count=false_count,
            true_total_mass=true_total_mass,
            false_total_mass=false_total_mass,
            true_action_totals=true_action_totals,
            false_action_totals=false_action_totals,
            true_prediction=constant_prediction,
            false_prediction=constant_prediction,
        )

    true_prediction = _prediction_for_group(
        action_matrix[feature_mask],
        row_weight_array[feature_mask],
        prediction_mode,
    )
    false_prediction = _prediction_for_group(
        action_matrix[~feature_mask],
        row_weight_array[~feature_mask],
        prediction_mode,
    )
    split_loss = _loss(
        action_matrix[feature_mask],
        true_prediction,
        row_weight_array[feature_mask],
        loss_metric,
    )
    split_loss += _loss(
        action_matrix[~feature_mask],
        false_prediction,
        row_weight_array[~feature_mask],
        loss_metric,
    )

    return FeatureScore(
        feature_name=feature_name,
        loss=split_loss,
        baseline_loss=baseline_loss,
        improvement=baseline_loss - split_loss,
        true_count=true_count,
        false_count=false_count,
        true_total_mass=true_total_mass,
        false_total_mass=false_total_mass,
        true_action_totals=true_action_totals,
        false_action_totals=false_action_totals,
        true_prediction={
            column: float(value)
            for column, value in zip(prepared_actions.columns, true_prediction)
        },
        false_prediction={
            column: float(value)
            for column, value in zip(prepared_actions.columns, false_prediction)
        },
    )


def rank_single_features(
    features: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    loss_metric: LossMetric = "l2",
    prediction_mode: PredictionMode = "leaf_average",
    normalize_action_rows: bool = True,
    feature_names: Sequence[str] | None = None,
) -> list[FeatureScore]:

    aligned_features, aligned_actions = _validate_inputs(features, actions)
    columns = (
        list(feature_names)
        if feature_names is not None
        else list(aligned_features.columns)
    )

    scores = [
        score_single_feature(
            aligned_features,
            aligned_actions,
            feature_name,
            loss_metric=loss_metric,
            prediction_mode=prediction_mode,
            normalize_action_rows=normalize_action_rows,
        )
        for feature_name in columns
    ]
    return sorted(
        scores, key=lambda score: (-score.improvement, score.loss, score.feature_name)
    )


def get_overall_action_ratios(actions: pd.DataFrame) -> Dict[str, float]:
    """Return overall action ratios aggregated across all rows."""
    _, _, overall_action_ratios = _group_action_summary(actions)
    return overall_action_ratios


def get_best_single_feature(
    features: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    all_feature_names: Sequence[str] | None = None,
    level: int | None = None,
    loss_metric: LossMetric = "l2",
    prediction_mode: PredictionMode = "leaf_average",
    normalize_action_rows: bool = True,
    feature_names: Sequence[str] | None = None,
) -> FeatureScore:
    if "Full" in all_feature_names and level == 0:
        for f in ["Full", "Flush", "Straight", "Trips"]:
            if f in features.columns:
                features = features[[f]]
                break
    elif "Flush" in all_feature_names and level == 0:
        for f in ["Flush", "Flushblocker", "Straight", "Trips"]:
            if f in features.columns:
                features = features[[f]]
                break
    elif "Straight" in all_feature_names and level == 0:
        for f in ["Straight", "Trips"]:
            if f in features.columns:
                features = features[[f]]
                break
    elif "Flushdraw" in all_feature_names and level == 0:
        for f in ["Trips", "Top Two Pair", "Wrap", "Flushdraw", "Flushblocker"]:
            if f in features.columns:
                features = features[[f]]
                break
    elif level == 0:
        for f in ["Trips", "Top Two Pair", "Wrap"]:
            if f in features.columns:
                features = features[[f]]
                break

    # Exclude columns named as string representations of numbers like '2', '3', ..., '22', '33', etc.
    import re

    features_no_ranks = features[
        [
            col
            for col in features.columns
            if not col
            in [
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "T",
                "J",
                "Q",
                "K",
                "A",
                "22",
                "33",
                "44",
                "55",
                "66",
                "77",
                "88",
                "99",
                "TT",
                "JJ",
                "QQ",
                "KK",
                "AA",
            ]
        ]
    ]
    if not features_no_ranks.empty:
        features = features_no_ranks

    aligned_features, aligned_actions = _validate_inputs(features, actions)
    columns = (
        list(feature_names)
        if feature_names is not None
        else list(aligned_features.columns)
    )
    if not columns:
        raise ValueError("no features to score")

    # --- Prepare actions and weights ONCE ---
    raw_actions = aligned_actions.astype(np.float64).clip(lower=0.0)
    if normalize_action_rows:
        prepared_actions, row_weights = normalize_actions(aligned_actions)
    else:
        prepared_actions = aligned_actions.astype(np.float64)
        row_weights = pd.Series(1.0, index=prepared_actions.index, dtype=np.float64)

    A = prepared_actions.to_numpy(dtype=np.float64)  # (N, K)
    w = row_weights.to_numpy(dtype=np.float64)  # (N,)
    action_cols = list(prepared_actions.columns)

    # --- Build boolean feature matrix (N, M) ---
    F = aligned_features[columns].fillna(0).to_numpy(dtype=np.float64)
    F = (F != 0).astype(np.float64)
    # if first == 1:
    #     print(F.shape, A.shape, w.shape)
    # --- Vectorised weighted-sum-of-squares split search ---
    wA = A * w[:, None]  # (N, K)
    wA2 = (A**2) * w[:, None]  # (N, K)

    W = float(w.sum())
    S = wA.sum(axis=0)  # (K,)
    Q = wA2.sum(axis=0)  # (K,)

    W_true = F.T @ w  # (M,)
    S_true = F.T @ wA  # (M, K)
    Q_true = F.T @ wA2  # (M, K)

    W_false = W - W_true  # (M,)
    S_false = S[None, :] - S_true  # (M, K)
    Q_false = Q[None, :] - Q_true  # (M, K)

    # WSS = Q - S^2 / W  (per group, per action column)
    with np.errstate(divide="ignore", invalid="ignore"):
        wss_true = Q_true - S_true**2 / np.where(
            W_true[:, None] > 0, W_true[:, None], 1.0
        )
        wss_false = Q_false - S_false**2 / np.where(
            W_false[:, None] > 0, W_false[:, None], 1.0
        )

    wss_true = np.where(W_true[:, None] > 0, wss_true, 0.0)
    wss_false = np.where(W_false[:, None] > 0, wss_false, 0.0)

    split_losses = wss_true.sum(axis=1) + wss_false.sum(axis=1)  # (M,)
    baseline_loss = float((Q - S**2 / W).sum())

    improvements = baseline_loss - split_losses

    # Degenerate features (all-true or all-false) cannot improve
    true_counts = F.sum(axis=0).astype(int)
    N = F.shape[0]
    degenerate = (true_counts == 0) | (true_counts == N)
    improvements[degenerate] = -1.0

    best_idx = int(np.argmax(improvements))
    best_name = columns[best_idx]

    # --- Build detailed result for the winner only ---
    feature_mask = F[:, best_idx].astype(bool)
    tc = int(true_counts[best_idx])
    fc = N - tc

    true_total_mass, true_action_totals, _ = _group_action_summary(
        raw_actions.loc[feature_mask]
    )
    false_total_mass, false_action_totals, _ = _group_action_summary(
        raw_actions.loc[~feature_mask]
    )

    if W_true[best_idx] > 0:
        true_pred = S_true[best_idx] / W_true[best_idx]
    else:
        true_pred = A.mean(axis=0)
    if W_false[best_idx] > 0:
        false_pred = S_false[best_idx] / W_false[best_idx]
    else:
        false_pred = A.mean(axis=0)

    if prediction_mode == "leaf_argmax":
        tp = np.zeros_like(true_pred)
        tp[int(true_pred.argmax())] = 1.0
        true_pred = tp
        fp = np.zeros_like(false_pred)
        fp[int(false_pred.argmax())] = 1.0
        false_pred = fp

    true_prediction = {c: float(v) for c, v in zip(action_cols, true_pred)}
    false_prediction = {c: float(v) for c, v in zip(action_cols, false_pred)}

    best_score = FeatureScore(
        feature_name=best_name,
        loss=float(split_losses[best_idx]),
        baseline_loss=baseline_loss,
        improvement=float(improvements[best_idx]),
        true_count=tc,
        false_count=fc,
        true_total_mass=true_total_mass,
        false_total_mass=false_total_mass,
        true_action_totals=true_action_totals,
        false_action_totals=false_action_totals,
        true_prediction=true_prediction,
        false_prediction=false_prediction,
    )

    true_ratios_str = {k: f"{v:.2f}" for k, v in true_prediction.items()}
    false_ratios_str = {k: f"{v:.2f}" for k, v in false_prediction.items()}

    return (
        best_score,
        true_ratios_str,
        false_ratios_str,
    )


def main(board_number: int = 20, top_n: int = 3) -> None:
    db = PokiDatabase()
    board, data, stack, pos, pot, line = db.get_poki_nth_board_and_data(board_number)
    if board is None or data is None:
        raise ValueError(f"no board data found for row {board_number}")

    features, actions = extract_features(board, data)
    scores = rank_single_features(features, actions)

    print(f"Board {board}")
    print(f"Stack: {stack} | Position: {pos} | Pot: {pot} | Line: {line}")
    print(
        f"Hands: {len(features)} | Features: {features.shape[1]} | Actions: {actions.shape[1]}"
    )

    overall_total_mass, overall_action_totals, overall_action_ratios = (
        _group_action_summary(actions)
    )
    overall_ratios_str = {k: f"{v:.2f}" for k, v in overall_action_ratios.items()}
    print(f"Overall action_ratios={overall_ratios_str}")

    print("Top single-feature splits (L2 loss on normalized action rows):")
    for rank, score in enumerate(scores[:top_n], start=1):
        print(
            f"{rank:>2}. {score.feature_name:<18} improvement={score.improvement:.2f} "
            f"loss={score.loss:.2f} true={score.true_count} false={score.false_count}"
        )
        true_ratios_str = {k: f"{v:.2f}" for k, v in score.true_prediction.items()}
        false_ratios_str = {k: f"{v:.2f}" for k, v in score.false_prediction.items()}
        print(f"    feature=1 action_ratios={true_ratios_str}")
        print(f"    feature=0 action_ratios={false_ratios_str}")


if __name__ == "__main__":
    main()
