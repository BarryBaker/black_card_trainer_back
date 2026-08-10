from __future__ import annotations

from decimal import Decimal
import random

from flask import Blueprint, current_app, jsonify, g, request
from psycopg2.extras import RealDictCursor
from psycopg2 import sql
from db import PokiDatabase
from features.extractor import extract_features
from trees.best_feature import (
    get_best_single_feature,
    get_overall_action_ratios,
    rank_single_features,
)

db_blueprint = Blueprint("db", __name__)


def get_database() -> PokiDatabase:
    database = g.get("poki_database")
    if database is None:
        database = PokiDatabase()
        g.poki_database = database
    return database


@db_blueprint.teardown_app_request
def clear_database_reference(_exception):
    g.pop("poki_database", None)


def get_required_payload_fields(payload, required_fields):
    missing_fields = [field for field in required_fields if field not in payload]
    return missing_fields


def get_street_filter_condition(street):
    """Convert street value to SQL condition for is_flop/is_turn columns"""
    street_lower = street.lower() if isinstance(street, str) else str(street).lower()
    if street_lower == "flop":
        return "is_flop", True
    elif street_lower == "turn":
        return "is_turn", True
    else:
        raise ValueError(f"Invalid street value: {street}. Must be 'flop' or 'turn'")


FLOP_BOARD_FILTER_COLUMNS = (
    ("flopSuited", "is_flopsuited"),
    ("flopPaired", "is_floppaired"),
    ("flopStraight", "is_flopstr8"),
    ("flopFlush", "is_flopflush"),
)

TURN_BOARD_FILTER_COLUMNS = (
    ("turnSuited", "is_suited"),
    ("turnPaired", "is_paired"),
    ("turnStraight", "is_str8"),
    ("turnFlush", "is_flush"),
)


def get_board_filter_columns(street):
    street_lower = street.lower() if isinstance(street, str) else str(street).lower()
    if street_lower == "flop":
        return FLOP_BOARD_FILTER_COLUMNS
    if street_lower == "turn":
        return FLOP_BOARD_FILTER_COLUMNS + TURN_BOARD_FILTER_COLUMNS
    raise ValueError(f"Invalid street value: {street}. Must be 'flop' or 'turn'")


def get_board_filter_clauses(street, board_filters):
    if not isinstance(board_filters, dict):
        raise ValueError("Field 'board' must be an object")

    clauses = []
    params = []
    for payload_key, column_name in get_board_filter_columns(street):
        filter_value = board_filters.get(payload_key)
        if filter_value is None:
            continue
        if not isinstance(filter_value, bool):
            raise ValueError(f"Board field '{payload_key}' must be boolean or null")

        clauses.append(sql.SQL("and {} = %s").format(sql.Identifier(column_name)))
        params.append(filter_value)

    return clauses, params


def normalize_json_value(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    return value


def get_data_item_weight(item_value):
    if not isinstance(item_value, dict):
        return 0

    weight = 0
    for value in item_value.values():
        if isinstance(value, (int, float, Decimal)):
            weight += float(value)
    return weight


def select_weighted_task(data):
    if not isinstance(data, dict) or not data:
        return None

    weighted_items = []
    for key, value in data.items():
        item_weight = get_data_item_weight(value)
        if item_weight > 0:
            weighted_items.append((key, value, item_weight))

    if not weighted_items:
        return None

    selected_key, selected_value, _ = random.choices(
        weighted_items,
        weights=[item[2] for item in weighted_items],
        k=1,
    )[0]
    return {selected_key: selected_value}


"""
curl -X POST http://localhost:3000/api/db/lines \
  -H "Content-Type: application/json" \
  -d '{"stack": 100, "pos": "BTN_BB", "pot": "SRP", "hero": "BTN"}'
"""


@db_blueprint.get("/health")
def db_healthcheck():
    database = get_database()

    try:
        with database.get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("select 1 as ok")
                row = cursor.fetchone()
    except Exception as exc:
        current_app.logger.exception("Database healthcheck failed")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Database connection failed",
                    "details": str(exc),
                }
            ),
            500,
        )

    return jsonify(
        {
            "status": "ok",
            "database": {
                "host": database.host,
                "port": database.port,
                "name": database.dbname,
                "user": database.user,
            },
            "query_result": row[0] if row else None,
        }
    )


@db_blueprint.post("/lines")
def get_unique_lines_for_filters():

    payload = request.get_json(silent=True) or {}

    required_fields = ("stack", "pos", "pot", "hero", "street", "board")
    missing_fields = get_required_payload_fields(payload, required_fields)

    if missing_fields:

        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Missing required fields",
                    "missing_fields": missing_fields,
                }
            ),
            400,
        )
    print("Received lines request with payload:", payload)
    database = get_database()

    try:
        street_column, street_value = get_street_filter_condition(payload["street"])
        board_filter_clauses, board_filter_params = get_board_filter_clauses(
            payload["street"], payload["board"]
        )

        with database.get_connection() as connection:
            with connection.cursor() as cursor:
                query = sql.SQL(
                    "select distinct line from public.poki where stack = %s and pos = %s and pot = %s and hero = %s and {} = %s {board_filters} and line is not null order by line"
                ).format(
                    sql.Identifier(street_column),
                    board_filters=(
                        sql.SQL("\n                      ").join(board_filter_clauses)
                        if board_filter_clauses
                        else sql.SQL("")
                    ),
                )
                params = [
                    payload["stack"],
                    payload["pos"],
                    payload["pot"],
                    payload["hero"],
                    street_value,
                ] + board_filter_params

                cursor.execute(query, params)
                line_values = [row[0] for row in cursor.fetchall()]
    except Exception as exc:
        current_app.logger.exception("Fetching unique lines failed")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Failed to fetch unique line values",
                    "details": str(exc),
                }
            ),
            500,
        )

    return jsonify(
        {
            "lines": line_values,
        }
    )


@db_blueprint.post("/train")
def get_weighted_training_record():
    payload = request.get_json(silent=True) or {}
    required_fields = ("stack", "pos", "pot", "hero", "line", "street", "board")

    missing_fields = get_required_payload_fields(payload, required_fields)
    if missing_fields:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Missing required fields",
                    "missing_fields": missing_fields,
                }
            ),
            400,
        )

    requested_lines = payload["line"]
    if not isinstance(requested_lines, list) or not requested_lines:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Field 'line' must be a non-empty list",
                }
            ),
            400,
        )

    database = get_database()

    try:
        street_column, street_value = get_street_filter_condition(payload["street"])
        board_filter_clauses, board_filter_params = get_board_filter_clauses(
            payload["street"], payload["board"]
        )

        with database.get_connection() as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                query = sql.SQL(
                    """
                    select *
                    from public.poki
                    where stack = %s
                      and pos = %s
                      and pot = %s
                      and hero = %s
                      and {street_column} = %s
                      and line = any(%s)
                      {board_filters}
                      and total_weight is not null
                      and total_weight > 0
                    order by -ln(random()) / total_weight
                    limit 1
                    """
                ).format(
                    street_column=sql.Identifier(street_column),
                    board_filters=(
                        sql.SQL("\n                      ").join(board_filter_clauses)
                        if board_filter_clauses
                        else sql.SQL("")
                    ),
                )
                params = [
                    payload["stack"],
                    payload["pos"],
                    payload["pot"],
                    payload["hero"],
                    street_value,
                    requested_lines,
                ] + board_filter_params

                cursor.execute(query, params)
                record = cursor.fetchone()
    except ValueError as exc:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(exc),
                }
            ),
            400,
        )
    except Exception as exc:
        current_app.logger.exception("Fetching weighted training record failed")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Failed to fetch weighted training record",
                    "details": str(exc),
                }
            ),
            500,
        )

    if record is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "No matching training record found",
                }
            ),
            404,
        )

    task = select_weighted_task(record.get("data"))
    if task is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Matching record has no weighted task data",
                }
            ),
            404,
        )

    return jsonify(
        {
            "id": record["id"],
            "board": record["board"],
            "stack": record["stack"],
            "pos": record["pos"],
            "pot": record["pot"],
            "street": (
                "flop"
                if record["is_flop"]
                else "turn" if record["is_turn"] else "unknown"
            ),
            "line": record["line"],
            "hero": record["hero"],
            "task": normalize_json_value(task),
        }
    )


@db_blueprint.post("/tree")
def get_tree():
    payload = request.get_json(silent=True) or {}
    required_fields = ("id", "taskhand")
    missing_fields = get_required_payload_fields(payload, required_fields)
    if missing_fields:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Missing required fields",
                    "missing_fields": missing_fields,
                }
            ),
            400,
        )

    db = PokiDatabase()

    db = PokiDatabase()
    board, data, stack, pos, pot, line = db.get_poki_nth_board_and_data(payload["id"])
    features, actions = extract_features(board, data)
    # scores = rank_single_features(features, actions)
    # for i in scores[:4]:
    #     print(i.feature_name, i.loss)

    total_mass = actions.shape[0]

    def split(features, actions, all_feature_names=[], level=None):

        features = features.loc[:, (features.nunique() > 1)]
        mass = features.shape[0] / total_mass
        overall_action_ratios = get_overall_action_ratios(actions)
        if features.empty:

            try:
                taskhandhasit = features.loc[payload["taskhand"], :]
            except KeyError:
                taskhandhasit = None

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
        ) = get_best_single_feature(
            features, actions, all_feature_names=all_feature_names, level=level
        )
        mass = features.shape[0] / total_mass
        # print("splitben", best_score.feature_name)
        best_overall_action = max(overall_action_ratios, key=overall_action_ratios.get)
        best_true_action = max(true_ratios_str, key=true_ratios_str.get)
        best_false_action = max(false_ratios_str, key=false_ratios_str.get)

        actions_split = (
            best_overall_action != best_true_action
            or best_true_action != best_false_action
            or best_overall_action != best_false_action
        )

        if mass < 0.025 and not actions_split:
            try:
                taskhandhasit = features.loc[payload["taskhand"], :]

            except KeyError:
                taskhandhasit = None

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
            taskhandhasit = features.loc[payload["taskhand"], :]
        except KeyError:
            taskhandhasit = None
        # print("taskhand else", taskhandhasit)
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

        return {
            "mass": mass,
            "actions": overall_action_ratios,
            "best_feature": best_score.feature_name,
            "split": actions_split,
            "true": split(
                features_true, actions_true, all_feature_names, level=level + 1
            ),
            "false": split(
                features_false, actions_false, all_feature_names, level=level
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

    all_feature_names = features.loc[:, (features.nunique() > 1)].columns.tolist()

    tree = split(features, actions, all_feature_names, 0)

    overall_actions = tree["actions"]

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
                    (subTree["true"]["cards"].to_dict(orient="index"))
                    if not true_splits
                    else {}
                ),
                "taskhand": subTree["true"]["taskhand"] if not true_splits else "",
                "best_feature": subTree["best_feature"],
                "mass": subTree["true"]["mass"],
                "show": False,
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
                    "cards": (subTree["false"]["cards"].to_dict(orient="index")),
                    "taskhand": subTree["false"]["taskhand"],
                    "best_feature": "",
                    "mass": subTree["false"]["mass"],
                    "show": False,
                    "subtree": [],
                }
            )

        if subTree["false"]["best_feature"] is not None:
            subresult = subresult + build_result(subTree["false"])
        return subresult

    result = build_result(tree)

    return jsonify({"overall_actions": overall_actions, "tree": result})
