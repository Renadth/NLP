"""Lab 6: sliced evaluation report."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


MIN_SLICE_SIZE = 30


def _score(y_true, y_pred):
    if len(y_true) == 0:
        return float("nan")

    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
        )
    )


def _add_slice(results, name, values, y_true, y_pred):
    values = pd.Series(values).reset_index(drop=True)
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    for value in sorted(values.dropna().unique(), key=str):
        mask = values == value

        size = int(mask.sum())

        results[f"{name}={value}"] = {
            "slice": name,
            "value": value,
            "n": size,
            "macro_f1": _score(
                y_true[mask],
                y_pred[mask],
            ),
            "small_slice": size < min_slice_size,
        }


def sliced_report(
    frame,
    y_true,
    y_pred,
    *,
    min_slice_size=MIN_SLICE_SIZE,
):
    """Return macro-F1 for useful language/dialect/class/length slices.

    Parameters
    ----------
    frame:
        DataFrame containing available slice columns such as
        ``lang``, ``dialect_region``, ``topic`` and optionally ``text``.
    y_true:
        Ground-truth class labels.
    y_pred:
        Predicted class labels.
    min_slice_size:
        Slices smaller than this are flagged as ``small_slice=True``.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(frame) != len(y_true) or len(frame) != len(y_pred):
        raise ValueError(
            "frame, y_true, and y_pred must have the same length"
        )

    if min_slice_size <= 0:
        raise ValueError(
            "min_slice_size must be positive"
        )

    results = {}

    # Language slice.
    if "lang" in frame.columns:
        _add_slice(
            results,
            "language",
            frame["lang"],
            y_true,
            y_pred,
        )

    # Dialect slice.
    if "dialect_region" in frame.columns:
        _add_slice(
            results,
            "dialect",
            frame["dialect_region"],
            y_true,
            y_pred,
        )

    # Class slice.
    # Use individual classes, with binary F1 for each class.
    if "topic" in frame.columns:
        topics = pd.Series(
            frame["topic"]
        ).reset_index(drop=True)

        true_series = pd.Series(y_true)
        pred_series = pd.Series(y_pred)

        for topic in sorted(
            topics.dropna().unique(),
            key=str,
        ):
            mask = topics == topic
            n = int(mask.sum())

            # For a class slice, report whether the model
            # correctly predicts that class.
            class_true = (true_series == topic).astype(int)
            class_pred = (pred_series == topic).astype(int)

            results[f"class={topic}"] = {
                "slice": "class",
                "value": topic,
                "n": n,
                "f1": float(
                    f1_score(
                        class_true[mask],
                        class_pred[mask],
                        zero_division=0,
                    )
                ),
                "small_slice": n < min_slice_size,
            }

    # Length slice.
    if "text" in frame.columns:
        lengths = (
            frame["text"]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )

        bins = pd.cut(
            lengths,
            bins=[-np.inf, 10, 30, 60, np.inf],
            labels=[
                "short<=10",
                "medium=11-30",
                "long=31-60",
                "very_long>60",
            ],
        )

        _add_slice(
            results,
            "length",
            bins,
            y_true,
            y_pred,
        )

    return results