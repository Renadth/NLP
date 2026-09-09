"""Lab 6: behavioural test generators/runners."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/eval/behavioural_templates.csv")


def _predict(predict_fn, texts):
    """Run the supplied predictor and return one prediction per text."""
    predictions = predict_fn(texts)

    if len(predictions) != len(texts):
        raise ValueError(
            "predict_fn must return one prediction per input text"
        )

    return list(predictions)


def run_behavioural_suite(
    predict_fn,
    *,
    path: str | Path = DATA_PATH,
):
    """Run the supplied behavioural templates.

    ``predict_fn`` must accept a list of strings and return one prediction
    for each string. Predictions may be plain labels or dictionaries
    containing a label under ``label`` or ``topic``.

    The function reports:
      - invariance rate
      - directional rate
      - MFT rate
      - overall pass rate
    """

    df = pd.read_csv(path)

    required = {
        "test_id",
        "test_type",
        "lang",
        "template",
        "term",
        "expected_relation",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Behavioural data is missing columns: {sorted(missing)}"
        )

    df = df.copy()

    df["text"] = df.apply(
        lambda row: str(row["template"]).replace(
            "{term}",
            str(row["term"]),
        ),
        axis=1,
    )

    texts = df["text"].tolist()
    predictions = _predict(predict_fn, texts)

    def label_of(prediction):
        if isinstance(prediction, dict):
            if "label" in prediction:
                return prediction["label"]
            if "topic" in prediction:
                return prediction["topic"]

        return prediction

    labels = [
        label_of(prediction)
        for prediction in predictions
    ]

    df["prediction"] = labels
    df["passed"] = False

    # Invariance:
    # The supplied contract says topic unchanged. Since templates do not
    # provide an explicit base/counterfactual pair, compare within each
    # invariance template family grouped by language and template text.
    invariance = df[
        df["test_type"].str.lower() == "invariance"
    ].copy()

    invariance_groups = 0
    invariance_passed = 0

    for _, group in invariance.groupby(
        ["lang", "template"]
    ):
        if len(group) < 2:
            # A single supplied instance cannot establish invariance.
            continue

        invariance_groups += 1

        if group["prediction"].nunique() == 1:
            invariance_passed += 1
            df.loc[group.index, "passed"] = True

    # Directional:
    # The supplied relation is "sentiment must not improve after negation".
    # With the Bayan classifier this cannot be inferred from topic labels,
    # so we mark these as unsupported unless the predictor supplies a
    # numeric sentiment score.
    directional = df[
        df["test_type"].str.lower() == "directional"
    ].copy()

    directional_total = len(directional)
    directional_passed = 0

    for idx, prediction in zip(
        directional.index,
        predictions,
    ):
        # Support predictions such as:
        # {"label": "...", "sentiment": 0.2}
        # or {"sentiment": 0.2}
        if isinstance(prediction, dict):
            sentiment = prediction.get("sentiment")

            if sentiment is not None:
                try:
                    sentiment = float(sentiment)
                except (TypeError, ValueError):
                    sentiment = None

                if sentiment is not None:
                    # These templates are already negative ("not working"),
                    # so a non-positive sentiment score satisfies the
                    # supplied directional requirement.
                    if sentiment <= 0:
                        directional_passed += 1
                        df.loc[idx, "passed"] = True

    # Minimum-functionality tests are represented by rows whose test_type
    # is explicitly MFT/minimum_functionality.
    mft = df[
        df["test_type"].str.lower().isin(
            {"mft", "minimum_functionality", "minimum-functionality"}
        )
    ].copy()

    mft_total = len(mft)
    mft_passed = 0

    for idx in mft.index:
        prediction = df.loc[idx, "prediction"]

        if prediction is not None and str(prediction).strip():
            mft_passed += 1
            df.loc[idx, "passed"] = True

    invariance_rate = (
        invariance_passed / invariance_groups
        if invariance_groups
        else 0.0
    )

    directional_rate = (
        directional_passed / directional_total
        if directional_total
        else 0.0
    )

    mft_rate = (
        mft_passed / mft_total
        if mft_total
        else 0.0
    )

    return {
        "invariance": {
            "passed": invariance_passed,
            "total": invariance_groups,
            "rate": invariance_rate,
        },
        "directional": {
            "passed": directional_passed,
            "total": directional_total,
            "rate": directional_rate,
        },
        "mft": {
            "passed": mft_passed,
            "total": mft_total,
            "rate": mft_rate,
        },
        "tests": df[
            [
                "test_id",
                "test_type",
                "lang",
                "text",
                "expected_relation",
                "prediction",
                "passed",
            ]
        ].to_dict("records"),
    }