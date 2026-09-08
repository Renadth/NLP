"""Lab 3A: dataset construction and grouped split integrity."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def build_topic_dataset(
    path: str | Path = DATA_PATH,
    random_state: int = 42,
):
    """Build train/validation/test splits grouped by citizen."""

    df = pd.read_csv(path)

    required_columns = {
        "text",
        "topic",
        "citizen_group_id",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=["text", "topic", "citizen_group_id"]
    ).copy()

    df["text"] = df["text"].astype(str)
    df["topic"] = df["topic"].astype(str)
    df["citizen_group_id"] = df["citizen_group_id"].astype(str)

    # First split: 80% train, 20% temporary
    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_state,
    )

    train_idx, temp_idx = next(
        splitter_1.split(
            df,
            groups=df["citizen_group_id"],
        )
    )

    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()

    # Second split: split the 20% temporary set equally
    # into validation and test.
    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=random_state,
    )

    validation_idx, test_idx = next(
        splitter_2.split(
            temp,
            groups=temp["citizen_group_id"],
        )
    )

    validation = temp.iloc[validation_idx].copy()
    test = temp.iloc[test_idx].copy()

    # Verify citizen-level leakage is impossible.
    train_ids = set(train["citizen_group_id"])
    validation_ids = set(validation["citizen_group_id"])
    test_ids = set(test["citizen_group_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


if __name__ == "__main__":
    dataset = build_topic_dataset()

    print("Train:", len(dataset["train"]))
    print("Validation:", len(dataset["validation"]))
    print("Test:", len(dataset["test"]))

    train_ids = set(dataset["train"]["citizen_group_id"])
    validation_ids = set(dataset["validation"]["citizen_group_id"])
    test_ids = set(dataset["test"]["citizen_group_id"])

    print(
        "Train/Validation citizen overlap:",
        len(train_ids & validation_ids),
    )
    print(
        "Train/Test citizen overlap:",
        len(train_ids & test_ids),
    )
    print(
        "Validation/Test citizen overlap:",
        len(validation_ids & test_ids),
    )