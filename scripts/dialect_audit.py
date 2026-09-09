"""Lab 4: audit Arabic dialect distribution."""


import pandas as pd
from pathlib import Path


def main():
    data_path = Path("data/raw/bayan_feedback.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Feedback data not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    arabic = df[df["lang"].str.lower() == "ar"].copy()

    print(f"Total feedback rows: {len(df)}")
    print(f"Arabic rows: {len(arabic)}")
    print()

    distribution = (
        arabic["dialect_region"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("dialect_region")
        .reset_index(name="count")
    )

    distribution["percentage"] = (
        distribution["count"] / len(arabic) * 100
    ).round(2)

    print("Arabic dialect/region distribution:")
    print(distribution.to_string(index=False))

    print()
    print(
        "Implication: Evaluating only on MSA may overestimate model performance "
        "because it does not represent the full dialect distribution of the Arabic data."
    )


if __name__ == "__main__":
    main()