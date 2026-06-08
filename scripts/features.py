"""Feature engineering for the Ames house-price model.

This module holds *stateless, deterministic* feature transforms only. The same
function is called from training and from the model service so that features are
built identically offline and online (train/serve parity). Anything that needs
to be *fitted* on the training data (imputation values, scaling, categorical
encoding) belongs in a downstream scikit-learn pipeline, not here.

Usage
-----
As a library::

    from scripts.features import build_features
    X = build_features(raw_df)            # inference
    X, y = build_features(raw_df, return_target=True)   # training

As a script::

    python scripts/features.py data/raw_data/train.csv data/good_data/train_features.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

TARGET = "SalePrice"
ID_COL = "Id"

# Quality/condition columns recorded as Po < Fa < TA < Gd < Ex. NaN means the
# feature is absent (e.g. no basement), which is a meaningful 0 here.
QUALITY_MAP = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLS = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
    "PoolQC",
]

# Other ordinal columns with their own ordering.
ORDINAL_MAPS = {
    "BsmtExposure": {"No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {
        "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
        "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8,
    },
    "CentralAir": {"N": 0, "Y": 1},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
}


def _encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Map ordinal text columns to integers; absent categories become 0."""
    for col in QUALITY_COLS:
        if col in df.columns:
            df[col] = df[col].map(QUALITY_MAP).fillna(0).astype("int8")
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype("int8")
    return df


def _add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numeric features common for the Ames dataset."""
    # Fill the area/count fields used in arithmetic so sums don't go NaN. These
    # NaNs encode "feature absent" (no basement, no garage), so 0 is correct.
    area_cols = [
        "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "BsmtFinSF1", "BsmtFinSF2",
        "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
        "ScreenPorch", "MasVnrArea", "GarageArea", "PoolArea",
        "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "Fireplaces",
    ]
    for col in area_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Total living/usable area.
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalFinishedBsmtSF"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]

    # Bathrooms count half-baths as 0.5.
    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # Porch / deck footprint.
    df["TotalPorchSF"] = (
        df["WoodDeckSF"]
        + df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
    )

    # Ages relative to the year sold (clip negatives from data quirks to 0).
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype("int8")
    df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype("int8")

    # Presence flags.
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype("int8")
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype("int8")
    df["HasGarage"] = (df["GarageArea"] > 0).astype("int8")
    df["HasPool"] = (df["PoolArea"] > 0).astype("int8")
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype("int8")
    df["HasPorch"] = (df["TotalPorchSF"] > 0).astype("int8")

    # Overall quality interaction.
    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]

    return df


def build_features(
    df: pd.DataFrame,
    *,
    return_target: bool = False,
    drop_id: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Transform a raw Ames dataframe into model-ready features.

    Parameters
    ----------
    df:
        Raw data as loaded from the Kaggle CSVs. May or may not contain the
        ``SalePrice`` target (it is absent at serving time).
    return_target:
        If ``True``, also return the ``SalePrice`` series. Requires the column
        to be present.
    drop_id:
        Drop the ``Id`` column from the returned features.

    Returns
    -------
    Either ``X`` or the tuple ``(X, y)``.
    """
    df = df.copy()

    df = _encode_ordinals(df)
    df = _add_engineered(df)

    y = None
    if return_target:
        if TARGET not in df.columns:
            raise KeyError(
                f"{TARGET!r} not found; cannot return target for inference data."
            )
        y = df[TARGET].astype(float)

    X = df.drop(columns=[c for c in (TARGET,) if c in df.columns])
    if drop_id and ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])

    if return_target:
        return X, y
    return X


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build engineered features from a raw Ames CSV."
    )
    parser.add_argument("input_csv", help="Path to raw input CSV (e.g. train.csv).")
    parser.add_argument("output_csv", help="Path to write the feature CSV.")
    parser.add_argument(
        "--keep-id",
        action="store_true",
        help="Keep the Id column in the output.",
    )
    args = parser.parse_args(argv)

    raw = pd.read_csv(args.input_csv)
    features = build_features(raw, drop_id=not args.keep_id)
    features.to_csv(args.output_csv, index=False)
    print(
        f"Wrote {features.shape[0]} rows x {features.shape[1]} cols -> "
        f"{args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
