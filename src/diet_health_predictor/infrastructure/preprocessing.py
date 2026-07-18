"""
Infrastructure Layer - Preprocessing & Feature Engineering Module
===================================================================

Handles the mechanics of turning a raw DataFrame into model-ready features:
- Cleaning (duplicates, missing values, outliers)
- Feature engineering (derived columns)
- Encoding / scaling (scikit-learn transformers)
- Train/test splitting
- Persisting processed data and fitted transformers to disk

Following Clean Architecture principles:
- No orchestration logic here (that belongs to the Application layer's use cases)
- Wraps third-party libraries (pandas, scikit-learn, joblib) so the rest of the
  codebase never imports them directly
"""

import logging
from pathlib import Path
from typing import Optional, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Cleans a DataFrame: duplicates, missing values, outliers.

    Deduplication is stateless (`drop_duplicates()`) and meant to run on the
    full dataset *before* splitting -- it must, so an exact-duplicate row
    can't end up once in train and once in test.

    Imputation and outlier clipping are learned from data (median/mode,
    IQR bounds) and follow the same fit/transform split as
    `FeatureTransformer`: `fit()` on the training split only, then
    `transform()` applies those training-derived values to both train and
    test. Fitting them on the full dataset before splitting would leak
    test-set statistics into the values used to clean the training data.
    """

    def __init__(
        self, outlier_columns: Optional[list[str]] = None, outlier_iqr_factor: float = 3.0
    ):
        self.outlier_columns = outlier_columns or []
        self.outlier_iqr_factor = outlier_iqr_factor
        self._impute_values: dict = {}
        self._outlier_bounds: dict = {}
        self._fitted = False

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset="Person_ID" if "Person_ID" in df.columns else None)
        dropped = before - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} duplicate row(s)")
        return df.reset_index(drop=True)

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        # Learn an impute value for every column unconditionally -- not just
        # ones with a missing value *at fit time* -- so a column that's
        # complete in the fitting data but has a gap in whatever gets
        # transform()'d later (typically the test split) still has a valid
        # training-derived fallback ready.
        self._impute_values = {}
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                self._impute_values[column] = df[column].median()
            else:
                mode = df[column].mode(dropna=True)
                if not mode.empty:
                    self._impute_values[column] = mode.iloc[0]

        self._outlier_bounds = {}
        for column in self.outlier_columns:
            if column not in df.columns:
                continue
            q1, q3 = df[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            self._outlier_bounds[column] = (
                q1 - self.outlier_iqr_factor * iqr,
                q3 + self.outlier_iqr_factor * iqr,
            )

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("DataCleaner must be fitted before calling transform()")
        df = df.copy()

        # Columns absent here (e.g. Person_ID/Health_Status on a single
        # inference-time record that carries only feature columns) are
        # skipped rather than raising -- fit() records values for every
        # column seen in the training data, which is a superset of what
        # any later caller is guaranteed to provide.
        for column, fill_value in self._impute_values.items():
            if column not in df.columns:
                continue
            missing = df[column].isna().sum()
            if missing:
                df[column] = df[column].fillna(fill_value)
                logger.info(f"Imputed {missing} missing value(s) in '{column}' with {fill_value}")

        for column, (lower, upper) in self._outlier_bounds.items():
            if column not in df.columns:
                continue
            clipped = ((df[column] < lower) | (df[column] > upper)).sum()
            if clipped:
                logger.info(
                    f"Clipped {clipped} outlier(s) in '{column}' to [{lower:.2f}, {upper:.2f}]"
                )
            df[column] = df[column].clip(lower, upper)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience for the (train-only) fit + transform of the same data."""
        return self.fit(df).transform(df)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved DataCleaner to {path}")

    @classmethod
    def load(cls, path: str) -> "DataCleaner":
        return cast("DataCleaner", joblib.load(path))


class FeatureEngineer:
    """
    Derives new columns from the raw health/diet columns.

    These formulas mirror the single-record business rules in
    `diet_health_predictor.domain.NutritionData` / `Person`, reimplemented
    vectorized over a DataFrame for batch processing.
    """

    KCAL_PER_G_PROTEIN = 4.0
    KCAL_PER_G_CARB = 4.0
    KCAL_PER_G_FAT = 9.0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Age_Group"] = self._age_group(df["Age"])
        df["Calorie_Balance"] = df["Daily_Calorie_Consumed"] - df["Daily_Calorie_Requirement"]
        df["Calorie_Deviation_Pct"] = np.where(
            df["Daily_Calorie_Requirement"] != 0,
            df["Calorie_Balance"] / df["Daily_Calorie_Requirement"] * 100,
            0.0,
        )

        protein_kcal = df["Protein_Intake_g"] * self.KCAL_PER_G_PROTEIN
        carb_kcal = df["Carbohydrate_Intake_g"] * self.KCAL_PER_G_CARB
        fat_kcal = df["Fat_Intake_g"] * self.KCAL_PER_G_FAT
        total_kcal = protein_kcal + carb_kcal + fat_kcal

        df["Total_Macros_g"] = (
            df["Protein_Intake_g"] + df["Carbohydrate_Intake_g"] + df["Fat_Intake_g"]
        )
        df["Protein_Ratio"] = np.where(total_kcal != 0, protein_kcal / total_kcal, 0.0)
        df["Carb_Ratio"] = np.where(total_kcal != 0, carb_kcal / total_kcal, 0.0)
        df["Fat_Ratio"] = np.where(total_kcal != 0, fat_kcal / total_kcal, 0.0)

        df["Adequate_Water_Intake"] = (df["Water_Intake_Liters"] >= 2.0).astype(int)

        df["BMR"] = self._bmr(df["Weight_kg"], df["Height_cm"], df["Age"], df["Gender"])
        df["Activity_Calorie_Multiplier"] = np.where(
            df["BMR"] != 0, df["Daily_Calorie_Requirement"] / df["BMR"], 0.0
        )
        df["Protein_per_kg_Bodyweight"] = np.where(
            df["Weight_kg"] != 0, df["Protein_Intake_g"] / df["Weight_kg"], 0.0
        )
        df["Water_Intake_ml_per_kg"] = np.where(
            df["Weight_kg"] != 0, df["Water_Intake_Liters"] * 1000 / df["Weight_kg"], 0.0
        )
        df["Ideal_Weight_kg"] = self._ideal_weight_kg(df["Height_cm"], df["Gender"])
        df["Weight_Deviation_kg"] = df["Weight_kg"] - df["Ideal_Weight_kg"]

        logger.info(
            "Engineered features: Age_Group, Calorie_Balance, Calorie_Deviation_Pct, "
            "Total_Macros_g, Protein_Ratio, Carb_Ratio, Fat_Ratio, Adequate_Water_Intake, "
            "BMR, Activity_Calorie_Multiplier, Protein_per_kg_Bodyweight, "
            "Water_Intake_ml_per_kg, Ideal_Weight_kg, Weight_Deviation_kg"
        )
        return df

    @staticmethod
    def _age_group(age: pd.Series) -> pd.Series:
        return pd.cut(
            age,
            bins=[-np.inf, 17, 29, 59, np.inf],
            labels=["Child", "Young Adult", "Adult", "Senior"],
        ).astype(str)

    @staticmethod
    def _bmr(
        weight_kg: pd.Series, height_cm: pd.Series, age: pd.Series, gender: pd.Series
    ) -> pd.Series:
        """Basal Metabolic Rate, kcal/day (Mifflin-St Jeor). 'Other' uses the average offset."""
        base = 10 * weight_kg + 6.25 * height_cm - 5 * age
        offset = np.select(
            [gender == "Male", gender == "Female"],
            [5, -161],
            default=-78,  # average of +5 and -161
        )
        return base + offset

    @staticmethod
    def _ideal_weight_kg(height_cm: pd.Series, gender: pd.Series) -> pd.Series:
        """Ideal body weight, kg (Devine formula). 'Other' uses the average offset."""
        inches_over_5ft = height_cm / 2.54 - 60
        base_offset = np.select(
            [gender == "Male", gender == "Female"],
            [50, 45.5],
            default=47.75,  # average of 50 and 45.5
        )
        return base_offset + 2.3 * inches_over_5ft


class FeatureTransformer:
    """
    Encodes categorical columns and scales numeric columns using scikit-learn.

    Wraps a `ColumnTransformer` (StandardScaler + OneHotEncoder) so the rest of
    the codebase interacts with plain DataFrames, not raw sklearn objects.
    """

    def __init__(self, numeric_features: list[str], categorical_features: list[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric_features),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
            ]
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        self._column_transformer.fit(df[self.numeric_features + self.categorical_features])
        self._fitted = True
        logger.info(
            f"Fitted FeatureTransformer on {len(self.numeric_features)} numeric and "
            f"{len(self.categorical_features)} categorical column(s)"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureTransformer must be fitted before calling transform()")
        array = self._column_transformer.transform(
            df[self.numeric_features + self.categorical_features]
        )
        return pd.DataFrame(array, columns=self.feature_names_out(), index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def feature_names_out(self) -> list[str]:
        return list(self._column_transformer.get_feature_names_out())

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved FeatureTransformer to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureTransformer":
        return cast("FeatureTransformer", joblib.load(path))


class DataSplitter:
    """Splits a DataFrame into stratified train/test sets."""

    def split(
        self, df: pd.DataFrame, target_column: str, test_size: float, random_state: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_column],
        )
        logger.info(f"Split data into {len(train_df)} train / {len(test_df)} test rows")
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


class ProcessedDataWriter:
    """Persists processed features/targets and the fitted transformer to disk."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def write_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(self.output_dir / "X_train.csv", index=False)
        X_test.to_csv(self.output_dir / "X_test.csv", index=False)
        logger.info(f"Wrote X_train/X_test to {self.output_dir}")

    def write_targets(self, y_train: pd.Series, y_test: pd.Series) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        y_train.to_csv(self.output_dir / "y_train.csv", index=False)
        y_test.to_csv(self.output_dir / "y_test.csv", index=False)
        logger.info(f"Wrote y_train/y_test to {self.output_dir}")

    def transformer_path(self) -> str:
        return str(self.output_dir / "feature_transformer.joblib")

    def cleaner_path(self) -> str:
        return str(self.output_dir / "data_cleaner.joblib")
