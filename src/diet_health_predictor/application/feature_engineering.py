"""
Application Layer - Preprocessing & Feature Engineering Use Case
===================================================================

Orchestrates the Phase 2 pipeline: load raw data, clean it, engineer features,
split into train/test, encode/scale, and persist the result.

This layer decides *which* columns are numeric/categorical and *how* the
pipeline is sequenced; the Infrastructure layer (`DataCleaner`,
`FeatureEngineer`, `FeatureTransformer`, `DataSplitter`, `ProcessedDataWriter`)
implements the mechanics.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from diet_health_predictor.infrastructure import (
    DataCleaner,
    DataSplitter,
    FeatureEngineer,
    FeatureTransformer,
    HealthDietDataLoader,
    ProcessedDataWriter,
)

logger = logging.getLogger(__name__)

RAW_NUMERIC_COLUMNS = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Daily_Calorie_Requirement",
    "Daily_Calorie_Consumed",
    "Protein_Intake_g",
    "Carbohydrate_Intake_g",
    "Fat_Intake_g",
    "Water_Intake_Liters",
]
RAW_CATEGORICAL_COLUMNS = ["Gender", "Activity_Level", "Diet_Type"]
ENGINEERED_NUMERIC_COLUMNS = [
    "Calorie_Balance",
    "Calorie_Deviation_Pct",
    "Total_Macros_g",
    "Protein_Ratio",
    "Carb_Ratio",
    "Fat_Ratio",
    "Adequate_Water_Intake",
    "BMR",
    "Activity_Calorie_Multiplier",
    "Protein_per_kg_Bodyweight",
    "Water_Intake_ml_per_kg",
    "Ideal_Weight_kg",
    "Weight_Deviation_kg",
]
ENGINEERED_CATEGORICAL_COLUMNS = ["Age_Group"]

# `Health_Status` is a deterministic function of `BMI` in this dataset (WHO
# BMI thresholds, verified on the full 6000-row dataset with zero exceptions)
# -- a model given BMI trivially learns the threshold rule (100% accuracy).
# `Height_cm`/`Weight_kg` jointly reconstruct BMI almost perfectly (~98%
# accuracy with BMI removed but those two kept), and every engineered column
# below is itself a function of Height_cm/Weight_kg (BMR, Ideal_Weight_kg,
# Weight_Deviation_kg, and the two per-kg-bodyweight ratios), so excluding
# BMI alone is not enough. These columns are still computed by
# `FeatureEngineer` (useful for analysis/notebooks) but excluded here from
# the columns the model actually trains on.
LEAKING_NUMERIC_COLUMNS = [
    "Height_cm",
    "Weight_kg",
    "BMI",
    "BMR",
    "Activity_Calorie_Multiplier",
    "Protein_per_kg_Bodyweight",
    "Water_Intake_ml_per_kg",
    "Ideal_Weight_kg",
    "Weight_Deviation_kg",
]

NUMERIC_FEATURES = [
    column
    for column in RAW_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
    if column not in LEAKING_NUMERIC_COLUMNS
]
CATEGORICAL_FEATURES = RAW_CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS


@dataclass
class PreprocessingResult:
    """Output of the preprocessing pipeline, ready for model training (Phase 3)"""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    transformer_path: str


class PreprocessDataUseCase:
    """
    Use Case: Turn raw health diet data into model-ready train/test features.

    Pipeline: load -> dedup -> split -> clean (fit on train only) ->
    engineer features -> encode/scale (fit on train only) -> persist.

    Deduplication runs before the split (an exact-duplicate row must not end
    up once in train and once in test). Imputation, outlier clipping, and
    encoding/scaling are all fit on the training split only and then applied
    to test -- fitting any of them on the full dataset before splitting
    would leak test-set statistics into the training data.
    """

    def __init__(
        self,
        data_loader: HealthDietDataLoader,
        output_dir: str,
        target_column: str = "Health_Status",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def execute(self, sample_size: int | None = None) -> PreprocessingResult:
        logger.info("Starting data preprocessing & feature engineering pipeline...")

        raw_df = self.data_loader.load(sample_size)

        cleaner = DataCleaner(outlier_columns=RAW_NUMERIC_COLUMNS)
        deduped_df = cleaner.drop_duplicates(raw_df)

        train_raw, test_raw = DataSplitter().split(
            deduped_df, self.target_column, self.test_size, self.random_state
        )

        # Imputation/outlier bounds are fit on the training split only, then
        # applied to both -- fitting them on the full dataset before the
        # split would leak test-set statistics into the training data.
        cleaner.fit(train_raw)
        train_clean = cleaner.transform(train_raw)
        test_clean = cleaner.transform(test_raw)

        feature_engineer = FeatureEngineer()
        train_engineered = feature_engineer.transform(train_clean)
        test_engineered = feature_engineer.transform(test_clean)

        transformer = FeatureTransformer(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        X_train = transformer.fit_transform(train_engineered)
        X_test = transformer.transform(test_engineered)
        y_train = train_engineered[self.target_column]
        y_test = test_engineered[self.target_column]

        writer = ProcessedDataWriter(self.output_dir)
        writer.write_features(X_train, X_test)
        writer.write_targets(y_train, y_test)
        transformer.save(writer.transformer_path())

        logger.info(
            f"Preprocessing complete: {X_train.shape[1]} features, "
            f"{len(X_train)} train / {len(X_test)} test rows"
        )

        return PreprocessingResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=transformer.feature_names_out(),
            transformer_path=writer.transformer_path(),
        )
