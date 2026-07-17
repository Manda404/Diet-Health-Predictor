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

NUMERIC_FEATURES = RAW_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
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

    Pipeline: load -> clean -> engineer features -> split -> encode/scale -> persist
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
        clean_df = cleaner.clean(raw_df)

        engineered_df = FeatureEngineer().transform(clean_df)

        train_df, test_df = DataSplitter().split(
            engineered_df, self.target_column, self.test_size, self.random_state
        )

        transformer = FeatureTransformer(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        X_train = transformer.fit_transform(train_df)
        X_test = transformer.transform(test_df)
        y_train = train_df[self.target_column]
        y_test = test_df[self.target_column]

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
