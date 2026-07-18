"""
Integration tests: the full Phase 2 pipeline (load -> clean -> engineer ->
split -> encode/scale -> persist), exercised together with real file I/O
against the shared mock dataset. Unit tests already cover each collaborator
(DataCleaner, FeatureEngineer, FeatureTransformer, DataSplitter,
ProcessedDataWriter) in isolation — these tests are about the seams between
them.
"""

import pandas as pd
import pytest

from diet_health_predictor.application.feature_engineering import PreprocessDataUseCase
from diet_health_predictor.infrastructure import (
    DataCleaner,
    FeatureTransformer,
    HealthDietDataLoader,
    get_dataset_summary,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def preprocess_use_case(mock_csv_path, tmp_path):
    loader = HealthDietDataLoader(str(mock_csv_path))
    return PreprocessDataUseCase(
        data_loader=loader,
        output_dir=str(tmp_path),
        target_column="Health_Status",
        test_size=0.25,
        random_state=42,
    )


class TestPreprocessDataUseCase:
    def test_train_and_test_rows_add_up_to_the_deduplicated_dataset(self, preprocess_use_case):
        result = preprocess_use_case.execute()
        assert len(result.X_train) + len(result.X_test) == 16  # 17 rows, 1 exact duplicate

    def test_output_has_no_missing_values_after_cleaning_and_encoding(self, preprocess_use_case):
        result = preprocess_use_case.execute()
        assert result.X_train.isna().sum().sum() == 0
        assert result.X_test.isna().sum().sum() == 0

    def test_all_four_health_status_classes_are_present_across_train_and_test(
        self, preprocess_use_case
    ):
        result = preprocess_use_case.execute()
        classes_seen = set(result.y_train) | set(result.y_test)
        assert classes_seen == {"Healthy", "Overweight", "Obese", "Underweight"}

    def test_feature_names_match_output_columns(self, preprocess_use_case):
        result = preprocess_use_case.execute()
        assert result.feature_names == list(result.X_train.columns)
        assert result.feature_names == list(result.X_test.columns)

    def test_persists_train_test_features_targets_and_transformer_to_disk(
        self, preprocess_use_case, tmp_path
    ):
        preprocess_use_case.execute()

        assert (tmp_path / "X_train.csv").exists()
        assert (tmp_path / "X_test.csv").exists()
        assert (tmp_path / "y_train.csv").exists()
        assert (tmp_path / "y_test.csv").exists()
        assert (tmp_path / "feature_transformer.joblib").exists()
        assert (tmp_path / "data_cleaner.joblib").exists()

    def test_persisted_x_train_matches_the_returned_result(self, preprocess_use_case, tmp_path):
        """Regression check that ProcessedDataWriter wrote exactly what the use case returned."""
        result = preprocess_use_case.execute()
        x_train_on_disk = pd.read_csv(tmp_path / "X_train.csv")
        pd.testing.assert_frame_equal(
            x_train_on_disk, result.X_train.reset_index(drop=True), check_exact=False
        )

    def test_saved_transformer_reloads_with_matching_feature_names(self, preprocess_use_case):
        result = preprocess_use_case.execute()
        reloaded_transformer = FeatureTransformer.load(result.transformer_path)
        assert reloaded_transformer.feature_names_out() == result.feature_names

    def test_saved_cleaner_reloads_and_transforms_a_record_missing_id_and_target(
        self, preprocess_use_case, mock_raw_df
    ):
        # A prediction-time record (Phase 4) has no Person_ID/Health_Status --
        # transform() must not require columns it hasn't seen since training.
        result = preprocess_use_case.execute()
        reloaded_cleaner = DataCleaner.load(result.cleaner_path)

        # P0009 (index 8) has a missing Activity_Level -- exercises imputation
        # via the training-derived mode, not just a pass-through no-op.
        record = mock_raw_df.drop(columns=["Person_ID", "Health_Status"]).iloc[[8]]
        assert record["Activity_Level"].isna().all()
        transformed = reloaded_cleaner.transform(record)

        assert len(transformed) == 1
        assert transformed.isna().sum().sum() == 0

    def test_split_is_reproducible_given_the_same_random_state(self, mock_csv_path, tmp_path):
        loader = HealthDietDataLoader(str(mock_csv_path))
        use_case_a = PreprocessDataUseCase(
            loader, str(tmp_path / "a"), "Health_Status", 0.25, random_state=7
        )
        use_case_b = PreprocessDataUseCase(
            loader, str(tmp_path / "b"), "Health_Status", 0.25, random_state=7
        )

        result_a = use_case_a.execute()
        result_b = use_case_b.execute()

        pd.testing.assert_frame_equal(result_a.X_train, result_b.X_train)
        pd.testing.assert_series_equal(
            result_a.y_train.reset_index(drop=True), result_b.y_train.reset_index(drop=True)
        )


class TestGetDatasetSummaryOnMockData:
    def test_reports_the_known_duplicate_missing_and_outlier_signals(self, mock_raw_df):
        summary = get_dataset_summary(mock_raw_df).set_index("Column")

        assert summary.loc["Weight_kg", "Missing"] == 1
        assert summary.loc["Activity_Level", "Missing"] == 1
        assert summary.loc["Fat_Intake_g", "Missing"] == 1
        assert summary.loc["Fat_Intake_g", "% Outliers"] > 0
        assert summary.loc["Person_ID", "Cardinality"] == 16  # 17 rows, 1 duplicate ID
