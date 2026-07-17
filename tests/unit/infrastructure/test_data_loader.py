"""Unit tests for diet_health_predictor.infrastructure.data_loader"""

import pandas as pd
import pytest

from diet_health_predictor.infrastructure.data_loader import (
    DataLoader,
    DataLoadError,
    HealthDietDataLoader,
    get_dataset_summary,
)

pytestmark = pytest.mark.unit


class TestDataLoader:
    def test_raises_file_not_found_for_missing_path(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            DataLoader(str(missing_path))

    def test_load_reads_the_full_csv(self, mock_csv_path):
        loader = DataLoader(str(mock_csv_path))
        df = loader.load()
        assert len(df) == 17  # see tests/fixtures/README.md
        assert "Person_ID" in df.columns

    def test_load_with_sample_size_subsamples(self, mock_csv_path):
        loader = DataLoader(str(mock_csv_path))
        df = loader.load(sample_size=5)
        assert len(df) == 5

    def test_load_with_sample_size_larger_than_dataset_returns_full_dataset(self, mock_csv_path):
        loader = DataLoader(str(mock_csv_path))
        df = loader.load(sample_size=10_000)
        assert len(df) == 17

    def test_get_info_reports_existing_file(self, mock_csv_path):
        loader = DataLoader(str(mock_csv_path))
        info = loader.get_info()
        assert info["exists"] is True
        assert info["size_bytes"] > 0
        assert info["path"] == str(mock_csv_path)


class TestHealthDietDataLoader:
    def test_load_validates_and_returns_all_rows(self, mock_csv_path):
        loader = HealthDietDataLoader(str(mock_csv_path))
        df = loader.load()
        assert len(df) == 17
        assert list(df.columns) == HealthDietDataLoader.EXPECTED_COLUMNS

    def test_load_raises_when_a_required_column_is_missing(self, tmp_path, mock_raw_df):
        broken_csv = tmp_path / "missing_column.csv"
        mock_raw_df.drop(columns=["BMI"]).to_csv(broken_csv, index=False)

        loader = HealthDietDataLoader(str(broken_csv))
        with pytest.raises(DataLoadError, match="Missing expected columns"):
            loader.load()

    def test_get_summary_matches_the_mock_dataset(self, mock_csv_path):
        loader = HealthDietDataLoader(str(mock_csv_path))
        summary = loader.get_summary()

        assert summary["total_records"] == 17
        assert summary["total_columns"] == 15
        assert summary["health_status_distribution"] == {
            "Healthy": 9,
            "Overweight": 4,
            "Obese": 2,
            "Underweight": 2,
        }


class TestGetDatasetSummary:
    def test_empty_dataframe_returns_empty_summary_with_expected_columns(self):
        summary = get_dataset_summary(pd.DataFrame())
        assert summary.empty
        assert list(summary.columns) == [
            "Column",
            "Type",
            "Missing",
            "% Missing",
            "Cardinality",
            "Constant",
            "Skewness",
            "% Outliers",
            "Examples",
        ]

    def test_none_input_returns_empty_summary(self):
        summary = get_dataset_summary(None)
        assert summary.empty

    def test_reports_missing_value_count_and_percentage(self):
        df = pd.DataFrame({"a": [1, 2, None, 4]})
        summary = get_dataset_summary(df)
        row = summary[summary["Column"] == "a"].iloc[0]
        assert row["Missing"] == 1
        assert row["% Missing"] == pytest.approx(25.0)

    def test_flags_constant_columns(self):
        df = pd.DataFrame({"constant": [1, 1, 1], "varying": [1, 2, 3]})
        summary = get_dataset_summary(df)
        assert bool(summary.set_index("Column").loc["constant", "Constant"]) is True
        assert bool(summary.set_index("Column").loc["varying", "Constant"]) is False

    def test_skewness_and_outliers_are_none_for_non_numeric_columns(self):
        df = pd.DataFrame({"category": ["a", "b", "c"]})
        summary = get_dataset_summary(df)
        row = summary.iloc[0]
        assert row["Skewness"] is None
        assert row["% Outliers"] is None

    def test_flags_outliers_via_iqr_on_numeric_columns(self):
        # 10 tightly-packed values plus one extreme outlier
        df = pd.DataFrame({"value": [10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 1000]})
        summary = get_dataset_summary(df)
        row = summary.iloc[0]
        assert row["% Outliers"] > 0

    def test_examples_respects_max_examples(self):
        df = pd.DataFrame({"value": list(range(20))})
        summary = get_dataset_summary(df, max_examples=3)
        assert len(summary.iloc[0]["Examples"]) == 3
