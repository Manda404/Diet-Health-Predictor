"""Unit tests for diet_health_predictor.infrastructure.preprocessing"""

import numpy as np
import pandas as pd
import pytest

from diet_health_predictor.infrastructure.preprocessing import (
    DataCleaner,
    DataSplitter,
    FeatureEngineer,
    FeatureTransformer,
    ProcessedDataWriter,
)

pytestmark = pytest.mark.unit


class TestDataCleaner:
    """
    `fit()`/`transform()` mirror `FeatureTransformer`: statistics (median/mode,
    IQR bounds) are learned from whatever is passed to `fit()` and applied to
    whatever is passed to `transform()` -- deliberately allowing a caller to
    fit on train and transform a *different* test set, unlike the old
    single-shot `clean()` this replaced.
    """

    def test_drop_duplicates_removes_duplicate_person_ids(self, mock_raw_df):
        deduped = DataCleaner().drop_duplicates(mock_raw_df)
        assert len(deduped) == 16  # 17 rows, 1 exact duplicate of P0001
        assert deduped["Person_ID"].duplicated().sum() == 0

    def test_transform_before_fit_raises(self, mock_raw_df):
        with pytest.raises(RuntimeError, match="must be fitted"):
            DataCleaner().transform(mock_raw_df)

    def test_fit_transform_imputes_all_missing_values(self, mock_raw_df):
        assert mock_raw_df.isna().sum().sum() > 0  # sanity check the fixture actually has gaps
        cleaner = DataCleaner()
        cleaned = cleaner.fit_transform(cleaner.drop_duplicates(mock_raw_df))
        assert cleaned.isna().sum().sum() == 0

    def test_imputes_numeric_column_with_median(self):
        df = pd.DataFrame({"Person_ID": ["A", "B", "C"], "value": [10.0, 20.0, np.nan]})
        cleaned = DataCleaner().fit_transform(df)
        assert cleaned.loc[cleaned["Person_ID"] == "C", "value"].iloc[0] == 15.0

    def test_imputes_categorical_column_with_mode(self):
        df = pd.DataFrame({"Person_ID": ["A", "B", "C", "D"], "category": ["x", "x", "y", None]})
        cleaned = DataCleaner().fit_transform(df)
        assert cleaned.loc[cleaned["Person_ID"] == "D", "category"].iloc[0] == "x"

    def test_clips_outliers_to_iqr_bounds(self):
        df = pd.DataFrame(
            {
                "Person_ID": [f"P{i}" for i in range(11)],
                "value": [10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 1000],
            }
        )
        cleaner = DataCleaner(outlier_columns=["value"])
        cleaned = cleaner.fit_transform(df)
        assert cleaned["value"].max() < 1000

    def test_columns_not_listed_as_outlier_targets_are_left_untouched(self):
        df = pd.DataFrame({"Person_ID": [f"P{i}" for i in range(5)], "value": [1, 2, 3, 4, 1000]})
        cleaner = DataCleaner(outlier_columns=[])  # nothing targeted
        cleaned = cleaner.fit_transform(df)
        assert cleaned["value"].max() == 1000

    def test_fit_on_train_applies_train_derived_median_to_test(self):
        """
        The whole point of fit()/transform(): a test row's own missing value
        is filled with the *training* set's median, not one computed from
        the test row's own (single-row, undefined) distribution.
        """
        train_df = pd.DataFrame({"Person_ID": ["A", "B", "C"], "value": [10.0, 20.0, 30.0]})
        test_df = pd.DataFrame({"Person_ID": ["D"], "value": [np.nan]})

        cleaner = DataCleaner().fit(train_df)
        cleaned_test = cleaner.transform(test_df)

        assert cleaned_test["value"].iloc[0] == 20.0  # median of the training set

    def test_fit_on_train_applies_train_derived_outlier_bounds_to_test(self):
        train_df = pd.DataFrame(
            {
                "Person_ID": [f"P{i}" for i in range(10)],
                "value": [10, 11, 9, 10, 11, 9, 10, 11, 9, 10],
            }
        )
        test_df = pd.DataFrame({"Person_ID": ["T1"], "value": [1000]})

        cleaner = DataCleaner(outlier_columns=["value"]).fit(train_df)
        cleaned_test = cleaner.transform(test_df)

        assert cleaned_test["value"].iloc[0] < 1000

    def test_transform_skips_columns_absent_from_the_incoming_frame(self):
        # A single-record prediction-time DataFrame (Phase 4) won't carry
        # Person_ID/Health_Status, even though those were present at fit()
        # time -- transform() must not require them.
        train_df = pd.DataFrame(
            {
                "Person_ID": ["A", "B", "C"],
                "Health_Status": ["Healthy", "Obese", "Healthy"],
                "value": [10.0, 20.0, 30.0],
            }
        )
        record = pd.DataFrame({"value": [np.nan]})

        cleaner = DataCleaner().fit(train_df)
        transformed = cleaner.transform(record)

        assert transformed["value"].iloc[0] == 20.0  # training median
        assert list(transformed.columns) == ["value"]

    def test_save_and_load_round_trip_preserves_fitted_state(self, tmp_path):
        train_df = pd.DataFrame({"Person_ID": ["A", "B", "C"], "value": [10.0, 20.0, np.nan]})
        cleaner = DataCleaner(outlier_columns=["value"]).fit(train_df)

        path = str(tmp_path / "data_cleaner.joblib")
        cleaner.save(path)
        reloaded = DataCleaner.load(path)

        test_df = pd.DataFrame({"Person_ID": ["D"], "value": [np.nan]})
        pd.testing.assert_frame_equal(cleaner.transform(test_df), reloaded.transform(test_df))


class TestFeatureEngineer:
    """
    Formula correctness is asserted against independently-computed expected
    values on a single hand-picked row, rather than against the shared mock
    dataset — see tests/conftest.py docstring for the rationale.
    """

    @pytest.fixture
    def single_row_df(self):
        return pd.DataFrame(
            [
                {
                    "Person_ID": "P0001",
                    "Age": 30,
                    "Gender": "Male",
                    "Height_cm": 180.0,
                    "Weight_kg": 80.0,
                    "BMI": 24.7,
                    "Activity_Level": "Moderately Active",
                    "Daily_Calorie_Requirement": 2500.0,
                    "Daily_Calorie_Consumed": 2300.0,
                    "Protein_Intake_g": 150.0,
                    "Carbohydrate_Intake_g": 250.0,
                    "Fat_Intake_g": 70.0,
                    "Water_Intake_Liters": 2.5,
                    "Diet_Type": "Balanced",
                    "Health_Status": "Healthy",
                }
            ]
        )

    def test_adds_all_14_engineered_columns(self, single_row_df):
        engineered = FeatureEngineer().transform(single_row_df)
        new_columns = set(engineered.columns) - set(single_row_df.columns)
        assert new_columns == {
            "Age_Group",
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
        }

    def test_calorie_balance_and_deviation(self, single_row_df):
        engineered = FeatureEngineer().transform(single_row_df)
        row = engineered.iloc[0]
        assert row["Calorie_Balance"] == pytest.approx(2300.0 - 2500.0)
        assert row["Calorie_Deviation_Pct"] == pytest.approx((-200.0 / 2500.0) * 100)

    def test_macro_ratios_sum_to_one(self, single_row_df):
        row = FeatureEngineer().transform(single_row_df).iloc[0]
        assert row["Protein_Ratio"] + row["Carb_Ratio"] + row["Fat_Ratio"] == pytest.approx(1.0)

    def test_bmr_uses_male_offset(self, single_row_df):
        row = FeatureEngineer().transform(single_row_df).iloc[0]
        expected = 10 * 80.0 + 6.25 * 180.0 - 5 * 30 + 5
        assert row["BMR"] == pytest.approx(expected)

    def test_ideal_weight_and_deviation(self, single_row_df):
        row = FeatureEngineer().transform(single_row_df).iloc[0]
        height_inches = 180.0 / 2.54
        expected_ideal = 50 + 2.3 * (height_inches - 60)
        assert row["Ideal_Weight_kg"] == pytest.approx(expected_ideal)
        assert row["Weight_Deviation_kg"] == pytest.approx(80.0 - expected_ideal)

    def test_protein_per_kg_and_water_per_kg(self, single_row_df):
        row = FeatureEngineer().transform(single_row_df).iloc[0]
        assert row["Protein_per_kg_Bodyweight"] == pytest.approx(150.0 / 80.0)
        assert row["Water_Intake_ml_per_kg"] == pytest.approx(2.5 * 1000 / 80.0)

    def test_adequate_water_intake_is_binary(self, single_row_df):
        row = FeatureEngineer().transform(single_row_df).iloc[0]
        assert row["Adequate_Water_Intake"] == 1

    @pytest.mark.parametrize(
        "age, expected_group",
        [
            (17, "Child"),
            (18, "Young Adult"),
            (29, "Young Adult"),
            (30, "Adult"),
            (59, "Adult"),
            (60, "Senior"),
        ],
    )
    def test_age_group_boundaries(self, single_row_df, age, expected_group):
        single_row_df.loc[0, "Age"] = age
        row = FeatureEngineer().transform(single_row_df).iloc[0]
        assert row["Age_Group"] == expected_group

    def test_other_gender_uses_averaged_bmr_and_ideal_weight_offsets(self, single_row_df):
        single_row_df.loc[0, "Gender"] = "Other"
        row = FeatureEngineer().transform(single_row_df).iloc[0]

        base = 10 * 80.0 + 6.25 * 180.0 - 5 * 30
        expected_bmr = base + (5 + -161) / 2
        assert row["BMR"] == pytest.approx(expected_bmr)

        height_inches = 180.0 / 2.54
        expected_ideal = (50 + 45.5) / 2 + 2.3 * (height_inches - 60)
        assert row["Ideal_Weight_kg"] == pytest.approx(expected_ideal)

    def test_does_not_mutate_the_input_dataframe(self, single_row_df):
        original_columns = list(single_row_df.columns)
        FeatureEngineer().transform(single_row_df)
        assert list(single_row_df.columns) == original_columns


class TestFeatureTransformer:
    @pytest.fixture
    def small_df(self):
        return pd.DataFrame(
            {
                "num_a": [1.0, 2.0, 3.0, 4.0],
                "num_b": [10.0, 20.0, 30.0, 40.0],
                "cat_a": ["x", "y", "x", "y"],
            }
        )

    def test_fit_transform_returns_expected_shape(self, small_df):
        transformer = FeatureTransformer(
            numeric_features=["num_a", "num_b"], categorical_features=["cat_a"]
        )
        result = transformer.fit_transform(small_df)
        # 2 scaled numeric columns + 2 one-hot columns (x, y)
        assert result.shape == (4, 4)

    def test_transform_before_fit_raises(self, small_df):
        transformer = FeatureTransformer(numeric_features=["num_a"], categorical_features=["cat_a"])
        with pytest.raises(RuntimeError):
            transformer.transform(small_df)

    def test_unseen_category_at_transform_time_is_ignored_not_raised(self, small_df):
        transformer = FeatureTransformer(
            numeric_features=["num_a", "num_b"], categorical_features=["cat_a"]
        )
        transformer.fit(small_df)

        unseen = pd.DataFrame({"num_a": [5.0], "num_b": [50.0], "cat_a": ["z"]})
        result = transformer.transform(unseen)  # should not raise despite "z" never seen in fit
        assert len(result) == 1

    def test_save_and_load_round_trip(self, small_df, tmp_path):
        transformer = FeatureTransformer(
            numeric_features=["num_a", "num_b"], categorical_features=["cat_a"]
        )
        transformer.fit(small_df)

        save_path = tmp_path / "transformer.joblib"
        transformer.save(str(save_path))
        assert save_path.exists()

        loaded = FeatureTransformer.load(str(save_path))
        pd.testing.assert_frame_equal(loaded.transform(small_df), transformer.transform(small_df))


class TestDataSplitter:
    def test_split_sizes_add_up_to_the_input(self, mock_raw_df):
        cleaner = DataCleaner()
        cleaned = cleaner.fit_transform(cleaner.drop_duplicates(mock_raw_df))
        train_df, test_df = DataSplitter().split(
            cleaned, target_column="Health_Status", test_size=0.25, random_state=42
        )
        assert len(train_df) + len(test_df) == len(cleaned)

    def test_split_has_no_overlapping_person_ids(self, mock_raw_df):
        cleaner = DataCleaner()
        cleaned = cleaner.fit_transform(cleaner.drop_duplicates(mock_raw_df))
        train_df, test_df = DataSplitter().split(
            cleaned, target_column="Health_Status", test_size=0.25, random_state=42
        )
        assert set(train_df["Person_ID"]) & set(test_df["Person_ID"]) == set()

    def test_split_is_reproducible_with_the_same_random_state(self, mock_raw_df):
        cleaner = DataCleaner()
        cleaned = cleaner.fit_transform(cleaner.drop_duplicates(mock_raw_df))
        train_1, test_1 = DataSplitter().split(cleaned, "Health_Status", 0.25, random_state=7)
        train_2, test_2 = DataSplitter().split(cleaned, "Health_Status", 0.25, random_state=7)
        pd.testing.assert_frame_equal(train_1, train_2)
        pd.testing.assert_frame_equal(test_1, test_2)


class TestProcessedDataWriter:
    def test_write_features_creates_expected_files(self, tmp_path):
        writer = ProcessedDataWriter(str(tmp_path))
        X_train = pd.DataFrame({"a": [1, 2]})
        X_test = pd.DataFrame({"a": [3]})

        writer.write_features(X_train, X_test)

        assert (tmp_path / "X_train.csv").exists()
        assert (tmp_path / "X_test.csv").exists()

    def test_write_targets_creates_expected_files(self, tmp_path):
        writer = ProcessedDataWriter(str(tmp_path))
        writer.write_targets(pd.Series(["Healthy", "Obese"]), pd.Series(["Healthy"]))

        assert (tmp_path / "y_train.csv").exists()
        assert (tmp_path / "y_test.csv").exists()

    def test_transformer_path_is_under_output_dir(self, tmp_path):
        writer = ProcessedDataWriter(str(tmp_path))
        assert writer.transformer_path() == str(tmp_path / "feature_transformer.joblib")

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "does" / "not" / "exist"
        writer = ProcessedDataWriter(str(nested))
        writer.write_features(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]}))
        assert nested.exists()
