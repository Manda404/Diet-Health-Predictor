"""
Integration test: Phase 2 (PreprocessDataUseCase) feeding into
AnalyzeDataDriftUseCase, against the shared mock dataset -- confirms the
seam between preprocessing output and drift analysis works on real encoded
(scaled/one-hot) features, not just synthetic toy DataFrames.
"""

import pytest

from diet_health_predictor.application.data_drift import AnalyzeDataDriftUseCase
from diet_health_predictor.application.feature_engineering import PreprocessDataUseCase
from diet_health_predictor.infrastructure import HealthDietDataLoader

pytestmark = pytest.mark.integration


@pytest.fixture
def preprocessing_result(mock_csv_path, tmp_path):
    loader = HealthDietDataLoader(str(mock_csv_path))
    use_case = PreprocessDataUseCase(
        data_loader=loader,
        output_dir=str(tmp_path / "processed"),
        target_column="Health_Status",
        test_size=0.25,
        random_state=42,
    )
    return use_case.execute()


class TestAnalyzeDataDriftUseCasePipeline:
    def test_analyzes_drift_on_real_preprocessed_train_test_split(self, preprocessing_result):
        use_case = AnalyzeDataDriftUseCase()

        result = use_case.execute(preprocessing_result.X_train, preprocessing_result.X_test)

        assert result.n_features_checked == len(preprocessing_result.feature_names)
        assert set(result.report["feature"]) == set(preprocessing_result.feature_names)
        assert set(result.report["drift_severity"]) <= {"none", "moderate", "major"}

    def test_every_feature_score_is_non_negative(self, preprocessing_result):
        use_case = AnalyzeDataDriftUseCase()

        result = use_case.execute(preprocessing_result.X_train, preprocessing_result.X_test)

        assert (result.report["psi"] >= 0).all()
        assert (result.report["ks_statistic"] >= 0).all()
        assert ((result.report["ks_pvalue"] >= 0) & (result.report["ks_pvalue"] <= 1)).all()
