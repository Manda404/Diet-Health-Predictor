"""
Integration test: Phase 2 (PreprocessDataUseCase) feeding straight into
Phase 3 (TrainModelUseCase), one model at a time, against the shared mock
dataset. Each model is trained independently -- there is no "train all 4 and
compare" orchestrator (yet); this test simply drives each of the 4 through
the same preprocessed data to confirm the seam between the two phases works
for every model type.
"""

import numpy as np
import pytest

from diet_health_predictor.application.feature_engineering import PreprocessDataUseCase
from diet_health_predictor.application.model_training import ModelType, TrainModelUseCase
from diet_health_predictor.infrastructure import HealthDietDataLoader

pytestmark = pytest.mark.integration

# Neither wrapper hardcodes hyperparameter defaults (only random_state) -- an
# unconfigured CatBoost falls back to the library's own verbose=1, printing a
# line per iteration. Passed here explicitly to keep test output readable.
_QUIET_KWARGS = {ModelType.CATBOOST: {"verbose": False}}


def build_use_case(model_type, output_dir):
    return TrainModelUseCase(
        model_type=model_type,
        output_dir=str(output_dir),
        random_state=42,
        **_QUIET_KWARGS.get(model_type, {}),
    )


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


@pytest.mark.parametrize("model_type", list(ModelType))
class TestModelTrainingPipeline:
    def test_trains_and_evaluates_on_real_preprocessed_output(
        self, model_type, preprocessing_result, tmp_path
    ):
        use_case = build_use_case(model_type, tmp_path / "models")

        result = use_case.execute(
            preprocessing_result.X_train,
            preprocessing_result.y_train,
            preprocessing_result.X_test,
            preprocessing_result.y_test,
        )

        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert len(result.metrics["confusion_matrix"]) == len(result.metrics["class_labels"])

    def test_persists_artifacts_under_the_model_specific_subdirectory(
        self, model_type, preprocessing_result, tmp_path
    ):
        use_case = build_use_case(model_type, tmp_path / "models")
        use_case.execute(
            preprocessing_result.X_train,
            preprocessing_result.y_train,
            preprocessing_result.X_test,
            preprocessing_result.y_test,
        )

        model_dir = tmp_path / "models" / model_type.value
        assert (model_dir / "model.joblib").exists()
        assert (model_dir / "label_encoder.joblib").exists()

    def test_loaded_model_predicts_the_same_as_the_original(
        self, model_type, preprocessing_result, tmp_path
    ):
        from diet_health_predictor.infrastructure.models.base import BaseModelWrapper

        use_case = build_use_case(model_type, tmp_path / "models")
        result = use_case.execute(
            preprocessing_result.X_train,
            preprocessing_result.y_train,
            preprocessing_result.X_test,
            preprocessing_result.y_test,
        )

        reloaded = BaseModelWrapper.load(result.model_path)
        original_predictions = result.model.predict(preprocessing_result.X_test)
        reloaded_predictions = reloaded.predict(preprocessing_result.X_test)

        np.testing.assert_array_equal(original_predictions, reloaded_predictions)
