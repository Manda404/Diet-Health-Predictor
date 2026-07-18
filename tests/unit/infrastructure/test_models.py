"""
Unit tests for diet_health_predictor.infrastructure.models

Boosting models only (XGBoost, CatBoost). Both wrappers share the same
contract, so most behavior is tested once via parametrization rather than
duplicated per model.

Neither wrapper hardcodes hyperparameter defaults anymore (see
`_build_model()` in each) -- only `random_state`. That means an unconfigured
CatBoostWrapper() falls back to CatBoost's own `verbose=1`, which prints a
line per training iteration; `make_wrapper()` below passes `verbose=False`
explicitly for CatBoost in these tests to keep output readable. This is
caller-supplied configuration at instantiation time, not a wrapper default.
"""

import numpy as np
import pandas as pd
import pytest

from diet_health_predictor.infrastructure.models import CatBoostWrapper, XGBoostWrapper

pytestmark = pytest.mark.unit

WRAPPER_CLASSES = [XGBoostWrapper, CatBoostWrapper]
_QUIET_KWARGS = {XGBoostWrapper: {}, CatBoostWrapper: {"verbose": False}}


def make_wrapper(wrapper_cls, **overrides):
    kwargs = {**_QUIET_KWARGS[wrapper_cls], **overrides}
    return wrapper_cls(**kwargs)


@pytest.fixture
def toy_classification_data():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"feature_a": rng.rand(40), "feature_b": rng.rand(40)})
    y = np.array([0, 1, 2, 3] * 10)
    return X, y


@pytest.fixture
def toy_train_val_data():
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame({"feature_a": rng.rand(40), "feature_b": rng.rand(40)})
    y_train = np.array([0, 1, 2, 3] * 10)
    X_val = pd.DataFrame({"feature_a": rng.rand(12), "feature_b": rng.rand(12)})
    y_val = np.array([0, 1, 2, 3] * 3)
    return X_train, y_train, X_val, y_val


@pytest.mark.parametrize("wrapper_cls", WRAPPER_CLASSES)
class TestBaseModelWrapperContract:
    """Behavior every wrapper must satisfy identically."""

    def test_predict_before_fit_raises(self, wrapper_cls, toy_classification_data):
        X, _ = toy_classification_data
        with pytest.raises(RuntimeError, match="must be fitted"):
            make_wrapper(wrapper_cls).predict(X)

    def test_predict_proba_before_fit_raises(self, wrapper_cls, toy_classification_data):
        X, _ = toy_classification_data
        with pytest.raises(RuntimeError, match="must be fitted"):
            make_wrapper(wrapper_cls).predict_proba(X)

    def test_fit_returns_self(self, wrapper_cls, toy_classification_data):
        X, y = toy_classification_data
        model = make_wrapper(wrapper_cls)
        assert model.fit(X, y) is model

    def test_predict_returns_one_label_per_row(self, wrapper_cls, toy_classification_data):
        X, y = toy_classification_data
        model = make_wrapper(wrapper_cls).fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == (len(X),)

    def test_predict_proba_returns_one_row_per_sample_and_one_column_per_class(
        self, wrapper_cls, toy_classification_data
    ):
        X, y = toy_classification_data
        model = make_wrapper(wrapper_cls).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 4)  # 4 classes in toy_classification_data
        # Every row's predicted probabilities must sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_save_and_load_round_trip(self, wrapper_cls, toy_classification_data, tmp_path):
        X, y = toy_classification_data
        model = make_wrapper(wrapper_cls).fit(X, y)

        save_path = tmp_path / f"{wrapper_cls.__name__}.joblib"
        model.save(str(save_path))
        assert save_path.exists()

        loaded = wrapper_cls.load(str(save_path))
        np.testing.assert_array_equal(loaded.predict(X), model.predict(X))

    def test_hyperparameters_override_defaults(self, wrapper_cls, toy_classification_data):
        model = make_wrapper(wrapper_cls, random_state=7)
        assert model.random_state == 7

    def test_unconfigured_hyperparameters_fall_back_to_the_library_defaults(self, wrapper_cls):
        # No n_estimators/iterations passed -> _build_model() must not have
        # invented one; the underlying library's own default should apply.
        model = make_wrapper(wrapper_cls)
        assert "n_estimators" not in model.hyperparameters
        assert "iterations" not in model.hyperparameters

    def test_get_evals_result_without_a_validation_set_raises(
        self, wrapper_cls, toy_classification_data
    ):
        X, y = toy_classification_data
        model = make_wrapper(wrapper_cls).fit(X, y)  # no X_val/y_val given
        with pytest.raises(RuntimeError, match="without a validation set"):
            model.get_evals_result()

    def test_get_evals_result_before_fit_raises(self, wrapper_cls):
        with pytest.raises(RuntimeError, match="must be fitted"):
            make_wrapper(wrapper_cls).get_evals_result()

    def test_get_evals_result_with_validation_set_tracks_train_and_validation_curves(
        self, wrapper_cls, toy_train_val_data
    ):
        X_train, y_train, X_val, y_val = toy_train_val_data
        model = make_wrapper(wrapper_cls).fit(X_train, y_train, X_val, y_val)

        evals_result = model.get_evals_result()

        assert set(evals_result.keys()) == {"train", "validation"}
        train_curve = next(iter(evals_result["train"].values()))
        validation_curve = next(iter(evals_result["validation"].values()))
        assert len(train_curve) == len(validation_curve) > 0


class TestCatBoostWrapper:
    def test_predict_output_is_flattened_to_1d(self, toy_classification_data):
        # CatBoost's raw .predict() returns shape (n, 1); the wrapper must
        # normalize this to match XGBoostWrapper's 1D output.
        X, y = toy_classification_data
        model = CatBoostWrapper(verbose=False).fit(X, y)
        assert model.predict(X).ndim == 1
