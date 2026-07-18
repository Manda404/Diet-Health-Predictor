"""
Application Layer - Use Cases and Business Logic
================================================

The Application layer contains:
- `use_cases`: loading raw data into domain objects, analyzing health statistics
- `feature_engineering`: the Phase 2 preprocessing/feature engineering pipeline
- `model_training`: the Phase 3 per-model training/evaluation use case
- `model_evaluation`: the Phase 3 cross-validation and model comparison use cases
- `data_drift`: train-vs-test feature drift analysis (PSI + KS test)
- `prediction`: Phase 4 single-record prediction use case

This layer depends on Domain and Infrastructure but orchestrates them.
"""

from diet_health_predictor.application.data_drift import AnalyzeDataDriftUseCase, DataDriftResult
from diet_health_predictor.application.feature_engineering import (
    PreprocessDataUseCase,
    PreprocessingResult,
)
from diet_health_predictor.application.model_evaluation import (
    CompareModelsUseCase,
    CrossValidateModelUseCase,
    CrossValidationResult,
    best_model,
    save_best_model,
)
from diet_health_predictor.application.model_training import (
    MODEL_WRAPPER_REGISTRY,
    ModelTrainingResult,
    ModelType,
    TrainModelUseCase,
)
from diet_health_predictor.application.prediction import (
    PredictHealthStatusUseCase,
    PredictionResult,
)
from diet_health_predictor.application.use_cases import (
    AnalyzeHealthStatsUseCase,
    LoadHealthDietDataUseCase,
)

__all__ = [
    "LoadHealthDietDataUseCase",
    "AnalyzeHealthStatsUseCase",
    "PreprocessDataUseCase",
    "PreprocessingResult",
    "ModelType",
    "TrainModelUseCase",
    "ModelTrainingResult",
    "MODEL_WRAPPER_REGISTRY",
    "CrossValidateModelUseCase",
    "CrossValidationResult",
    "CompareModelsUseCase",
    "best_model",
    "save_best_model",
    "AnalyzeDataDriftUseCase",
    "DataDriftResult",
    "PredictHealthStatusUseCase",
    "PredictionResult",
]
