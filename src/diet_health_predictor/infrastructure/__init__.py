"""
Infrastructure Layer
====================

Handles all I/O and third-party library concerns:
- `data_loader`: reading and validating the raw CSV dataset
- `preprocessing`: cleaning, feature engineering, encoding/scaling, splitting,
  and persisting processed data
- `models`: one wrapper per boosting classifier (XGBoost, CatBoost), all
  sharing the same fit/predict/save/load/get_evals_result contract
- `drift`: PSI / KS-test feature drift detection between two datasets

Following Clean Architecture principles:
- No business logic here, just I/O and library-specific operations
- Abstracted so the Application layer never imports pandas/sklearn/joblib directly
"""

from diet_health_predictor.infrastructure.data_loader import (
    DataLoader,
    DataLoadError,
    HealthDietDataLoader,
    get_dataset_summary,
)
from diet_health_predictor.infrastructure.drift import DriftDetector
from diet_health_predictor.infrastructure.models import (
    BaseModelWrapper,
    CatBoostWrapper,
    XGBoostWrapper,
)
from diet_health_predictor.infrastructure.preprocessing import (
    DataCleaner,
    DataSplitter,
    FeatureEngineer,
    FeatureTransformer,
    ProcessedDataWriter,
)

__all__ = [
    "DataLoadError",
    "DataLoader",
    "HealthDietDataLoader",
    "get_dataset_summary",
    "DataCleaner",
    "DataSplitter",
    "FeatureEngineer",
    "FeatureTransformer",
    "ProcessedDataWriter",
    "BaseModelWrapper",
    "XGBoostWrapper",
    "CatBoostWrapper",
    "DriftDetector",
]
