"""
Infrastructure Layer
====================

Handles all I/O and third-party library concerns:
- `data_loader`: reading and validating the raw CSV dataset
- `preprocessing`: cleaning, feature engineering, encoding/scaling, splitting,
  and persisting processed data

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
    "summarize_dataset",
    "DataCleaner",
    "DataSplitter",
    "FeatureEngineer",
    "FeatureTransformer",
    "ProcessedDataWriter",
]
