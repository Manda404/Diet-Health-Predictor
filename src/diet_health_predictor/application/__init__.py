"""
Application Layer - Use Cases and Business Logic
================================================

The Application layer contains:
- `use_cases`: loading raw data into domain objects, analyzing health statistics
- `feature_engineering`: the Phase 2 preprocessing/feature engineering pipeline

This layer depends on Domain and Infrastructure but orchestrates them.
"""

from diet_health_predictor.application.feature_engineering import (
    PreprocessDataUseCase,
    PreprocessingResult,
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
]
