"""
Configuration Management Module
================================

Handles loading and validation of configuration from YAML files and environment variables.
Supports multi-environment setup: development, staging, production.

Architecture:
    - Pydantic models for configuration validation
    - YAML file loading
    - Environment variable override support
    - Singleton pattern for settings access
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    """Data layer configuration"""

    raw_data_path: str = Field(default="data/healthy_diet_calorie_intake.csv")
    processed_data_path: str = Field(default="data/processed")
    sample_size: Optional[int] = Field(default=None)
    target_column: str = Field(default="Health_Status")


class ModelConfig(BaseModel):
    """Model configuration"""

    test_size: float = Field(default=0.2)
    random_state: int = Field(default=42)
    models_output_dir: str = Field(default="models")
    # Passed straight to the wrapper constructor (XGBClassifier/CatBoostClassifier
    # kwargs); leave empty to fall back to each wrapper's built-in defaults.
    xgboost_params: dict = Field(default_factory=dict)
    catboost_params: dict = Field(default_factory=dict)
    # Metric HealthDietAPI.select_best_model() uses to pick a winner from
    # compare_models(): one of the keys in ModelTrainingResult.metrics
    # (accuracy, precision_macro, recall_macro, f1_macro, mcc).
    selection_metric: str = Field(default="mcc")


class APIConfig(BaseModel):
    """API configuration"""

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    reload: bool = Field(default=True)


class StorageConfig(BaseModel):
    """Storage and caching configuration"""

    cache_enabled: bool = Field(default=False)
    cache_dir: str = Field(default=".cache")


class Settings(BaseSettings):
    """
    Main Settings class using Pydantic Settings for validation.

    Loads configuration in the following priority:
    1. Environment variables
    2. YAML config file (based on ENVIRONMENT variable)
    3. Default values

    Example:
        settings = Settings()
        print(settings.environment)  # 'development'
        print(settings.data.raw_data_path)  # 'data/healthy_diet_calorie_intake.csv'
    """

    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @classmethod
    def from_yaml(cls, env: Optional[str] = None) -> "Settings":
        """
        Load settings from YAML file.

        Args:
            env: Environment name (dev, staging, prod).
                 If None, reads from ENVIRONMENT env var or defaults to 'development'

        Returns:
            Settings instance with values from YAML file

        Raises:
            FileNotFoundError: If YAML config file doesn't exist
        """
        # Determine environment
        if env is None:
            env = os.getenv("ENVIRONMENT", "development").lower()

        # Map environment names to the on-disk config file suffix
        # (files are named settings.dev/staging/prod.yaml)
        file_suffix_mapping = {
            "dev": "dev",
            "development": "dev",
            "staging": "staging",
            "prod": "prod",
            "production": "prod",
        }
        file_suffix = file_suffix_mapping.get(env, env)

        # Load YAML file (project root's config/ directory)
        config_dir = Path(__file__).parent.parent.parent.parent / "config"
        yaml_file = config_dir / f"settings.{file_suffix}.yaml"

        if not yaml_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {yaml_file}\n"
                f"Available environments: development, staging, production"
            )

        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Create Settings instance from YAML
        return cls(**config_dict)


# Singleton instance - lazy loaded
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance (singleton).

    Loads settings from YAML on first call, subsequent calls return cached instance.

    Returns:
        Global Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml()
    return _settings


# Convenience alias
settings = get_settings()
