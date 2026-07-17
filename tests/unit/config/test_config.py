"""
Unit tests for diet_health_predictor.config

`from_yaml()` resolves the config/ directory relative to this file's own
location (not the process cwd), so these tests work regardless of where
pytest is invoked from.
"""

import pytest

import diet_health_predictor.config as config_module
from diet_health_predictor.config import DataConfig, Settings

pytestmark = pytest.mark.unit


class TestDataConfigDefaults:
    def test_defaults(self):
        data_config = DataConfig()
        assert data_config.raw_data_path == "data/healthy_diet_calorie_intake.csv"
        assert data_config.processed_data_path == "data/processed"
        assert data_config.sample_size is None
        assert data_config.target_column == "Health_Status"


class TestSettingsFromYaml:
    @pytest.mark.parametrize("env_alias", ["dev", "development"])
    def test_dev_aliases_resolve_to_settings_dev_yaml(self, env_alias):
        settings = Settings.from_yaml(env=env_alias)
        assert settings.environment == "development"
        assert settings.debug is True

    @pytest.mark.parametrize("env_alias", ["prod", "production"])
    def test_prod_aliases_resolve_to_settings_prod_yaml(self, env_alias):
        settings = Settings.from_yaml(env=env_alias)
        assert settings.environment == "production"
        assert settings.debug is False

    def test_staging_resolves_to_settings_staging_yaml(self):
        settings = Settings.from_yaml(env="staging")
        assert settings.environment == "staging"

    def test_unknown_environment_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Settings.from_yaml(env="does-not-exist")

    def test_reads_environment_variable_when_env_argument_omitted(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "staging")
        settings = Settings.from_yaml()
        assert settings.environment == "staging"


class TestGetSettingsSingleton:
    def test_returns_the_same_cached_instance_across_calls(self, monkeypatch):
        monkeypatch.setattr(config_module, "_settings", None)
        first = config_module.get_settings()
        second = config_module.get_settings()
        assert first is second
