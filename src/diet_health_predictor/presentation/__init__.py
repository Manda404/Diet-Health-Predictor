"""
Presentation Layer - API and User Interface
=============================================

The Presentation layer contains:
- Interfaces with the outside world
- Controllers/endpoints
- User-facing functions
- Formatting of responses

This layer depends on Application and Infrastructure but not vice versa.
"""

import logging
from pathlib import Path

from diet_health_predictor.application import (
    AnalyzeHealthStatsUseCase,
    LoadHealthDietDataUseCase,
    PreprocessDataUseCase,
    PreprocessingResult,
)
from diet_health_predictor.config import get_settings
from diet_health_predictor.infrastructure import HealthDietDataLoader

logger = logging.getLogger(__name__)


class HealthDietAPI:
    """
    Main API for interacting with the health diet prediction system.

    This is the entry point for external consumers of the package.
    """

    def __init__(self):
        """Initialize the API with configuration"""
        self.settings = get_settings()
        self.data_path = Path(self.settings.data.raw_data_path)
        logger.info(f"Initialized HealthDietAPI with environment: {self.settings.environment}")

    def load_data(self):
        """
        Load health diet data from configured path.

        Returns:
            List of DietAssessment objects
        """
        logger.info("Loading data via API...")
        loader = HealthDietDataLoader(str(self.data_path))
        use_case = LoadHealthDietDataUseCase(loader)

        return use_case.execute(sample_size=self.settings.data.sample_size)

    def get_health_statistics(self, assessments=None):
        """
        Analyze health statistics.

        Args:
            assessments: List of DietAssessment objects. If None, loads from data.

        Returns:
            Dictionary with statistics
        """
        if assessments is None:
            assessments = self.load_data()

        use_case = AnalyzeHealthStatsUseCase()
        return use_case.execute(assessments)

    def preprocess_data(self) -> PreprocessingResult:
        """
        Run the Phase 2 pipeline: clean, engineer features, split, encode/scale,
        and persist train/test data + fitted transformer to the configured
        processed data path.

        Returns:
            PreprocessingResult with X_train, X_test, y_train, y_test, feature
            names, and the path to the saved transformer.
        """
        logger.info("Preprocessing data via API...")
        loader = HealthDietDataLoader(str(self.data_path))
        use_case = PreprocessDataUseCase(
            data_loader=loader,
            output_dir=self.settings.data.processed_data_path,
            target_column=self.settings.data.target_column,
            test_size=self.settings.model.test_size,
            random_state=self.settings.model.random_state,
        )
        return use_case.execute(sample_size=self.settings.data.sample_size)

    def print_summary(self):
        """Print a summary of the dataset to console"""
        logger.info("Generating summary...")

        try:
            loader = HealthDietDataLoader(str(self.data_path))
            summary = loader.get_summary()

            print("\n" + "=" * 60)
            print("DIET-HEALTH-PREDICTOR - DATA SUMMARY")
            print("=" * 60)
            print(f"Environment: {self.settings.environment.upper()}")
            print(f"Total Records: {summary['total_records']}")
            print(f"Total Columns: {summary['total_columns']}")
            print(f"Age Range: {summary['age_range']} years")
            print(f"BMI Range: {summary['bmi_range']}")
            print("\nHealth Status Distribution:")
            for status, count in summary["health_status_distribution"].items():
                print(f"  - {status}: {count}")
            print("\nDiet Type Distribution:")
            for diet, count in summary["diet_type_distribution"].items():
                print(f"  - {diet}: {count}")
            print("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            print(f"Error: {e}")


def main():
    """Entry point for the package"""
    api = HealthDietAPI()
    api.print_summary()


if __name__ == "__main__":
    main()
