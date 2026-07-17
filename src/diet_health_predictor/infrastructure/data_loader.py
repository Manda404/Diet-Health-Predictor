"""
Infrastructure Layer - Data Loading Module
============================================

Handles low-level data I/O operations for raw data. This module is responsible for:
- Loading raw data from sources
- Data validation
- Exception handling

Following Clean Architecture principles:
- No business logic here, just I/O operations
- Abstracted data source (could be CSV, database, API, etc.)
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when data loading fails"""

    pass


class DataLoader:
    """
    Responsible for loading data from file system.

    Attributes:
        data_path: Path to the raw data file
        logger: Logger instance
    """

    def __init__(self, data_path: str):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to the CSV file

        Raises:
            FileNotFoundError: If the data file doesn't exist
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.debug(f"DataLoader initialized with path: {self.data_path}")

    def load(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            sample_size: If provided, return only a random sample of this size

        Returns:
            DataFrame with loaded data

        Raises:
            DataLoadError: If loading fails
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

            # Apply sampling if requested
            if sample_size is not None and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"Sampled to {len(df)} rows")

            return df

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise DataLoadError(f"Failed to load data from {self.data_path}: {str(e)}")

    def get_info(self) -> dict:
        """
        Get basic information about the data file without loading all data.

        Returns:
            Dictionary with file metadata
        """
        return {
            "path": str(self.data_path),
            "exists": self.data_path.exists(),
            "size_bytes": self.data_path.stat().st_size if self.data_path.exists() else None,
        }


class HealthDietDataLoader(DataLoader):
    """
    Specialized DataLoader for the health diet dataset.

    Knows about the specific structure and columns of the health diet data.
    """

    EXPECTED_COLUMNS = [
        "Person_ID",
        "Age",
        "Gender",
        "Height_cm",
        "Weight_kg",
        "BMI",
        "Activity_Level",
        "Daily_Calorie_Requirement",
        "Daily_Calorie_Consumed",
        "Protein_Intake_g",
        "Carbohydrate_Intake_g",
        "Fat_Intake_g",
        "Water_Intake_Liters",
        "Diet_Type",
        "Health_Status",
    ]

    def load(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load and validate health diet data.

        Args:
            sample_size: If provided, return only a random sample of this size

        Returns:
            Validated DataFrame

        Raises:
            DataLoadError: If validation fails
        """
        df = super().load(sample_size)
        self._validate_columns(df)
        self._validate_data_types(df)

        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Verify all expected columns are present"""
        missing_columns = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_columns:
            raise DataLoadError(
                f"Missing expected columns: {missing_columns}\n"
                f"Expected: {self.EXPECTED_COLUMNS}\n"
                f"Found: {list(df.columns)}"
            )
        logger.info("✓ All expected columns present")

    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate data types"""
        # Check for numeric columns
        numeric_cols = [
            "Age",
            "Height_cm",
            "Weight_kg",
            "BMI",
            "Daily_Calorie_Requirement",
            "Daily_Calorie_Consumed",
            "Protein_Intake_g",
            "Carbohydrate_Intake_g",
            "Fat_Intake_g",
            "Water_Intake_Liters",
        ]

        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column {col} is not numeric, attempting conversion")
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info("✓ Data types validated")

    def get_summary(self) -> dict:
        """
        Get summary statistics of the dataset without loading all data into memory.

        Returns:
            Dictionary with dataset summary info
        """
        df = self.load()
        return {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "health_status_distribution": df["Health_Status"].value_counts().to_dict(),
            "diet_type_distribution": df["Diet_Type"].value_counts().to_dict(),
            "age_range": f"{df['Age'].min()} - {df['Age'].max()}",
            "bmi_range": f"{df['BMI'].min():.1f} - {df['BMI'].max():.1f}",
        }


def get_dataset_summary(
    df: pd.DataFrame, max_examples: int = 5, outlier_iqr_factor: float = 1.5
) -> pd.DataFrame:
    """
    Provide a detailed dataset summary:
    - Column name
    - Data type
    - Count and percentage of missing values
    - Cardinality (number of unique values) and whether the column is constant
    - Skewness (numeric columns only)
    - Percentage of outlier values via the IQR method (numeric columns only)
    - Representative examples

    Args:
        df: The DataFrame to analyze.
        max_examples: Maximum number of examples to display per column.
        outlier_iqr_factor: IQR multiplier used to flag outliers (default 1.5).

    Returns:
        Summary table of the columns and their characteristics.
    """
    columns = [
        "Column",
        "Type",
        "Missing",
        "% Missing",
        "Cardinality",
        "Constant",
        "Skewness",
        "% Outliers",
        "Examples",
    ]

    if df is None or df.empty:
        logger.warning("No dataset was provided or the dataset is empty.")
        return pd.DataFrame(columns=columns)

    logger.info("Building detailed dataset summary...")
    total_rows = len(df)
    column_details = []

    for col in df.columns:
        col_type = df[col].dtype
        is_numeric = pd.api.types.is_numeric_dtype(col_type)

        # Missing values
        missing_count = df[col].isna().sum()
        missing_pct = round((missing_count / total_rows) * 100, 2)

        # Cardinality
        cardinality = df[col].nunique(dropna=True)
        is_constant = cardinality <= 1

        # Skewness and outliers (numeric columns only)
        skewness = None
        outlier_pct = None
        if is_numeric:
            skewness = round(df[col].skew(), 2)
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - outlier_iqr_factor * iqr, q3 + outlier_iqr_factor * iqr
            outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_pct = round((outlier_count / total_rows) * 100, 2)

        # Representative examples
        unique_values = df[col].dropna().unique()
        if col_type == "object" or col_type.name == "category":
            examples = unique_values[:max_examples]
        else:
            examples = sorted(unique_values[:max_examples])

        column_details.append(
            [
                col,
                col_type,
                missing_count,
                missing_pct,
                cardinality,
                is_constant,
                skewness,
                outlier_pct,
                examples,
            ]
        )

    summary_df = pd.DataFrame(column_details, columns=columns).sort_values(
        by="% Missing", ascending=False
    )

    logger.info(f"Summary complete: {len(summary_df)} columns, {total_rows} rows.")

    return summary_df
