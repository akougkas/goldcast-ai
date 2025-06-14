import pandas as pd
from ..config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

def load_data(file_path: str = str(settings.RAW_DATA_PATH)) -> pd.DataFrame:
    """Loads gold price data from CSV, parses dates, and sets index."""
    logger.info("loading_data_started", file_path=file_path)
    try:
        df = pd.read_csv(
            file_path,
            parse_dates=['time'],
            index_col='time'
        )
        # Rename columns to match expected names
        df = df.rename(columns={'gold': 'Gold_Price_USD'})

        # Validate required columns
        required_cols = ['Gold_Price_USD']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error("missing_required_columns", missing_columns=missing)
            raise ValueError(f"Missing required columns: {missing}")

        # Validate date range
        expected_start_date = pd.to_datetime("1987-01-01")
        expected_end_date = pd.to_datetime("2025-05-28")
        # Ensure index is DatetimeIndex for min/max operations
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("index_not_datetime", index_type=type(df.index))
            raise TypeError("DataFrame index must be a DatetimeIndex.")

        if not (df.index.min() <= expected_start_date and df.index.max() >= expected_end_date):
             logger.warning("unexpected_date_range", 
                            actual_start=str(df.index.min()), 
                            actual_end=str(df.index.max()),
                            expected_start=str(expected_start_date),
                            expected_end=str(expected_end_date))
        
        # Basic data quality checks (e.g., for NaNs in key columns)
        for col in required_cols:
            if df[col].isnull().any():
                logger.warning("nans_found_in_column", column=col, count=df[col].isnull().sum())
                # For now, log and proceed. More robust handling in preprocessing.

        logger.info("loading_data_completed", num_rows=len(df), columns=list(df.columns))
        return df
    except FileNotFoundError:
        logger.error("data_file_not_found", file_path=file_path)
        raise
    except Exception as e:
        logger.exception("error_loading_data", exc_info=str(e))
        raise

def generate_data_quality_report(df: pd.DataFrame) -> dict:
    """Generates a basic data quality report."""
    report = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "start_date": str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else "N/A",
        "end_date": str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else "N/A",
        "missing_values_per_column": df.isnull().sum().to_dict(),
        "descriptive_stats": df.describe().to_dict()
    }
    logger.info("data_quality_report_generated")
    return report
