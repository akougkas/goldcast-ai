import pytest
import pandas as pd
from pathlib import Path
from gold_forecasting.data_processing.loader import load_data, generate_data_quality_report
from gold_forecasting.config import settings

@pytest.fixture
def dummy_csv_data_missing_cols(tmp_path: Path) -> Path:
    """Creates a dummy CSV with missing required columns."""
    data = (
        "time,Other_Col\n"  # Missing 'gold' which would become Gold_Price_USD
        "1987-01-01,200.0\n"
    )
    file_path = tmp_path / "dummy_missing_cols.csv"
    file_path.write_text(data)
    return file_path

@pytest.fixture
def dummy_csv_data_nans(tmp_path: Path) -> Path:
    """Creates a dummy CSV with NaN values in key columns."""
    data = (
        "time,gold,Other_Col\n"
        "1987-01-01,,200.0\n"  # NaN for gold
        "1987-01-02,401.0,201.0\n"
    )
    file_path = tmp_path / "dummy_nans.csv"
    file_path.write_text(data)
    return file_path

@pytest.fixture
def dummy_csv_data_short_range(tmp_path: Path) -> Path:
    """Creates a dummy CSV with a date range shorter than expected."""
    data = ("""time,gold,Other_Col\n1990-01-01,500.0,300.0\n1990-01-02,501.0,301.0\n"""
    )
    file_path = tmp_path / "dummy_short_range.csv"
    file_path.write_text(data)
    return file_path

def test_load_data_success():
    """Test successful data loading using the real data file."""
    # This test now relies on 'db_com_capm_gold.csv' being in settings.RAW_DATA_PATH
    df = load_data()
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == 'time'
    assert 'Gold_Price_USD' in df.columns
    # Ensure other columns from the original spec are NOT expected from this CSV
    assert 'SP500_Index' not in df.columns 
    assert 'US_Treasury_10Y_Yield' not in df.columns
    assert 'USD_Index' not in df.columns

def test_load_data_file_not_found():
    """Test FileNotFoundError when CSV does not exist."""
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")

def test_load_data_missing_columns(dummy_csv_data_missing_cols: Path):
    """Test ValueError when required columns are missing."""
    with pytest.raises(ValueError, match=r"Missing required columns: \['Gold_Price_USD'\]"):
        load_data(str(dummy_csv_data_missing_cols))

def test_load_data_handles_nans(dummy_csv_data_nans: Path, caplog):
    """Test that NaNs are loaded and a warning is logged."""
    df = load_data(str(dummy_csv_data_nans))
    assert df['Gold_Price_USD'].isnull().sum() == 1
    assert "nans_found_in_column" in caplog.text
    assert "column=Gold_Price_USD" in caplog.text

@pytest.fixture
def dummy_csv_data_short_range(tmp_path: Path) -> Path:
    """Creates a dummy CSV with a date range shorter than expected."""
    data = ("""time,gold,Other_Col\n1990-01-01,500.0,300.0\n1990-01-02,501.0,301.0\n"""
    )
    file_path = tmp_path / "dummy_short_range.csv"
    file_path.write_text(data)
    return file_path

def test_load_data_date_range_warning(dummy_csv_data_short_range: Path, caplog):
    """Test warning for date range not fully matching expectations."""
    df = load_data(str(dummy_csv_data_short_range))
    assert "unexpected_date_range" in caplog.text
    assert "actual_start='1990-01-01 00:00:00'" in caplog.text
    assert "expected_start='1987-01-01 00:00:00'" in caplog.text

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Creates a sample DataFrame for testing report generation."""
    dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
    data = {
        'Gold_Price_USD': [1500, 1502, None], 
        'SP500_Index': [3000, 3005, 3010]
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_generate_data_quality_report(sample_dataframe: pd.DataFrame):
    """Test the data quality report generation."""
    report = generate_data_quality_report(sample_dataframe)
    assert report["num_rows"] == 3
    assert report["num_cols"] == 2
    assert report["start_date"] == "2020-01-01 00:00:00"
    assert report["end_date"] == "2020-01-03 00:00:00"
    assert report["missing_values_per_column"]['Gold_Price_USD'] == 1
    assert 'Gold_Price_USD' in report["descriptive_stats"]
