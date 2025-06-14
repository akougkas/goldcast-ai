from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class CommonSettings(BaseSettings):
    APP_NAME: str = "Gold Price Forecasting API"
    DEBUG: bool = False
    BASE_DIR: Path = Path(__file__).resolve().parent.parent # src/gold_forecasting
    DATA_DIR: Path = BASE_DIR.parent / "data" # gold-forecasting/data
    RAW_DATA_PATH: Path = DATA_DIR / "raw" / "db_com_capm_gold.csv"
    PROCESSED_DATA_PATH: Path = DATA_DIR / "processed" / "processed_gold_data.csv"
    MODEL_OUTPUT_DIR: Path = BASE_DIR.parent / "models"
    LOG_LEVEL: str = "INFO"

class ServerSettings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000

class ModelSettings(BaseSettings):
    # Example hyperparameters
    PROPHET_SEASONALITY_MODE: str = "multiplicative"
    XGB_N_ESTIMATORS: int = 100
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

class Settings(CommonSettings, ServerSettings, ModelSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

settings = Settings()
