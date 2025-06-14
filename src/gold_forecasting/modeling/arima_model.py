"""
Module for ARIMA model specific functions.
"""
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
import structlog

logger = structlog.get_logger(__name__)

def train_arima_model(train_data: pd.Series, order: tuple = (5, 1, 0)) -> ARIMAResultsWrapper:
    """
    Trains an ARIMA model.

    Args:
        train_data (pd.Series): The training data for the model.
        order (tuple): The (p, d, q) order of the ARIMA model.
                       Default is (5, 1, 0), a common starting point.

    Returns:
        ARIMAResultsWrapper: The fitted ARIMA model.
    """
    if not isinstance(train_data, pd.Series):
        logger.error(
            "Training data must be a pandas Series for ARIMA.", data_type=type(train_data)
        )
        raise TypeError("Training data must be a pandas Series for ARIMA.")
    if train_data.empty:
        logger.error("Training data for ARIMA model is empty.")
        raise ValueError("Training data for ARIMA model cannot be empty.")

    # Validate ARIMA order
    if not (isinstance(order, tuple) and len(order) == 3 and all(isinstance(i, int) and i >= 0 for i in order)):
        logger.error(
            "Invalid ARIMA order parameter.",
            order=order,
            expected_format="tuple of 3 non-negative integers (p, d, q)"
        )
        raise ValueError("Invalid ARIMA order parameter. Must be a tuple of 3 non-negative integers (p, d, q).")

    try:
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()
        logger.info("ARIMA model trained successfully.", order=order, data_length=len(train_data))
        return fitted_model
    except ValueError as ve: # Catch specific ValueErrors from statsmodels (e.g., non-stationary)
        logger.error(
            "ValueError during ARIMA model training.", order=order, error=str(ve), exc_info=True
        )
        raise # Re-raise the specific ValueError
    except Exception as e: # Catch other potential exceptions
        logger.error(
            "Failed to train ARIMA model with an unexpected error.", order=order, error=str(e), exc_info=True
        )
        raise # Re-raise the exception

def predict_with_arima(model: ARIMAResultsWrapper, steps: int) -> pd.Series:
    """
    Makes predictions with a trained ARIMA model.

    Args:
        model (ARIMAResultsWrapper): The trained ARIMA model.
        steps (int): The number of steps ahead to forecast.

    Returns:
        pd.Series: The forecasted values.
    """
    if not isinstance(model, ARIMAResultsWrapper):
        logger.error(
            "Provided model is not a valid ARIMAResultsWrapper instance.", model_type=type(model)
        )
        raise TypeError("Model must be an ARIMAResultsWrapper instance.")
    if not isinstance(steps, int) or steps <= 0:
        logger.error("Number of steps must be a positive integer.", steps=steps)
        raise ValueError("Number of steps must be a positive integer.")

    try:
        forecast = model.forecast(steps=steps)
        logger.info("ARIMA model prediction successful.", steps=steps)
        return forecast
    except Exception as e:
        logger.error(
            "Failed to make predictions with ARIMA model.", error=str(e), exc_info=True
        )
        raise # Re-raise
