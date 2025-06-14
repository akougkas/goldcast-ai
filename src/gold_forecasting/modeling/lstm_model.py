"""
LSTM model implementation using TensorFlow/Keras.
"""

import tensorflow as tf
import structlog

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple, Dict, Any, Optional

# Initialize logger
logger = structlog.get_logger(__name__)

def create_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences from time series data.

    Args:
        data: Input time series data (scaled).
        sequence_length: Length of each input sequence.

    Returns:
        A tuple containing: 
            - X: Input sequences (features).
            - y: Target values.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0]) # Assuming single feature input for now
        y.append(data[i + sequence_length, 0])
    if not X:
        # If X is empty, ensure it has the correct 2D shape (0, sequence_length)
        return np.empty((0, sequence_length)), np.array(y)
    return np.array(X), np.array(y)

def build_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: int = 50,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Builds a Keras LSTM model.

    Args:
        input_shape: Shape of the input data (sequence_length, n_features).
        lstm_units: Number of units in the LSTM layer.
        dropout_rate: Dropout rate for regularization.
        learning_rate: Learning rate for the Adam optimizer.

    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model

def train_lstm_model(
    series_train: pd.Series, 
    sequence_length: int,
    model_build_params: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Any]] = None
) -> Tuple[Sequential, MinMaxScaler]:
    """
    Trains an LSTM model.

    # --- Begin new/modified code ---
    logger.info("Starting LSTM model training...")

    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Allow memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU(s) available: {gpus}. TensorFlow will attempt to use GPU.")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.error(f"Error during GPU setup: {e}. Proceeding with CPU.")
    else:
        logger.info("No GPU found. TensorFlow will use CPU.")
    # --- End new/modified code ---

    Args:
        series_train: Training time series data (Pandas Series).
        sequence_length: Length of input sequences.
        model_build_params: Parameters for build_lstm_model (lstm_units, dropout_rate, learning_rate).
        fit_params: Parameters for model.fit (epochs, batch_size, verbose, callbacks).

    Returns:
        A tuple containing:
            - model: The trained Keras model.
            - scaler: The MinMaxScaler used for training data.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series_train.values.reshape(-1, 1))

    X_train_seq, y_train_seq = create_sequences(scaled_data, sequence_length)
    
    # Reshape X_train_seq for LSTM input [samples, time_steps, features]
    X_train_seq = np.reshape(X_train_seq, (X_train_seq.shape[0], X_train_seq.shape[1], 1))

    _model_build_params = model_build_params if model_build_params is not None else {}
    model = build_lstm_model(input_shape=(sequence_length, 1), **_model_build_params)

    _fit_params = {
        "epochs": 50,
        "batch_size": 32,
        "verbose": 0, # Default to no verbose logging during training for cleaner output
        "callbacks": [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
    }
    if fit_params is not None:
        _fit_params.update(fit_params)

    model.fit(X_train_seq, y_train_seq, **_fit_params)
    return model, scaler

def predict_with_lstm(
    model: Sequential, 
    series_test: pd.Series, 
    series_train_for_scaling_context: pd.Series, # Used for initial sequence generation
    scaler: MinMaxScaler, 
    sequence_length: int
) -> np.ndarray:
    """
    Makes predictions using a trained LSTM model.

    Args:
        model: Trained Keras LSTM model.
        series_test: Test time series data (Pandas Series).
        series_train_for_scaling_context: Training series data, needed for the initial sequence.
        scaler: The MinMaxScaler used during training.
        sequence_length: Length of input sequences.

    Returns:
        Array of predictions, inverse transformed.
    """
    # Combine last part of training data with test data for context
    full_data = pd.concat([series_train_for_scaling_context, series_test])
    scaled_full_data = scaler.transform(full_data.values.reshape(-1, 1))

    # Prepare test sequences
    inputs = scaled_full_data[len(scaled_full_data) - len(series_test) - sequence_length:]
    X_test_seq, _ = create_sequences(inputs, sequence_length) # y_test_seq is not used here
    X_test_seq = np.reshape(X_test_seq, (X_test_seq.shape[0], X_test_seq.shape[1], 1))

    if X_test_seq.shape[0] == 0:
        return np.array([]) # Not enough data to make predictions

    predicted_scaled = model.predict(X_test_seq)
    predictions = scaler.inverse_transform(predicted_scaled)
    return predictions.flatten()

def evaluate_lstm_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluates the LSTM model performance.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Dictionary of evaluation metrics (RMSE, MAE).
    """
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return {"rmse": float('nan'), "mae": float('nan')} # Or raise error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae}
