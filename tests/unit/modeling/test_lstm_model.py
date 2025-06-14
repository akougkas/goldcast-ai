"""
Unit tests for the LSTM model functions.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, call
from sklearn.preprocessing import MinMaxScaler

# TensorFlow components will be mocked directly in test functions where needed.
from gold_forecasting.modeling.lstm_model import (
    create_sequences,
    build_lstm_model,
    train_lstm_model,
    predict_with_lstm,
    evaluate_lstm_model,
)

@pytest.fixture
def sample_series_data() -> pd.Series:
    """Returns a sample Pandas Series for testing."""
    return pd.Series(np.arange(1, 21, dtype=float))

@pytest.fixture
def sample_scaled_data() -> np.ndarray:
    """Returns a sample scaled numpy array (e.g., from MinMaxScaler)."""
    return np.linspace(0, 1, 20).reshape(-1, 1)


# --- Test create_sequences --- 

def test_create_sequences_correct_shape_and_values(sample_scaled_data: np.ndarray):
    """Test if sequences are created with correct shapes and values."""
    data = sample_scaled_data
    sequence_length = 3
    X, y = create_sequences(data, sequence_length)

    assert X.shape == (len(data) - sequence_length, sequence_length)
    assert y.shape == (len(data) - sequence_length,)
    
    # Check first sequence
    assert np.array_equal(X[0], data[0:3, 0])
    assert y[0] == data[3, 0]
    
    # Check last sequence
    assert np.array_equal(X[-1], data[len(data)-sequence_length-1 : len(data)-1, 0])
    assert y[-1] == data[len(data)-1, 0]

def test_create_sequences_empty_if_not_enough_data(sample_scaled_data: np.ndarray):
    """Test returns empty arrays if data length is less than sequence_length."""
    data = sample_scaled_data[:2] # Data length 2
    sequence_length = 3
    X, y = create_sequences(data, sequence_length)
    assert X.shape == (0, sequence_length) # Expect shape (0, 3) for X
    assert y.shape == (0,)


# --- Test build_lstm_model --- 

@patch('gold_forecasting.modeling.lstm_model.Sequential')
@patch('gold_forecasting.modeling.lstm_model.LSTM')
@patch('gold_forecasting.modeling.lstm_model.Dropout')
@patch('gold_forecasting.modeling.lstm_model.Dense')
@patch('gold_forecasting.modeling.lstm_model.Adam')
def test_build_lstm_model(MockAdam: MagicMock, MockDense: MagicMock, MockDropout: MagicMock, MockLSTM: MagicMock, MockSequential: MagicMock):
    """Test the LSTM model architecture and compilation."""
    mock_model_instance = MagicMock(spec=["add", "compile"])
    MockSequential.return_value = mock_model_instance
    MockAdam.return_value = "mock_adam_optimizer"

    input_shape = (5, 1) # sequence_length=5, n_features=1
    lstm_units = 64
    dropout_rate = 0.3
    learning_rate = 0.005

    returned_model = build_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate)

    MockSequential.assert_called_once()
    assert returned_model == mock_model_instance
    
    # Check layers are added
    calls = mock_model_instance.add.call_args_list
    assert len(calls) == 3
    MockLSTM.assert_called_once_with(units=lstm_units, return_sequences=False, input_shape=input_shape)
    MockDropout.assert_called_once_with(dropout_rate)
    MockDense.assert_called_once_with(units=1)
    
    # Check compilation
    MockAdam.assert_called_once_with(learning_rate=learning_rate)
    mock_model_instance.compile.assert_called_once_with(optimizer="mock_adam_optimizer", loss="mean_squared_error")


# --- Test train_lstm_model --- 

@patch('gold_forecasting.modeling.lstm_model.MinMaxScaler')
@patch('gold_forecasting.modeling.lstm_model.create_sequences')
@patch('gold_forecasting.modeling.lstm_model.build_lstm_model')
@patch('gold_forecasting.modeling.lstm_model.EarlyStopping') # Patch EarlyStopping directly
def test_train_lstm_model(MockEarlyStopping: MagicMock, mock_build_model: MagicMock, mock_create_seq: MagicMock, mock_scaler_cls: MagicMock, sample_series_data: pd.Series):
    """Test the training process flow and calls to sub-functions."""
    # Setup mocks
    mock_scaler_instance = MagicMock()
    mock_scaler_instance.fit_transform.return_value = np.random.rand(len(sample_series_data), 1)
    mock_scaler_cls.return_value = mock_scaler_instance

    X_train_seq_mock = np.random.rand(10, 5) # 10 samples, 5 timesteps
    y_train_seq_mock = np.random.rand(10)
    mock_create_seq.return_value = (X_train_seq_mock, y_train_seq_mock)

    mock_model_instance = MagicMock(spec_set=['fit']) # Use spec_set for stricter mocking
    mock_build_model.return_value = mock_model_instance
    MockEarlyStopping.return_value = "mock_early_stopping_instance"

    sequence_length = 5
    model_build_params = {"lstm_units": 30, "dropout_rate": 0.1}
    fit_params = {"epochs": 10, "batch_size": 16, "verbose": 1}

    trained_model, scaler = train_lstm_model(sample_series_data, sequence_length, model_build_params, fit_params)

    mock_scaler_cls.assert_called_once_with(feature_range=(0, 1))
    mock_scaler_instance.fit_transform.assert_called_once()
    # Get the actual data passed to create_sequences
    actual_scaled_data_for_seq = mock_scaler_instance.fit_transform.return_value
    mock_create_seq.assert_called_once_with(actual_scaled_data_for_seq, sequence_length)
    
    expected_X_train_reshaped = np.reshape(X_train_seq_mock, (X_train_seq_mock.shape[0], X_train_seq_mock.shape[1], 1))
    mock_build_model.assert_called_once_with(input_shape=(sequence_length, 1), **model_build_params)
    
    MockEarlyStopping.assert_called_once_with(monitor='loss', patience=5, restore_best_weights=True)
    
    # Check model.fit call
    args, kwargs = mock_model_instance.fit.call_args
    assert np.array_equal(args[0], expected_X_train_reshaped)
    assert np.array_equal(args[1], y_train_seq_mock)
    assert kwargs['epochs'] == fit_params['epochs']
    assert kwargs['batch_size'] == fit_params['batch_size']
    assert kwargs['verbose'] == fit_params['verbose']
    assert kwargs['callbacks'] == ["mock_early_stopping_instance"]

    assert trained_model == mock_model_instance
    assert scaler == mock_scaler_instance


# --- Test predict_with_lstm --- 

@patch('gold_forecasting.modeling.lstm_model.create_sequences')
def test_predict_with_lstm(mock_create_seq: MagicMock, sample_series_data: pd.Series):
    """Test the prediction flow."""
    mock_model = MagicMock(spec=["predict"])
    mock_scaler = MagicMock(spec=["transform", "inverse_transform"])
    sequence_length = 3

    series_train_context = sample_series_data[:10]
    series_test = sample_series_data[10:]

    # Mock scaler behavior
    scaled_full_data_mock = np.linspace(0,1, len(sample_series_data)).reshape(-1,1)
    mock_scaler.transform.return_value = scaled_full_data_mock
    
    # Mock create_sequences for test data
    # Inputs for create_sequences will be the tail of scaled_full_data_mock
    expected_input_for_create_seq = scaled_full_data_mock[len(scaled_full_data_mock) - len(series_test) - sequence_length:]
    X_test_seq_mock = np.random.rand(len(series_test), sequence_length) # Shape based on test series length
    mock_create_seq.return_value = (X_test_seq_mock, np.array([])) # y is not used

    # Mock model.predict behavior
    predicted_scaled_mock = np.random.rand(len(series_test), 1)
    mock_model.predict.return_value = predicted_scaled_mock

    # Mock scaler.inverse_transform behavior
    final_predictions_mock = np.random.rand(len(series_test))
    mock_scaler.inverse_transform.return_value = final_predictions_mock.reshape(-1,1)

    predictions = predict_with_lstm(mock_model, series_test, series_train_context, mock_scaler, sequence_length)

    # Assert scaler.transform was called correctly
    # We need to check the argument passed to transform, which is series_train_context + series_test
    # The argument to scaler.transform is full_data.values.reshape(-1, 1)
    expected_transform_arg = pd.concat([series_train_context, series_test]).values.reshape(-1,1)
    actual_transform_arg = mock_scaler.transform.call_args[0][0]
    np.testing.assert_array_equal(actual_transform_arg, expected_transform_arg)
    
    # Assert create_sequences was called correctly
    # This requires careful construction of the expected input based on mocks
    assert np.array_equal(mock_create_seq.call_args[0][0], expected_input_for_create_seq)
    assert mock_create_seq.call_args[0][1] == sequence_length

    # Assert model.predict was called with reshaped X_test_seq
    expected_X_test_reshaped = np.reshape(X_test_seq_mock, (X_test_seq_mock.shape[0], X_test_seq_mock.shape[1], 1))
    assert np.array_equal(mock_model.predict.call_args[0][0], expected_X_test_reshaped)

    # Assert scaler.inverse_transform was called
    assert np.array_equal(mock_scaler.inverse_transform.call_args[0][0], predicted_scaled_mock)

    assert np.array_equal(predictions, final_predictions_mock)

def test_predict_with_lstm_not_enough_data(sample_series_data: pd.Series):
    """Test prediction returns empty array if not enough data to form a sequence."""
    mock_model = MagicMock()
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[0.1],[0.2]]) # Scaled data too short
    sequence_length = 3

    series_train_context = sample_series_data[:1]
    series_test = sample_series_data[1:2] # Only one test point
    
    # When create_sequences is called with insufficient data, it will return X with shape (0, seq_len)
    with patch('gold_forecasting.modeling.lstm_model.create_sequences', return_value=(np.empty((0,sequence_length)), np.array([])) ) as mock_create_seq_short:
        predictions = predict_with_lstm(mock_model, series_test, series_train_context, mock_scaler, sequence_length)
        assert predictions.shape == (0,)
        mock_create_seq_short.assert_called_once()
        mock_model.predict.assert_not_called() # predict should not be called if X_test_seq is empty

# --- Test evaluate_lstm_model --- 

def test_evaluate_lstm_model_valid_input():
    """Test evaluation with valid inputs."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    
    expected_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    expected_mae = np.mean(np.abs(y_true - y_pred))
    
    evaluation = evaluate_lstm_model(y_true, y_pred)
    assert "rmse" in evaluation
    assert "mae" in evaluation
    assert np.isclose(evaluation["rmse"], expected_rmse)
    assert np.isclose(evaluation["mae"], expected_mae)

def test_evaluate_lstm_model_empty_input():
    """Test evaluation with empty inputs returns NaN."""
    evaluation = evaluate_lstm_model(np.array([]), np.array([]))
    assert np.isnan(evaluation["rmse"])
    assert np.isnan(evaluation["mae"])

def test_evaluate_lstm_model_mismatched_lengths():
    """Test evaluation with mismatched length inputs returns NaN."""
    evaluation = evaluate_lstm_model(np.array([1,2]), np.array([1]))
    assert np.isnan(evaluation["rmse"])
    assert np.isnan(evaluation["mae"])
