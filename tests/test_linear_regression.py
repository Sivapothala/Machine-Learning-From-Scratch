import numpy as np
import pytest
from src.supervised.linear_regression import LinearRegression
from src.utils.metrics import Metrics

def test_linear_regression_fit():
    # Generate simple linear data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    
    # Create and train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Check if predictions are close to actual values
    assert np.allclose(y_pred, y, rtol=0.1)

def test_linear_regression_metrics():
    # Generate test data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    
    # Create and train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate MSE
    mse = Metrics.mean_squared_error(y, y_pred)
    
    # Check if MSE is small
    assert mse < 0.1 