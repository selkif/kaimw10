import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def check_stationarity(series):
    """
    Perform ADF test to check stationarity.
    
    Args:
        series (pd.Series): The time series data.
        
    Returns:
        bool: True if stationary, False otherwise.
    """
    adf_result = adfuller(series.dropna())
    return adf_result[1] < 0.05  # p-value < 0.05 means stationary

def define_models(arima_model, linear_model, rf_model):
    models = {
        'ARIMA': arima_model,
        'Linear Regression': linear_model,
        'Random Forest': rf_model
    }

    return models

def build_arima_model(df, order=(1, 1, 1)):
    """
    Build and fit an ARIMA model.
    
    Args:
        df (pd.DataFrame): DataFrame containing the oil prices.
        order (tuple): Order of the ARIMA model.
        
    Returns:
        ARIMA: Fitted ARIMA model.
    """
    model = ARIMA(df['Price'], order=order)
    model_fit = model.fit()
    return model_fit

def build_sarima_model(df, order, seasonal_order):
    """
    Build a SARIMA model.
    
    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        order (tuple): ARIMA order (p, d, q).
        seasonal_order (tuple): Seasonal order (P, D, Q, s).
    
    Returns:
        SARIMAX: Fitted SARIMA model.
    """
    model = SARIMAX(df['Price'], order=order, seasonal_order=seasonal_order)
    results = model.fit()
    return results

def build_linear_regression_model(df):
    """
    Build a linear regression model.
    
    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        
    Returns:
        sm.OLS: Fitted linear regression model.
    """
    # Convert the Date column to a numerical format (e.g., timestamp)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_num'] = df['Date'].map(pd.Timestamp.timestamp)
    
    # Create lagged variable
    df['Lagged_Price'] = df['Price'].shift(1)  # Shift by 1 to create lag
    
    # Drop NaN values resulting from lagging
    df = df.dropna()
    
    # Define independent and dependent variables
    X = df[['Date_num', 'Lagged_Price']]
    y = df['Price']
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    return model

def build_random_forest_model(df):
    """
    Build a Random Forest regression model.
    
    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
    
    Returns:
        RandomForestRegressor: Fitted Random Forest model.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_num'] = df['Date'].map(pd.Timestamp.timestamp)
    df['Lagged_Price'] = df['Price'].shift(1)
    df = df.dropna()
    
    X = df[['Date_num', 'Lagged_Price']]  # Use the same features as the linear model
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

# Function to compare multiple models
def compare_models(models, df, forecast_steps=30):
    """
    Compare multiple models based on performance metrics.
    
    Args:
        models (dict): A dictionary where keys are model names and values are fitted models.
        df (pd.DataFrame): DataFrame containing the time series data.
        forecast_steps (int): Number of steps to forecast.
        
    Returns:
        pd.DataFrame: DataFrame containing performance metrics for each model.
    """
    results = {}
    
    for name, model in models.items():
        metrics = evaluate_model(model, df, forecast_steps)
        results[name] = metrics
    
    return pd.DataFrame(results).T  # Transpose for better readability

def evaluate_model(model, df, forecast_steps=30):
    """
    Evaluate the ARIMA model's performance using RMSE, MAE, and R-squared.
    
    Args:
        model: Fitted ARIMA model.
        df (pd.DataFrame): DataFrame containing the oil prices.
        forecast_steps (int): Number of steps to forecast.
        
    Returns:
        dict: Dictionary containing RMSE, MAE, and R-squared.
    """
    # Ensure that the necessary features are available
    df['Date_num'] = df['Date'].map(pd.Timestamp.timestamp)
    df['Lagged_Price'] = df['Price'].shift(1)
    df = df.dropna()
    
    # Generate predictions
    if hasattr(model, 'forecast'):
        # Time series models (like ARIMA)
        predictions = model.forecast(steps=forecast_steps)
    else:
        # For regression models, predict using the last available features
        X_last = df[['Date_num', 'Lagged_Price']].iloc[-forecast_steps:]
        predictions = model.predict(X_last)

    # Aligning predictions with actual values
    actual_values = df['Price'].iloc[-forecast_steps:]  # Last 'n' actual values
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    mae = mean_absolute_error(actual_values, predictions)
    r_squared = r2_score(actual_values, predictions)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r_squared
    }

def generate_insights(model):
    """
    Generate insights from the model output.
    
    Args:
        model: Fitted ARIMA model.
        
    Returns:
        str: Insights based on model results.
    """
    summary = model.summary()
    return summary
