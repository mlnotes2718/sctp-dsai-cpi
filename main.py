import pandas as pd
import numpy as np
from pmdarima import auto_arima
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def load_data(filepath):
    """Loads annual CPI data from a CSV file."""
    # Assuming CSV has 'Year' and 'CPI' columns
    df = pd.read_csv(filepath, parse_dates=['Year'], index_col='Year')
    df.index = df.index.year # Set index to just the year for annual data
    return df['CPI'] # Assuming the CPI data is in a column named 'CPI'

def forecast_cpi(data, params):
    """
    Fits an auto_arima model and generates forecasts.

    Args:
        data (pd.Series): Time series data (annual CPI).
        params (dict): Dictionary of auto_arima parameters.

    Returns:
        tuple: Fitted model and forecast.
    """
    print("Fitting auto_arima model...")
    # 'm=1' for annual data as typically there's no strong seasonality
    # 'seasonal=False' can be explicitly set if you are sure there's no seasonality
    # The auto_arima parameters are loaded from the YAML file
    model = auto_arima(data,
                       start_p=params['start_p'],
                       start_q=params['start_q'],
                       max_p=params['max_p'],
                       max_q=params['max_q'],
                       d=params['d'],
                       max_d=params['max_d'],
                       trace=params['trace'],
                       stepwise=params['stepwise'],
                       suppress_warnings=params['suppress_warnings'],
                       information_criterion=params['information_criterion'],
                       m=params['m'], # m=1 for annual data
                       seasonal=params['seasonal'], # Set to False for non-seasonal annual data
                       error_action=params['error_action'],
                       n_jobs=params['n_jobs'])

    print(model.summary())

    n_forecast = params['n_forecast']
    forecast = model.predict(n_periods=n_forecast)

    # Create a proper index for the forecast
    last_year = data.index[-1]
    forecast_index = pd.to_datetime([f"{last_year + i}-01-01" for i in range(1, n_forecast + 1)]).year
    forecast_series = pd.Series(forecast, index=forecast_index, name='Forecasted CPI')
    
    return model, forecast_series

def evaluate_model(actual, predicted):
    """Evaluates the model using RMSE."""
    rmse = sqrt(mean_squared_error(actual, predicted))
    print(f"RMSE: {rmse:.2f}")
    return rmse

def plot_results(train_data, test_data, forecast_data, title="Annual CPI Forecast"):
    """Plots the historical data, actual test data, and forecasts."""
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Training Data', color='blue')
    plt.plot(test_data.index, test_data, label='Actual Test Data', color='orange')
    plt.plot(forecast_data.index, forecast_data, label='Forecast', color='green', linestyle='--')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('CPI')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Create dummy annual CPI data for demonstration purposes
    # In a real scenario, you would load your actual CPI data.
    # Example: FRED (Federal Reserve Economic Data) or national statistics agencies
    # often provide annual CPI data.
    
    # Generate some synthetic data
    years = np.arange(1980, 2024, 1)
    # Simulate a trend and some noise
    cpi_values = 100 + np.cumsum(np.random.normal(1.5, 0.5, len(years))) + np.sin(np.linspace(0, 2*np.pi, len(years))) * 5
    cpi_data = pd.Series(cpi_values, index=years, name='CPI')
    cpi_data.index.name = 'Year'

    # Save dummy data to a CSV (for demonstration of load_data)
    dummy_data_path = 'annual_cpi_data.csv'
    cpi_data.to_csv(dummy_data_path, header=True)
    print(f"Dummy annual CPI data saved to {dummy_data_path}")

    # Load data
    cpi_series = load_data(dummy_data_path)

    # Split data into training and testing sets
    train_size = int(len(cpi_series) * 0.8)
    train_cpi = cpi_series[:train_size]
    test_cpi = cpi_series[train_size:]

    # Load parameters from YAML file
    try:
        with open('arima_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        arima_params = config['auto_arima_parameters']
        forecast_params = config['forecast_parameters']
    except FileNotFoundError:
        print("Error: arima_config.yaml not found. Please create the YAML file.")
        exit()
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        exit()

    # Combine parameters for auto_arima
    all_params = {**arima_params, **forecast_params}

    # Perform forecasting
    model, forecast_series = forecast_cpi(train_cpi, all_params)

    # Evaluate the model on the test set
    # Note: For annual data, the test set might be very small.
    # It's often better to retrain on the full dataset before final forecasting.
    if len(test_cpi) > 0:
        # Predict on the test set using the fitted model
        test_predictions = model.predict(n_periods=len(test_cpi))
        test_predictions_series = pd.Series(test_predictions, index=test_cpi.index, name='Test Predictions')
        
        print("\n--- Model Evaluation on Test Set ---")
        evaluate_model(test_cpi, test_predictions_series)
        
        # Plot results including test set
        plot_results(train_cpi, test_cpi, forecast_series, "Annual CPI Forecast with Test Data")
    else:
        print("\nNot enough data for a meaningful test set evaluation.")
        # Plot results without a test set (only historical data and forecast)
        plot_results(train_cpi, pd.Series([], dtype=float), forecast_series, "Annual CPI Forecast (No Test Data)")