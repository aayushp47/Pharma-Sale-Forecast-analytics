#!/usr/bin/env python3
"""
Advanced Forecasting Engine for Healthcare Sales Analysis
Implements multiple forecasting models: ARIMA, Prophet, XGBoost, etc.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Forecasting libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

import xgboost as xgb


class ForecastingEngine:
    """Advanced forecasting engine with multiple models"""
    
    def __init__(self):
        self.models = {
            'arima': self._arima_forecast,
            'prophet': self._prophet_forecast,
            'xgboost': self._xgboost_forecast,
            'linear': self._linear_forecast,
        }
        
    def get_available_models(self):
        """Get list of available forecasting models"""
        available = ['linear', 'xgboost']
        if STATSMODELS_AVAILABLE:
            available.append('arima')
        if PROPHET_AVAILABLE:
            available.append('prophet')
        return available
    
    def generate_forecast(self, data, model_name='arima', periods=12, **kwargs):
        """
        Generates a forecast with real accuracy calculation.
        1. Splits data into train/test sets.
        2. Trains model on train set and calculates accuracy on test set.
        3. Retrains model on ALL historical data.
        4. Generates final forecast for the future.
        """
        try:
            ts_data = self._prepare_time_series(data)
            
            if len(ts_data) < 15: # Need enough data for a meaningful train/test split
                raise ValueError("Insufficient data for accuracy calculation (min 15 data points).")
            
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not available.")
            
            forecast_func = self.models[model_name]

            # --- 1. Accuracy Calculation ---
            test_size = max(4, int(len(ts_data) * 0.2)) # Use last 20% or at least 4 periods for testing
            train_data = ts_data[:-test_size]
            test_data = ts_data[-test_size:]

            # Get predictions for the test period
            test_predictions_result = forecast_func(train_data, periods=test_size, **kwargs)
            test_predictions = test_predictions_result['values']

            # Calculate accuracy metrics
            mae = mean_absolute_error(test_data.values, test_predictions)
            rmse = np.sqrt(mean_squared_error(test_data.values, test_predictions))
            
            # Avoid division by zero for MAPE
            mape_mask = test_data.values != 0
            test_actuals_safe = test_data.values[mape_mask]
            test_predictions_safe = np.array(test_predictions)[mape_mask]

            if len(test_actuals_safe) > 0:
                mape = np.mean(np.abs((test_actuals_safe - test_predictions_safe) / test_actuals_safe)) * 100
            else:
                mape = 0
            
            accuracy = {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'mape': round(mape, 2),
                'accuracy_percentage': max(0, 100 - mape)
            }

            # --- 2. Final Forecast Generation ---
            # Retrain the model on the FULL dataset to get the best forecast
            final_forecast_result = forecast_func(ts_data, periods, **kwargs)

            return {
                'success': True,
                'model': model_name,
                'historical': self._format_time_series(ts_data),
                'forecast': final_forecast_result,
                'accuracy': accuracy,
                'periods': periods,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return { 'success': False, 'error': str(e), 'model': model_name }

    def _prepare_time_series(self, data):
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')
        ts_data = data.groupby('Date')['Revenue'].sum()
        # Resample to monthly start frequency, forward-fill gaps, then back-fill any remaining NaNs
        ts_data = ts_data.asfreq('MS').fillna(method='ffill').fillna(method='bfill')
        return ts_data

    def _format_time_series(self, ts_data):
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in ts_data.index],
            'values': ts_data.values.tolist()
        }

    def _arima_forecast(self, ts_data, periods, order=(1,1,1), seasonal_order=(0,1,1,12)):
        if not STATSMODELS_AVAILABLE:
            raise ValueError("ARIMA requires statsmodels.")
        model = ARIMA(ts_data, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit()
        forecast = fitted_model.get_forecast(steps=periods)
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in forecast_values.index],
            'values': forecast_values.tolist(),
            'lower_bound': conf_int.iloc[:, 0].tolist(),
            'upper_bound': conf_int.iloc[:, 1].tolist()
        }

    def _prophet_forecast(self, ts_data, periods, **kwargs):
        if not PROPHET_AVAILABLE:
            raise ValueError("Prophet requires prophet library.")
        prophet_data = pd.DataFrame({'ds': ts_data.index, 'y': ts_data.values})
        model = Prophet(**kwargs)
        model.fit(prophet_data)
        future = model.make_future_dataframe(periods=periods, freq='MS')
        forecast = model.predict(future)
        future_forecast = forecast.tail(periods)
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in future_forecast['ds']],
            'values': future_forecast['yhat'].tolist(),
            'lower_bound': future_forecast['yhat_lower'].tolist(),
            'upper_bound': future_forecast['yhat_upper'].tolist()
        }

    def _linear_forecast(self, ts_data, periods, **kwargs):
        X = np.arange(len(ts_data)).reshape(-1, 1)
        y = ts_data.values
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(ts_data), len(ts_data) + periods).reshape(-1, 1)
        forecast_values = model.predict(future_X)
        residuals = y - model.predict(X)
        std_residual = np.std(residuals)
        last_date = ts_data.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='MS')[1:]
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'values': forecast_values.tolist(),
            'lower_bound': (forecast_values - 1.96 * std_residual).tolist(),
            'upper_bound': (forecast_values + 1.96 * std_residual).tolist()
        }

    def _create_lag_features(self, ts_data, lags=12):
        df = pd.DataFrame({'value': ts_data})
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['value'].shift(i)
        df['month'] = ts_data.index.month
        df['year'] = ts_data.index.year
        df = df.dropna()
        return df.drop('value', axis=1), df['value']
        
    def _xgboost_forecast(self, ts_data, periods, **kwargs):
        X, y = self._create_lag_features(ts_data)
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model.fit(X, y)
        
        forecast_values = []
        current_data = ts_data.copy()
        
        for _ in range(periods):
            X_pred, _ = self._create_lag_features(current_data)
            next_value = model.predict(X_pred.iloc[-1:].values)[0]
            forecast_values.append(float(next_value))
            
            last_date = current_data.index[-1]
            next_date = last_date + pd.DateOffset(months=1)
            # Use pd.concat instead of .loc for appending to time series
            current_data = pd.concat([current_data, pd.Series([next_value], index=[next_date])])
        
        last_date = ts_data.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='MS')[1:]
        
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'values': forecast_values,
            'lower_bound': [v * 0.9 for v in forecast_values],
            'upper_bound': [v * 1.1 for v in forecast_values]
        }