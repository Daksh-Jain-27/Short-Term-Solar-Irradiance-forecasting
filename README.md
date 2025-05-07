# Short-Term Solar Irradiance Forecasting

## Overview
This project develops a machine learning model to forecast Global Horizontal Irradiance (GHI) using satellite data and weather observations. The system processes NetCDF satellite data, merges it with ground-based GHI measurements, and trains an XGBoost model for 1-7 day solar irradiance predictions.

## Features
- Processes INSAT-3D satellite NetCDF files
- Extracts statistical features (mean, std, min, max) from satellite data
- Merges satellite features with ground-based GHI measurements
- XGBoost regression model for accurate predictions
- Generates forecasts for future dates (September 1-7, 2024)

## Requirements
- Python 3.8+
- Required packages:
  ```bash
  pip install xarray numpy pandas scikit-learn xgboost


## Model Performance

### Validation Metrics (August 16-31, 2024)
| Metric  | Value Range  |
|---------|------------|
| RMSE    | 79.29 W/m² | 
| MAE     | 65.28 W/m² | 
| MAPE    | 47.53%      |
