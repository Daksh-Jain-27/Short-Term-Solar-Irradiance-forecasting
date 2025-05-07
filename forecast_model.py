import os
import xarray as xr
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

SAT_ROOT = 'satellite_nc'  # set to the common parent directory

def extract_features_from_nc(filepath):
    try:
        ds = xr.open_dataset(filepath)
        var = ds.to_array().squeeze()  # get the single variable regardless of name

        if var.ndim == 3:
            var = var[0]  # in case of multiple time steps, take first

        # Handle missing values and filter extreme values
        values = var.values.flatten()
        values = values[~np.isnan(values)]  # remove NaNs
        values = values[np.abs(values) < 1e10]  # filter extreme values
        
        if values.size == 0:
            raise ValueError("Empty array after filtering")
            
        ds.close()
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}")
        return None

# Scan and extract features
all_records = []

for root, dirs, files in os.walk(SAT_ROOT):
    for filename in sorted(files):
        if filename.endswith('.nc'):
            try:
                parts = filename.split('_')  # ['INSAT', '3DR', 'Kopal', '10Jun2024.nc']
                type_code = parts[1]
                date_str = parts[-1].replace('.nc', '').title()
                timestamp = pd.to_datetime(date_str, format='%d%b%Y')
            except Exception as e:
                print(f"[!] Skipping {filename} due to timestamp parse error: {e}")
                continue

            filepath = os.path.join(root, filename)
            features = extract_features_from_nc(filepath)
            if features is None:
                print(f"[!] Skipping {filename} due to empty array or invalid data")
                continue

            row = {'timestamp': timestamp, 'type': type_code}
            row.update(features)
            all_records.append(row)

# Convert to DataFrame and handle duplicates
sat_df = pd.DataFrame(all_records)
sat_df = sat_df.groupby(['timestamp', 'type']).mean().reset_index()

# Pivot with aggregation
sat_combined = sat_df.pivot_table(index='timestamp', columns='type', aggfunc='mean')
sat_combined.columns = ['_'.join(col).strip() for col in sat_combined.columns.values]
sat_combined.reset_index(inplace=True)

# Load and clean GHI data
ghi_df = pd.read_csv("Sample Dataset - ML Assignment - Sheet1.csv")

# Clean column names
ghi_df.columns = (ghi_df.columns
                 .str.replace(r'[\t\n]', '', regex=True)
                 .str.strip())

# Drop empty columns
ghi_df = ghi_df.dropna(axis=1, how='all')

# Rename columns
ghi_df.rename(columns={"Date": "timestamp", "Observed GHI": "GHI"}, inplace=True)

# Convert timestamp and filter
ghi_df['timestamp'] = pd.to_datetime(ghi_df['timestamp'], 
                                   format='%m/%d/%Y %I:%M:%S %p',
                                   errors='coerce').dt.normalize()
ghi_df = ghi_df.dropna(subset=['timestamp'])
ghi_df = ghi_df[(ghi_df['timestamp'] >= '2024-06-01')]

# Handle duplicates in GHI data
ghi_df = ghi_df.groupby('timestamp').mean().reset_index()

final_df = pd.merge(ghi_df, sat_combined, on='timestamp', how='inner')

# Add time-based features
final_df['hour'] = final_df['timestamp'].dt.hour
final_df['minute'] = final_df['timestamp'].dt.minute
final_df['dayofweek'] = final_df['timestamp'].dt.dayofweek

# Define features and target
features = [col for col in final_df.columns if col not in ['timestamp', 'GHI']]
target = 'GHI'

# Split: train = before Aug 16, val = Aug 16–31
train_df = final_df[final_df['timestamp'] < '2024-08-16']
val_df = final_df[(final_df['timestamp'] >= '2024-08-16') & (final_df['timestamp'] <= '2024-08-31')]

X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]


model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_val)


rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
mape = np.mean(np.abs((y_val - y_pred) / y_val.replace(0, np.nan))) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# Filter future timestamps (Sept 1–7)
test_df = final_df[(final_df['timestamp'] >= '2024-09-01') & (final_df['timestamp'] <= '2024-09-07')]
X_test = test_df[features]
test_df['predicted_GHI'] = model.predict(X_test)

# Save forecast
test_df[['timestamp', 'predicted_GHI']].to_csv('ghi_forecast_sept1_7.csv', index=False)



