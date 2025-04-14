import pandas as pd
import numpy as np
import os

# --- Parameters ---
input_failure_data = 'data/machine_failures.csv'
output_timeseries_data = 'data/sensor_readings_timeseries.csv'
time_step = 100  # Simulate sensor readings every 100 time units
near_failure_threshold = 100 # RUL threshold to flag 'near_failure'

# --- Sensor Simulation Models ---

def simulate_temperature(time, ttf, machine_id, department, failure_type):
    """Simulates temperature reading at a given time."""
    # Baseline temperature (higher in Welding)
    baseline = 50 + (15 if department == 'Welding' else 0)

    # Degradation effect (increases more for Mechanical, accelerates near ttf)
    # Using a simple quadratic increase towards failure
    time_ratio = time / ttf if ttf > 0 else 0
    degradation_effect = 0
    if failure_type == 'Mechanical':
        degradation_effect = 30 * (time_ratio ** 2)
    elif failure_type == 'Electrical':
         degradation_effect = 10 * (time_ratio ** 1.5) # Different profile for electrical

    # Random noise
    noise = np.random.normal(0, 2) # Gaussian noise with std dev 2

    # Combine base, degradation, and noise
    temp = baseline + degradation_effect + noise
    return temp

def simulate_vibration(time, ttf, machine_id, department, failure_type):
    """Simulates vibration reading at a given time."""
    # Baseline vibration (slightly higher in Welding)
    baseline = 0.1 + (0.05 if department == 'Welding' else 0)

    # Degradation effect (increases significantly for Mechanical, accelerates near ttf)
    time_ratio = time / ttf if ttf > 0 else 0
    degradation_effect = 0
    if failure_type == 'Mechanical':
        # Exponential increase for vibration is common
        degradation_effect = 0.5 * np.exp(3 * time_ratio) - 0.5
    elif failure_type == 'Electrical':
         # Less pronounced effect for electrical, maybe slight increase
         degradation_effect = 0.05 * (time_ratio**2)


    # Random noise
    noise = np.random.normal(0, 0.03) # Gaussian noise

    # Combine base, degradation, and noise
    vibration = baseline + degradation_effect + noise
    # Ensure vibration doesn't go below zero (or a small positive value)
    return max(0.01, vibration)


# --- Main Data Generation ---

# Load the original failure data
if not os.path.exists(input_failure_data):
    print(f"Error: Input file not found at {input_failure_data}")
    print("Please run generate_four_machines.py first.")
    exit()

print(f"Loading failure data from {input_failure_data}...")
failures_df = pd.read_csv(input_failure_data)
print(f"Loaded {len(failures_df)} failure events.")

all_sensor_readings = []
instance_counter = 0

print("Generating time series sensor data...")
# Iterate through each unique failure event (each row in failures_df)
for index, row in failures_df.iterrows():
    instance_counter += 1
    machine_id = row['machine_id']
    department = row['department']
    failure_type = row['failure_type']
    ttf = row['ttf']

    # Simulate readings at each time step from 0 up to ttf
    for time in np.arange(0, ttf + time_step, time_step):
        # Ensure we don't significantly overshoot ttf if ttf is not a multiple of time_step
        current_time = min(time, ttf)

        # Calculate sensor readings at this time
        temp_reading = simulate_temperature(current_time, ttf, machine_id, department, failure_type)
        vib_reading = simulate_vibration(current_time, ttf, machine_id, department, failure_type)

        # Calculate RUL
        rul = ttf - current_time

        # Determine if near failure
        is_near_failure = 1 if rul <= near_failure_threshold else 0

        # Store the data for this time step
        all_sensor_readings.append({
            'instance_id': instance_counter, # ID linking readings from the same lifecycle
            'machine_id': machine_id,
            'department': department,
            'failure_type': failure_type, # The eventual failure type
            'time': current_time,
            'sensor_temp': temp_reading,
            'sensor_vib': vib_reading,
            'rul': rul,
            'near_failure': is_near_failure
        })

    # Progress indicator (optional)
    if instance_counter % 500 == 0:
        print(f"  Processed {instance_counter}/{len(failures_df)} instances...")


print("Converting data to DataFrame...")
# Convert list of dictionaries to DataFrame
sensor_df = pd.DataFrame(all_sensor_readings)

# Optional: Shuffle the DataFrame rows if needed (might not be desirable for time series)
# sensor_df = sensor_df.sample(frac=1).reset_index(drop=True)

print(f"Generated {len(sensor_df)} time-series sensor readings.")
print("First 5 records:")
print(sensor_df.head())
print("Last 5 records:")
print(sensor_df.tail())


# --- Output ---
# Ensure the output directory exists
output_dir = os.path.dirname(output_timeseries_data)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Saving time-series data to {output_timeseries_data}...")
sensor_df.to_csv(output_timeseries_data, index=False)
print("Data generation complete.")

# --- Example Usage Hint ---
# To use this data for ML (predicting RUL or near_failure):
# 1. Load the CSV: df = pd.read_csv('data/sensor_readings_timeseries.csv')
# 2. Feature Engineering:
#    - Use 'sensor_temp', 'sensor_vib', 'time' as features.
#    - Consider creating lag features (sensor values from previous time steps)
#      or rolling window statistics (mean/std dev of sensors over recent steps).
#      This often requires grouping by 'instance_id'.
#    - Encode categorical features ('department', maybe 'machine_id', 'failure_type').
# 3. Model Training:
#    - For RUL Prediction: Train a regression model (e.g., RandomForestRegressor, GradientBoostingRegressor, LSTM) to predict the 'rul' column.
#    - For Classification: Train a classifier (e.g., Logistic Regression, RandomForestClassifier, XGBoost) to predict the 'near_failure' column.
# 4. Evaluation: Use appropriate metrics (RMSE/MAE for regression, Accuracy/Precision/Recall/F1 for classification). Be mindful of time series cross-validation if creating time-dependent features. 