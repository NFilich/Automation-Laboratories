import numpy as np
import pandas as pd

# --- Parameters ---
n_samples_per_combo = 1000  # Number of failure samples per machine/type combination
output_filename = 'data/machine_failures.csv'

# Department assignments and TTF modifiers
department_map = {
    1: 'Packaging',
    2: 'Welding',
    3: 'Packaging',
    4: 'Welding'
}
department_modifiers = {
    'Packaging': 1.5,  # Increase TTF for Packaging machines
    'Welding': 0.5     # Decrease TTF for Welding machines
}

# Weibull parameters (shape, scale) for each machine and failure type
# Shape (k): k<1 decreasing failure rate, k=1 constant, k>1 increasing failure rate
# Scale (lambda): Characteristic life
machine_params = {
    1: {'Mechanical': (1.5, 1000), 'Electrical': (0.8, 5000)}, # Machine 1: Incr FR (Mechanical), Decr FR (Electrical)
    2: {'Mechanical': (2.0, 800),  'Electrical': (1.0, 3000)}, # Machine 2: Faster Incr FR (Mechanical), Constant FR (Electrical)
    3: {'Mechanical': (1.2, 1500), 'Electrical': (2.5, 700)},  # Machine 3: Slow Incr FR (Mechanical), Rapid Incr FR (Electrical)
    4: {'Mechanical': (0.9, 6000), 'Electrical': (1.8, 1200)}  # Machine 4: Decr FR (Mechanical), Incr FR (Electrical)
}

# --- Data Generation ---
all_failures = []

for machine_id, types in machine_params.items():
    department = department_map[machine_id] # Get department for the machine
    modifier = department_modifiers[department] # Get the TTF modifier for the department

    for failure_type, params in types.items():
        shape, scale = params
        # Generate TTF values using Weibull distribution
        # numpy.random.weibull(shape) * scale gives Weibull distributed values
        ttf_values = np.random.weibull(shape, n_samples_per_combo) * scale

        # Append to list, applying the department modifier
        for ttf in ttf_values:
            adjusted_ttf = ttf * modifier # Apply department modifier
            all_failures.append({
                'machine_id': machine_id,
                'department': department,       # Add department column
                'failure_type': failure_type,
                'ttf': adjusted_ttf            # Use adjusted TTF
            })

# Convert list of dictionaries to DataFrame
failures_df = pd.DataFrame(all_failures)

# Reorder columns for clarity (optional)
failures_df = failures_df[['machine_id', 'department', 'failure_type', 'ttf']]

# Optional: Shuffle the DataFrame rows
failures_df = failures_df.sample(frac=1).reset_index(drop=True)

# --- Output ---
print(f"Generated {len(failures_df)} failure records.")
print("First 5 records:")
print(failures_df.head())

# Save to CSV
failures_df.to_csv(output_filename, index=False)
print(f"Data saved to {output_filename}")

# --- Example Usage Hint ---
# To use this data for ML:
# 1. Load the CSV: df = pd.read_csv('data/machine_failures.csv')
# 2. Feature Engineering: You might create features based on machine_id, department, failure_type,
#    or potentially simulate sensor readings leading up to the failure time (TTF).
# 3. Model Training: Train models (e.g., regression models like Random Forest, Gradient Boosting,
#    or survival analysis models) to predict 'ttf' based on the engineered features.