import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Motor_Vehicle_Collisions_-_Crashes_2020janto2026march5.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# This assumes you have 'crash_date' and 'crash_time' columns
# combining into the more common format of timestamp
df['timestamp'] = pd.to_datetime(df['crash_date'] + ' ' + df['crash_time'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# 3. Handle missing values for critical columns
critical_cols = [
    'latitude', 'longitude',
    'number_of_pedestrians_injured', 'number_of_pedestrians_killed',
    'number_of_cyclist_injured', 'number_of_cyclist_killed',
    'number_of_motorist_injured', 'number_of_motorist_killed'
]
df = df.dropna(subset=critical_cols)

# 4. Filter invalid coordinates (0,0)
df = df[(df['latitude'] != None) & (df['longitude'] != None)]

# 7. Select relevant features for your model
# We drop the identifiers and original raw casualty columns now that we have the target
cols_to_keep = [
    'latitude', 'longitude', 'number_of_pedestrians_injured', 'number_of_pedestrians_killed',
    'number_of_cyclist_injured', 'number_of_cyclist_killed',
    'number_of_motorist_injured', 'number_of_motorist_killed'
]
# Ensure columns exist before filtering (adds robustness)
df = df[[col for col in cols_to_keep if col in df.columns]]

# 8. Reset index
df = df.reset_index(drop=True)

# Save the preprocessed data
df.to_csv('crash_cleaned.csv', index=False)

print("Preprocessing complete. Data shape:", df.shape)
print(df.head())