import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
n = 500  # Number of samples
do = np.random.uniform(1, 9, n)  # DO values
temperature = np.random.uniform(20, 40, n)  # Temperature values
humidity = np.random.uniform(50, 100, n)  # Humidity values

# Classification Logic: Label the data
def classify(do, temp, hum):
    if do > 6 and temp < 30:
        return "Safe"
    elif 4 <= do <= 6 or temp >= 30:
        return "Moderate"
    else:
        return "Dangerous"

water_quality = [classify(d, t, h) for d, t, h in zip(do, temperature, humidity)]

# Create the DataFrame
data = pd.DataFrame({
    "DO": do,
    "Temperature": temperature,
    "Humidity": humidity,
    "Water Quality": water_quality
})

# Save the dataset to CSV
data.to_csv("water_quality_dataset.csv", index=False)
print("Dataset created and saved as water_quality_dataset.csv")
