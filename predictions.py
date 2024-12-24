import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import joblib

# Firebase Initialization
cred = credentials.Certificate("./esp32-firebase-demo-39ba1-firebase-adminsdk-x7eg7-77710dda04.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://esp32-firebase-demo-39ba1-default-rtdb.asia-southeast1.firebasedatabase.app"
})

# Load trained artifacts
model = joblib.load("water_quality_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

def fetch_sensor_data():
    sensor_data_ref = db.reference("sensorData")
    sensor_data = sensor_data_ref.get()

    # Check for missing or incorrectly fetched data
    if sensor_data is None:
        print("No data found in Firebase!")
        return pd.DataFrame()

    # Ensure no null or empty entries
    sensor_data = [entry for entry in sensor_data if entry is not None]
    print("Fetched Raw Data:", sensor_data)  # Debug: Check raw data

    # Convert to DataFrame
    sensor_df = pd.DataFrame(sensor_data)
    print("Fetched DataFrame:", sensor_df.head())  # Debug: Check the DataFrame
    return sensor_df

def classify_and_update():
    sensor_df = fetch_sensor_data()

    # Check if DataFrame is empty
    if sensor_df.empty:
        print("No valid sensor data to process.")
        return

    # Validate columns
    required_columns = ['DO', 'Temperature', 'Humidity']
    if not all(col in sensor_df.columns for col in required_columns):
        raise KeyError(f"Missing columns. Expected {required_columns}, but got {list(sensor_df.columns)}")

    # Prepare features
    X = sensor_df[required_columns]
    X_scaled = scaler.transform(X)  # Apply saved scaler

    # Predict and decode labels
    predictions = model.predict(X_scaled)
    predicted_labels = encoder.inverse_transform(predictions)

    # Print predictions for debugging
    for i, (label, row) in enumerate(zip(predicted_labels, sensor_df.iterrows())):
        print(f"Prediction for Sensor {i+1}: {label}, Data: {row[1].to_dict()}")

    # Update Firebase with predictions
    for i, predicted_label in enumerate(predicted_labels):
        sensor_data_ref = db.reference(f"sensorData/{i+1}")
        sensor_data_ref.update({
            "prediction": predicted_label
        })

    print("Predictions updated successfully!")

# Run prediction and update
if __name__ == "__main__":
    classify_and_update()
# import firebase_admin
# from firebase_admin import credentials, db
# import pandas as pd
# import joblib

# # Firebase Initialization
# cred = credentials.Certificate("./esp32-firebase-demo-39ba1-firebase-adminsdk-x7eg7-77710dda04.json")
# firebase_admin.initialize_app(cred, {
#     'databaseURL': "https://esp32-firebase-demo-39ba1-default-rtdb.asia-southeast1.firebasedatabase.app"
# })

# # Load trained artifacts
# model = joblib.load("water_quality_model.pkl")
# scaler = joblib.load("scaler.pkl")
# encoder = joblib.load("label_encoder.pkl")

# def fetch_sensor_data():
#     sensor_data_ref = db.reference("sensorData")
#     sensor_data = sensor_data_ref.get()

#     # Remove null or empty entries
#     sensor_data = [entry for entry in sensor_data if entry is not None]

#     # Convert to DataFrame
#     sensor_df = pd.DataFrame(sensor_data)
#     print("Fetched DataFrame:", sensor_df.head())
#     return sensor_df

# def classify_and_update():
#     sensor_df = fetch_sensor_data()

#     # Validate columns
#     required_columns = ['DO', 'Temperature', 'Humidity']
#     if not all(col in sensor_df.columns for col in required_columns):
#         raise KeyError(f"Missing columns. Expected {required_columns}, but got {list(sensor_df.columns)}")

#     # Prepare features
#     X = sensor_df[required_columns]
#     X_scaled = scaler.transform(X)  # Apply saved scaler

#     # Predict and decode labels
#     predictions = model.predict(X_scaled)
#     predicted_labels = encoder.inverse_transform(predictions)

#     # Update Firebase with predictions
#     for i, predicted_label in enumerate(predicted_labels):
#         sensor_data_ref = db.reference(f"sensorData/{i+1}")
#         sensor_data_ref.update({
#             "prediction": predicted_label
#         })

#     print("Predictions updated successfully!")

# # Run prediction and update
# if __name__ == "__main__":
#     classify_and_update()

# import firebase_admin
# from firebase_admin import credentials, db
# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler

# # Firebase Initialization
# cred = credentials.Certificate("./esp32-firebase-demo-39ba1-firebase-adminsdk-x7eg7-77710dda04.json")
# firebase_admin.initialize_app(cred, {
#     'databaseURL': "https://esp32-firebase-demo-39ba1-default-rtdb.asia-southeast1.firebasedatabase.app"
# })

# # Load Random Forest Model
# model = joblib.load("./water_quality_model.pkl")

# # Initialize Scaler (assuming StandardScaler was used during training)
# scaler = StandardScaler()

# def fetch_sensor_data():
#     sensor_data_ref = db.reference("sensorData")
#     sensor_data = sensor_data_ref.get()  # This gets the array of sensor data
    
#     # Ensure there are no null or empty entries
#     sensor_data = [entry for entry in sensor_data if entry is not None]
    
#     # Convert to DataFrame
#     sensor_df = pd.DataFrame(sensor_data)
    
#     # Log the data structure for debugging
#     print("Fetched DataFrame:", sensor_df.head())
#     print("Columns:", sensor_df.columns)
    
#     return sensor_df

# def classify_and_update():
#     sensor_df = fetch_sensor_data()

#     # Ensure columns match
#     required_columns = [ 'Humidity', 'DO', 'Temperature']
#     if not all(col in sensor_df.columns for col in required_columns):
#         raise KeyError(f"Missing columns in data. Expected {required_columns}, but got {list(sensor_df.columns)}")

#     # Handle missing values (if any)
#     if sensor_df.isnull().values.any():
#         sensor_df = sensor_df.fillna(sensor_df.mean())  # Fill missing values with column mean

#     # Ensure data types are correct (float for continuous variables)
#     sensor_df['Temperature'] = sensor_df['Temperature'].astype(float)
#     sensor_df['Humidity'] = sensor_df['Humidity'].astype(float)
#     sensor_df['DO'] = sensor_df['DO'].astype(float)

#     # Scale the features (if scaling was used during training)
#     X = sensor_df[required_columns]
#     X_scaled = scaler.fit_transform(X)  # Use the same scaler that was used during training

#     # Make Predictions
#     predictions = model.predict(X_scaled)

#     # Map predictions to labels
#     label_mapping = {0: 'safe', 1: 'moderate', 2: 'dangerous'}
#     predicted_labels = [label_mapping[pred] for pred in predictions]

#     # Update Firebase with predictions
#     for i, predicted_label in enumerate(predicted_labels):
#         sensor_data_ref = db.reference(f"sensorData/{i+1}")  # Assuming the index as the sensor ID (e.g., 1, 2, 3...)
#         sensor_data_ref.update({
#             "prediction": predicted_label
#         })

#     print("Predictions updated successfully!")

# # Run Prediction and Update
# if __name__ == "__main__":
#     classify_and_update()