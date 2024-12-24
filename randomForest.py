from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

# Load dataset
data = pd.read_csv("water_quality_dataset.csv")

# Prepare features and labels
X = data[['DO', 'Temperature', 'Humidity']]  # Fix feature selection
y = data['Water Quality']

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Save encoder for later use
joblib.dump(encoder, "label_encoder.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for predictions
joblib.dump(scaler, "scaler.pkl")

# Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Save the model
joblib.dump(model, "water_quality_model.pkl")
print("Model saved as water_quality_model.pkl")

# Convert the model to ONNX format
onnx_model = convert_sklearn(
    model, 
    initial_types=[('input', FloatTensorType([None, X_train.shape[1]]))]
)
onnx.save_model(onnx_model, 'randomForest_model.onnx')










# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import joblib  # For saving the model
# import pandas as pd
# import joblib
# import skl2onnx
# import onnx
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType


# # Load dataset
# data = pd.read_csv("water_quality_dataset.csv")

# # Prepare features and labels
# X = data[['DO', 'Temperature', 'Humidity']]
# y = data['Water Quality']

# # Encode labels
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# y = encoder.fit_transform(y)  # Converts categories to numbers (Safe=0, Moderate=1, Dangerous=2)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Random Forest Classifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Evaluate model
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# # Save the model
# joblib.dump(model, "water_quality_model.pkl")
# print("Model saved as water_quality_model.pkl")

# # Convert the model to ONNX format
# onnx_model = convert_sklearn(model, 
#                              initial_types=[('input', FloatTensorType([None, len(model.feature_importances_)]))])

# # Save the ONNX model to a file
# onnx.save_model(onnx_model, 'randomForest_model.onnx')