import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import pickle

# Sample Data
data = pd.DataFrame({
    'Age': [25, 32, 47, 51, 62],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Salary': [50000, 60000, 80000, 72000, 90000],
    'Purchased': [0, 1, 1, 0, 1]
})

# One-hot encoding
data_encoded = pd.get_dummies(data, columns=['Gender'])

# Features and labels
X = data_encoded.drop('Purchased', axis=1)
y = data_encoded['Purchased']

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)

# Save model, scaler, columns
pickle.dump(model, open("model.pkl", 'wb'))
pickle.dump(scaler, open("scalar.pkl", 'wb'))
pickle.dump(X.columns.tolist(), open("columns.pkl", 'wb'))

# ✅ Print in expected format
print(f"ACCURACY={accuracy:.2f}")
print(f"PRECISION={precision:.2f}")
