import pandas as pd
import pickle

# Step 1: Load Model, Scaler, and Column Names
#with open('model.pkl', 'rb') as f:
#    model = pickle.load(f)
model=pickle.load(open("model.pkl","rb"))

#with open('scaler.pkl', 'rb') as f:
#    scaler = pickle.load(f)
scaler=pickle.load(open("scalar.pkl","rb"))
#with open('columns.pkl', 'rb') as f:
#    columns = pickle.load(f)
columns=pickle.load(open("columns.pkl","rb"))

# Step 2: New Input Data
new_data = pd.DataFrame({
    'Age': [45],
    'Gender': ['Male'],
    'Salary': [85000]
})

# Step 3: One-hot Encode the Same Way
new_data_encoded = pd.get_dummies(new_data)

# Add missing columns
for col in columns:
    if col not in new_data_encoded.columns:
        new_data_encoded[col] = 0

# Ensure column order matches
new_data_encoded = new_data_encoded[columns]

# Step 4: Scale
new_data_scaled = scaler.transform(new_data_encoded)

# Step 5: Predict
prediction = model.predict(new_data_scaled)
print(" Prediction:", prediction[0])
