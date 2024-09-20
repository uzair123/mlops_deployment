# Script to train machine learning model.
# Add the necessary imports for the starter code.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from ml.model import train_model, model_performance_on_slices,compute_model_metrics,inference
from ml.data import process_data
from ml.data import clean_data
import pickle
import logging

# Add code to load in the data.
df = pd.read_csv('../data/census.csv')
data = clean_data(df)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train
model = train_model(X_train, y_train)

# Save the model
model_filename = 'model.pkl'
print(f"Model saved to {model_filename}")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_filename}")

# Save the trained OneHotEncoder
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
print("Enocder saved!")

with open('lb.pkl', 'wb') as f:
    pickle.dump(lb, f)

#perform Inference
pred = inference(model, X_test)

#KPI metrics
precision, recall, fbeta = compute_model_metrics(y_test,pred)
print("precision, recall, fbeta",precision, recall, fbeta)

# Identify categorical columns (with dtype 'object' for strings or 'category')
categorical_columns = df.select_dtypes(include=['object']).columns

# Get indices of the categorical columns in the DataFrame
categorical_indices = [df.columns.get_loc(col) for col in categorical_columns]

# Print the categorical columns and their corresponding indices
print("Categorical Columns:", categorical_columns)
print("Categorical Indices:", categorical_indices)

print("model_performance_on_slices")
model_performance_on_slices(model, X_test,y_test,categorical_indices)
