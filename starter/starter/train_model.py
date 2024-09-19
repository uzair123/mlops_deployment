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
# Train and save a model.
model = train_model(X_train, y_train)
model_filename = 'model.pkl'
#joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")



# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_filename}")

# Save the trained OneHotEncoder
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("Enocder saved!")


with open('lb.pkl', 'wb') as f:
    pickle.dump(lb, f)

pred = inference(model, X_test)
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


     #  Evaluate the model
     #y_pred = model.predict(X_test)
     #accuracy = accuracy_score(y_test, y_pred)
     #print(f"Model Accuracy: {accuracy * 100:.2f}%")


# Optional: Load the saved model and make a prediction to verify
#loaded_model = joblib.load(model_filename)
#sample_data = X_test.iloc[0:1]  # Take a sample for prediction
#sample_prediction = loaded_model.predict(sample_data)
#print(f"Sample prediction: {sample_prediction}")

