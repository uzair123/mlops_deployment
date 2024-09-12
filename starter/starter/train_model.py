# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import data
import train_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# Add code to load in the data.
df = pd.read_csv('data/census.csv') 
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
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False
)
# Train and save a model.
model = train_model(X_train, y_train)
model_filename = 'random_forest_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")



     #  Evaluate the model
     #y_pred = model.predict(X_test)
     #accuracy = accuracy_score(y_test, y_pred)
     #print(f"Model Accuracy: {accuracy * 100:.2f}%")


# Optional: Load the saved model and make a prediction to verify
#loaded_model = joblib.load(model_filename)
#sample_data = X_test.iloc[0:1]  # Take a sample for prediction
#sample_prediction = loaded_model.predict(sample_data)
#print(f"Sample prediction: {sample_prediction}")

