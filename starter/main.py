from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from starter.ml.data import process_data
# Initialize the FastAPI app
app = FastAPI()

import pickle

# Load the model
with open('./starter/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the enoder
with open('./starter/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)



#model = joblib.load('./starter/model.pkl')
try:
 with open('/starter/encoder.pkl', 'rb') as file:
     # Load or process the file
     # Load your pre-trained model (assuming a RandomForestClassifier for this example)
     #model = joblib.load('../starter/model.pkl')
     print("File found! Model there")
except FileNotFoundError:
    print("File not found. Please check the file path and try again.")



# Define the Pydantic model to handle incoming JSON data
class InferenceData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(..., alias='education-num')
    marital_status: str = Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')
    native_country: str = Field(..., alias='native-country')




# Root endpoint to return a welcome message
@app.get("/")
async def root():
    return {"message": "Welcome to the ML Model API!"}


@app.post("/predict")
async def predict(data: InferenceData):
    # Convert incoming data to DataFrame for model input
    input_data = pd.DataFrame([data.dict(by_alias=True)])  # Use by_alias=True to handle hyphenated fields

    # Convert NumPy types (np.int64, np.float64) to native Python types
    input_data = input_data.astype({col: 'int' for col in input_data.select_dtypes(include=[np.int64]).columns})
    input_data = input_data.astype({col: 'float' for col in input_data.select_dtypes(include=[np.float64]).columns})

    # Define the categorical features that need to be encoded
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']

    # Process the input data using the same encoder used during training
    input_data_processed, _, _, _ = process_data(
        input_data,
        categorical_features=categorical_features,
        training=False,  # Ensure that this is set to False for inference
        encoder=encoder,  # Load the pre-trained OneHotEncoder
        lb=None  # No label binarization for inference
    )

    # Perform model inference
    prediction = model.predict(input_data_processed)
    # Map the output (0/1) to readable labels (for example: ">50K" or "<=50K")
    label_map = {0: "<=50K", 1: ">50K"}
    predicted_label = label_map.get(prediction[0], "Unknown")
    print("here",predicted_label, prediction[0])
    # Return the prediction result
    return {"prediction": predicted_label}


