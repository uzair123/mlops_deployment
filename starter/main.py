from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd

# Initialize the FastAPI app
app = FastAPI()

try:
 with open('model.pkl', 'rb') as file:
     # Load or process the file
     # Load your pre-trained model (assuming a RandomForestClassifier for this example)
     model = joblib.load('../starter/model.pkl')
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

    class Config:
        schema_extra = {
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlwgt": 284582,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 60,
                "native-country": "United-States"
            }
        }


# Root endpoint to return a welcome message
@app.get("/")
async def root():
    return {"message": "Welcome to the Machine Learning Inference API!"}


# POST endpoint for model inference
@app.post("/predict")
async def predict(data: InferenceData):
    # Convert incoming data to DataFrame for model input
    input_data = pd.DataFrame([data.dict(by_alias=True)])  # Use by_alias=True to handle hyphenated fields

    # Perform model inference
    prediction = model.predict(input_data)

    # Return the prediction result
    return {"prediction": prediction.tolist()}


