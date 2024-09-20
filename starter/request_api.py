import requests

# Define the URL of your live API
#url = "http://0.0.0.0:8000/predict"  # Replace with the actual URL of your API
url = "https://build-ml-ops1.onrender.com/predict"

# Define the data payload for the POST request
data = {
    "age": 35,
    "workclass": "Private",
    "fnlwgt": 215646,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response from the API
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Failed to get a response. Status code:", response.status_code)
    print("Response content:", response.content)
