
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('model.pkl', 'rb') as file:
    model_data = pickle.load(file)
    brand_models = model_data['models']
    brand_encoders = model_data['encoders']

class UserMeasurements(BaseModel):
    chest: float
    shoulder: float
    front_length: float
    sleeve_length: float

@app.get('/')
async def root():
    return {"message": "API is up and running"}

@app.post('/predict')
async def predict_size(user: UserMeasurements):
    user_input = pd.DataFrame([{
        'Chest': user.chest,
        'Shoulder': user.shoulder,
        'Front_length': user.front_length,
        'Sleeve_length': user.sleeve_length
    }])

    results = {}

    for brand in brand_models:
        model = brand_models[brand]
        encoder = brand_encoders[brand]

        predicted_size_num = model.predict(user_input)[0]
        
        predicted_size = encoder.inverse_transform([predicted_size_num])[0]

        results[brand] = predicted_size

    return {"predicted_sizes": results}

