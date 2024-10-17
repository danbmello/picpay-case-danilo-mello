import json
import os
import pickle

from contextlib import asynccontextmanager

import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from src.database import InMemoryDatabase

# Initialize Spark session for working with PySpark
spark = SparkSession.builder.appName("FlightDelayPredictionAPI").getOrCreate()

# Set Spark logging level to reduce the verbosity of logs
spark.sparkContext.setLogLevel("ERROR")


# Use async context manager to manage lifespan of the app and load the model
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_type = os.getenv("MODEL_TYPE", "pyspark")  # Get the model type (PySpark or Pickle) from environment variables
    model = None
    try:
        # Load PySpark model if 'MODEL_TYPE' is set to 'pyspark'
        if model_type == "pyspark":
            model_path = "./models/lr_model_pyspark"
            model = PipelineModel.load(model_path)
            print("PySpark PipelineModel loaded successfully.")
        # Load Pickle model if 'MODEL_TYPE' is set to 'pickle'
        elif model_type == "pickle":
            model_path = "./models/lr_model.pkl"
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            print("Pickle model loaded successfully.")
        else:
            print(f"Unknown MODEL_TYPE: {model_type}")  # Handle unknown model types
    except Exception as e:
        print(f"Error loading model: {e}")  # Log any errors that occur during model loading

    yield  # Yield control to the application during runtime
    print("Application is shutting down.")  # Log when the application is closing


# Initialize FastAPI app with the lifespan context
app = FastAPI(lifespan=lifespan)

# Store the model and prediction history in memory
model = None
prediction_history = []


# Define the input data structure for predictions using Pydantic
class FlightData(BaseModel):
    dep_delay: float
    origin_wind_speed: float
    dest_wind_speed: float
    distance: float
    carrier: str


# Health check endpoint to confirm the service is running
@app.get("/health", status_code=200, tags=["health"], summary="Health check")
async def health():
    return {"status": "ok"}


# Endpoint to make a prediction using the loaded model
@app.post("/model/predict/")
async def predict(data: FlightData):

    # Prepare the input features for prediction
    input_data = {
        "dep_delay": data.dep_delay,
        "origin_wind_speed": data.origin_wind_speed,
        "dest_wind_speed": data.dest_wind_speed,
        "distance": data.distance,
        "carrier": data.carrier,
    }

    model_type = os.getenv("MODEL_TYPE", "pyspark")  # Determine which model type to use

    # Use the PySpark model for prediction
    if model_type == "pyspark":
        try:
            input_df = spark.createDataFrame([input_data])  # Create a Spark DataFrame from the input data
            prediction = model.transform(input_df)  # Use the PySpark model to transform (predict) the data
            predicted_value = prediction.collect()[0]["prediction"]  # Extract the prediction result
        except Exception as e:
            print(f"PySpark prediction error: {e}")  # Log errors if prediction fails
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Use the Pickle model for prediction
    elif model_type == "pickle":
        try:
            df_input = pd.DataFrame([input_data])  # Create a Pandas DataFrame from the input data
            predicted_value = model.predict(df_input)[0]  # Use the Pickle model to predict
        except Exception as e:
            print(f"Pickle prediction error: {e}")  # Log errors if prediction fails
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")  # Handle unknown model types

    # Store the input data and the prediction result in memory
    prediction_history.append({"input": input_data, "prediction": predicted_value})

    return {"prediction": predicted_value}  # Return the prediction result


# Endpoint to retrieve the prediction history
@app.get("/model/history/", tags=["model"], summary="Prediction history")
async def get_history():
    json_history = jsonable_encoder({"history": prediction_history})  # Encode the prediction history to JSON format
    formatted_history = json.dumps(json_history, indent=2)  # Format the JSON with indentation
    return Response(content=formatted_history, media_type="application/json")  # Return the formatted history


# Example endpoint to insert a user into an in-memory database
@app.post("/user/", tags=["example"], summary="Insert user")
async def insert(data: dict):
    db = InMemoryDatabase()
    users = db.get_collection("users")
    users.insert_one(data)  # Insert the provided user data into the database
    return {"status": "ok"}


# Endpoint to get a user by their name from the in-memory database
@app.get("/user/{name}", status_code=200, tags=["example"], summary="Get user by name")
async def get(name: str):
    db = InMemoryDatabase()
    users = db.get_collection("users")
    user = users.find_one({"name": name})  # Find the user in the database by name
    json_user = jsonable_encoder({"status": "ok", "user": user})  # Encode the user data as JSON
    formatted_user = json.dumps(json_user, indent=4)  # Format the JSON with indentation
    return Response(content=formatted_user, media_type="application/json")  # Return the user data


# Endpoint to list all users in the in-memory database
@app.get("/user/", tags=["example"], summary="List all users")
async def list():
    db = InMemoryDatabase()
    users = db.get_collection("users")
    json_users = jsonable_encoder(
        {"status": "ok", "users": [x for x in users.find({}, {"_id": 0})]}
    )  # Encode all users as JSON
    formatted_users = json.dumps(json_users, indent=4)  # Format the JSON with indentation
    return Response(content=formatted_users, media_type="application/json")  # Return the list of users


# Entry point for running the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")  # Run the app on port 8080 using Uvicorn
