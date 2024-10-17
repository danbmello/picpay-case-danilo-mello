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
    # Get the model type from environment variables
    model_type = os.getenv("MODEL_TYPE", "pyspark")
    model = None
    try:
        if model_type == "pyspark":
            model_path = "./models/lr_model_pyspark"
            model = PipelineModel.load(model_path)
            print("PySpark PipelineModel loaded successfully.")
        elif model_type == "pickle":
            model_path = "./models/lr_model.pkl"
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            print("Pickle model loaded successfully.")
        # Handle unknown model types
        else:
            print(f"Unknown MODEL_TYPE: {model_type}")
    except Exception as e:
        # Log any errors that occur during model loading
        print(f"Error loading model: {e}")

    # Yield control to the application during runtime
    yield
    # Log when the application is closing
    print("Application is shutting down.")


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

    # Determine which model type to use
    model_type = os.getenv("MODEL_TYPE", "pyspark")

    # Use the PySpark model for prediction
    if model_type == "pyspark":
        try:
            # Create a Spark DataFrame from the input data
            input_df = spark.createDataFrame([input_data])
            # Use the PySpark model to transform (predict) the data
            prediction = model.transform(input_df)
            # Extract the prediction result
            predicted_value = prediction.collect()[0]["prediction"]
        except Exception as e:
            # Log errors if prediction fails
            print(f"PySpark prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Use the Pickle model for prediction
    elif model_type == "pickle":
        try:
            # Create a Pandas DataFrame from the input data
            df_input = pd.DataFrame([input_data])
            # Use the Pickle model to predict
            predicted_value = model.predict(df_input)[0]
        except Exception as e:
            # Log errors if prediction fails
            print(f"Pickle prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    else:
        # Handle unknown model types
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

    # Store the input data and the prediction result in memory
    prediction_history.append({"input": input_data, "prediction": predicted_value})

    # Return the prediction result
    return {"prediction": predicted_value}


# Endpoint to retrieve the prediction history
@app.get("/model/history/", tags=["model"], summary="Prediction history")
async def get_history():
    # Encode the prediction history to JSON format
    json_history = jsonable_encoder({"history": prediction_history})
    # Format the JSON with indentation
    formatted_history = json.dumps(json_history, indent=2)
    # Return the formatted history
    return Response(content=formatted_history, media_type="application/json")


# Example endpoint to insert a user into an in-memory database
@app.post("/user/", tags=["example"], summary="Insert user")
async def insert(data: dict):
    db = InMemoryDatabase()
    users = db.get_collection("users")
    # Insert the provided user data into the database
    users.insert_one(data)
    return {"status": "ok"}


# Endpoint to get a user by their name from the in-memory database
@app.get("/user/{name}", status_code=200, tags=["example"], summary="Get user by name")
async def get(name: str):
    db = InMemoryDatabase()
    users = db.get_collection("users")
    # Find the user in the database by name
    user = users.find_one({"name": name})
    # Encode the user data as JSON
    json_user = jsonable_encoder({"status": "ok", "user": user})
    # Format the JSON with indentation
    formatted_user = json.dumps(json_user, indent=4)

    # Return the user data
    return Response(content=formatted_user, media_type="application/json")


# Endpoint to list all users in the in-memory database
@app.get("/user/", tags=["example"], summary="List all users")
async def list():
    db = InMemoryDatabase()
    users = db.get_collection("users")
    # Encode all users as JSON
    json_users = jsonable_encoder({"status": "ok", "users": [x for x in users.find({}, {"_id": 0})]})
    # Format the JSON with indentation
    formatted_users = json.dumps(json_users, indent=4)

    # Return the list of users
    return Response(content=formatted_users, media_type="application/json")


# Entry point for running the FastAPI app
if __name__ == "__main__":
    # Run the app on port 8080 using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
