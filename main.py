from fastapi import FastAPI
import mlflow
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
import pandas as pd

# -------------------------
# 🚀 APP INIT
# -------------------------
app = FastAPI(title="Housing price prediction API")


# -------------------------
# 🔥 SAFE MODEL LOADING
# -------------------------
model = None


def load_latest_model():
    try:
        client = MlflowClient()

        experiment = client.get_experiment_by_name(
            "Housing_price_prediction"
        )

        if experiment is None:
            raise Exception("Experiment not found")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        if len(runs) == 0:
            raise Exception("No MLflow runs found")

        latest_run_id = runs[0].info.run_id
        model_uri = f"runs:/{latest_run_id}/model"

        loaded_model = mlflow.pyfunc.load_model(model_uri)

        print(f" Model loaded from run: {latest_run_id}")
        return loaded_model

    except Exception as e:
        print(" Model loading failed:", e)
        return None


# Load model safely (IMPORTANT FIX FOR 502)
try:
    model = load_latest_model()
except Exception as e:
    print("Startup model load skipped:", e)
    model = None



class InputData(BaseModel):
    area_sqft:int
    bedrooms:int
    age_years:int
    distance_to_city_km:float
    


# -------------------------
# 🏠 HEALTH CHECK
# -------------------------
@app.get("/")
def home():
    return {
        "message": "Housing price API is running",
        "model_loaded": model is not None
    }


# -------------------------
# 🔮 PREDICTION ROUTE
# -------------------------
@app.post("/predict")
def predict(data: InputData):

    if model is None:
        return {
            "error": "Model not loaded. Please check training pipeline."
        }

    try:
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)[0]

        # monitoring
        try:
            from src.logger import log_prediction
            log_prediction(input_dict, float(prediction))
        except Exception as log_error:
            print("Logging failed:", log_error)

        return {
            "input": input_dict,
            "predicted_price_lakhs": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}