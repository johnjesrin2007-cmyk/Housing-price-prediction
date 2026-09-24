import os
import pandas as pd
from datetime import datetime

LOG_FILE = "logs/predictions.csv"


def log_prediction(input_data: dict, prediction: float):
    os.makedirs("logs", exist_ok=True)

    log_data = input_data.copy()
    log_data["prediction"] = prediction
    log_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([log_data])

    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)