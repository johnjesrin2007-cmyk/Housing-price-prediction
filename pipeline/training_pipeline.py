import os
import sys
import mlflow

# -------------------------
# 📦 IMPORT PATH FIX
# -------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -------------------------
# 📥 IMPORT MODULES
# -------------------------
from src.preprocess import preprocess_data, get_preprocessor
from src.train import train_model
from src.model import get_model_pipeline


# -------------------------
# 🚀 TRAINING PIPELINE
# -------------------------
def training_pipeline():

    print("🚀 Starting training pipeline...")

    # -------------------------
    # 🔥 MLFLOW CONFIG
    # -------------------------
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Housing_price_prediction")

    # -------------------------
    # 📁 DATA PATH
    # -------------------------
    data_path = "data/raw.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found")

    # -------------------------
    # 📥 LOAD DATA
    # -------------------------
    X, y = preprocess_data(
        path=data_path,
        training=True,
        target_col="price_lakhs"
    )

    print("📊 Data loaded and preprocessed")

    # -------------------------
    # 🧠 PREPROCESSOR + PIPELINE
    # -------------------------
    preprocess = get_preprocessor(X)
    pipeline = get_model_pipeline(preprocess)

    # -------------------------
    # 🤖 TRAIN MODEL
    # -------------------------
    try:
        trained_pipeline = train_model(pipeline, X, y)
        print("✅ Model training completed")
    except Exception as e:
        print("❌ Training failed:", str(e))
        raise

    return trained_pipeline


# -------------------------
# ▶️ LOCAL RUN
# -------------------------
if __name__ == "__main__":
    training_pipeline()