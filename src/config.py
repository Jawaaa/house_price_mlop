import os

# BASE_DIR = folder utama project, dihitung otomatis dari lokasi file ini
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lokasi data
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

# Lokasi model tersimpan
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

# Lokasi log
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Pengaturan MLflow
MLFLOW_TRACKING_URI = "sqlite:///" + os.path.join(BASE_DIR, "mlflow.db").replace("\\", "/")
MLFLOW_EXPERIMENT_NAME = "house_price_prediction"

# Pengaturan training
RANDOM_STATE = 42
TEST_SIZE = 0.2