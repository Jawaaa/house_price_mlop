import pickle
import pandas as pd

from src.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

_model = None  # cache biar model cuma di-load sekali


def load_model():
    """Muat model dari file .pkl. Hanya dijalankan sekali (di-cache di memori)."""
    global _model
    if _model is None:
        logger.info(f"Memuat model dari {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        logger.info("Model berhasil dimuat")
    return _model


def predict_single(features: dict) -> float:
    """Prediksi untuk 1 baris data (dipakai oleh endpoint FastAPI)."""
    model = load_model()
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    logger.info(f"Prediksi untuk input {features} -> {prediction:.2f}")
    return float(prediction)


def predict_batch(df: pd.DataFrame):
    """Prediksi untuk banyak baris sekaligus (dipakai untuk upload CSV)."""
    model = load_model()
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    if "SalePrice" in df.columns:
        df = df.drop(columns=["SalePrice"])
    predictions = model.predict(df)
    logger.info(f"Prediksi batch selesai untuk {len(df)} baris")
    return predictions