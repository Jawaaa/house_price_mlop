import time
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Response

from src.api.schemas import HouseFeatures, PredictionResponse, BatchPredictionResponse
from src.api.monitoring import track_request, track_prediction, get_metrics
from src.models.predict import predict_single, predict_batch, load_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="House Price Prediction API",
    description="API untuk prediksi harga rumah menggunakan model Machine Learning",
    version="1.0.0"
)

try:
    load_model()
    logger.info("Model berhasil dimuat saat startup API")
except Exception as e:
    logger.error(f"Gagal memuat model saat startup: {e}")


@app.get("/")
def root():
    logger.info("Endpoint / diakses")
    return {"message": "House Price Prediction API is running", "status": "ok"}


@app.get("/health")
def health_check():
    start = time.time()
    try:
        load_model()
        status = "healthy"
    except Exception:
        status = "unhealthy"

    duration = time.time() - start
    track_request(endpoint="/health", method="GET", status=status, duration=duration)
    logger.info(f"Health check: {status}")
    return {"status": status}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    start = time.time()
    logger.info(f"Request prediksi diterima: {features.dict()}")

    try:
        result = predict_single(features.dict())
        track_prediction()

        duration = time.time() - start
        track_request(endpoint="/predict", method="POST", status="success", duration=duration)

        return PredictionResponse(predicted_price=result)

    except Exception as e:
        duration = time.time() - start
        track_request(endpoint="/predict", method="POST", status="error", duration=duration)
        logger.error(f"Error saat prediksi: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat prediksi: {str(e)}")


@app.post("/predict-csv", response_model=BatchPredictionResponse)
async def predict_csv(file: UploadFile = File(...)):
    start = time.time()
    logger.info(f"Request prediksi batch diterima: file={file.filename}")

    if not file.filename.endswith(".csv"):
        logger.warning(f"Format file ditolak: {file.filename}")
        raise HTTPException(status_code=400, detail="File harus berformat .csv")

    try:
        df = pd.read_csv(file.file)
        predictions = predict_batch(df)

        for _ in predictions:
            track_prediction()

        duration = time.time() - start
        track_request(endpoint="/predict-csv", method="POST", status="success", duration=duration)

        return BatchPredictionResponse(
            status="success",
            n_rows=len(df),
            predictions=predictions.tolist()
        )

    except Exception as e:
        duration = time.time() - start
        track_request(endpoint="/predict-csv", method="POST", status="error", duration=duration)
        logger.error(f"Error saat prediksi batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")


@app.get("/metrics")
def metrics():
    data, content_type = get_metrics()
    return Response(content=data, media_type=content_type)