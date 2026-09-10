import os
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import MODEL_PATH, MODEL_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.data.preprocessing import load_and_clean_data, build_preprocessor, split_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_all_models():
    # kasih tau MLflow mau nyimpen catatannya di mana
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_and_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor(X_train)

    model_candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    best_model_name = None
    best_r2 = -np.inf
    best_pipe = None

    for name, model in model_candidates.items():
        # with mlflow.start_run() artinya "mulai catatan eksperimen baru"
        with mlflow.start_run(run_name=name):
            logger.info(f"Mulai training model: {name}")

            pipe = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # catat parameter model ke MLflow
            mlflow.log_param("model_type", name)
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)

            # catat hasil metrik ke MLflow
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # simpan model itu sendiri sebagai "artifact" di MLflow
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                serialization_format="pickle"  # pakai pickle biasa, bukan skops
            )
            
            logger.info(f"{name} -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.3f}")

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_pipe = pipe

    logger.info(f"Model terbaik: {best_model_name} (R2={best_r2:.3f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipe, f)
    logger.info(f"Model terbaik disimpan ke {MODEL_PATH}")

    return best_model_name, best_r2


if __name__ == "__main__":
    train_all_models()