import pickle
import pandas as pd

from src.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

_model = None

# UPDATE: nilai default untuk kolom yang tidak diminta lewat form API
# (nilai diambil dari nilai umum/median dataset House Prices)
DEFAULT_FEATURES = {
    "MSSubClass": 60, "MSZoning": "RL", "LotFrontage": 70.0, "LotArea": 9600,
    "Street": "Pave", "Alley": "None", "LotShape": "Reg", "LandContour": "Lvl",
    "Utilities": "AllPub", "LotConfig": "Inside", "LandSlope": "Gtl",
    "Condition1": "Norm", "Condition2": "Norm", "BldgType": "1Fam",
    "HouseStyle": "1Story", "OverallCond": 5, "YearRemodAdd": 2003,
    "RoofStyle": "Gable", "RoofMatl": "CompShg", "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd", "MasVnrType": "None", "MasVnrArea": 0.0,
    "ExterQual": "TA", "ExterCond": "TA", "Foundation": "PConc",
    "BsmtQual": "TA", "BsmtCond": "TA", "BsmtExposure": "No",
    "BsmtFinType1": "Unf", "BsmtFinSF1": 0, "BsmtFinType2": "Unf",
    "BsmtFinSF2": 0, "BsmtUnfSF": 500, "Heating": "GasA", "HeatingQC": "TA",
    "CentralAir": "Y", "Electrical": "SBrkr", "1stFlrSF": 1200, "2ndFlrSF": 0,
    "LowQualFinSF": 0, "BsmtFullBath": 0, "BsmtHalfBath": 0, "HalfBath": 0,
    "BedroomAbvGr": 3, "KitchenAbvGr": 1, "KitchenQual": "TA",
    "TotRmsAbvGrd": 6, "Functional": "Typ", "Fireplaces": 0,
    "FireplaceQu": "None", "GarageType": "Attchd", "GarageYrBlt": 2003.0,
    "GarageFinish": "Unf", "GarageQual": "TA", "GarageCond": "TA",
    "PavedDrive": "Y", "WoodDeckSF": 0, "OpenPorchSF": 0, "EnclosedPorch": 0,
    "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0, "PoolQC": "None",
    "Fence": "None", "MiscFeature": "None", "MiscVal": 0, "MoSold": 6,
    "YrSold": 2008, "SaleType": "WD", "SaleCondition": "Normal"
}


def load_model():
    global _model
    if _model is None:
        logger.info(f"Memuat model dari {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        logger.info("Model berhasil dimuat")
    return _model


def predict_single(features: dict) -> float:
    model = load_model()

    # UPDATE: gabungkan input user dengan nilai default untuk kolom yang tidak diisi
    full_features = {**DEFAULT_FEATURES, **features}

    df = pd.DataFrame([full_features])
    prediction = model.predict(df)[0]
    logger.info(f"Prediksi untuk input {features} -> {prediction:.2f}")
    return float(prediction)


def predict_batch(df: pd.DataFrame):
    model = load_model()
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    if "SalePrice" in df.columns:
        df = df.drop(columns=["SalePrice"])
    predictions = model.predict(df)
    logger.info(f"Prediksi batch selesai untuk {len(df)} baris")
    return predictions