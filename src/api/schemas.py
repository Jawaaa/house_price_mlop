from pydantic import BaseModel, Field
from typing import List


class HouseFeatures(BaseModel):
    """Bentuk data yang wajib dikirim user untuk prediksi 1 rumah."""
    OverallQual: int = Field(..., ge=1, le=10, description="Kualitas keseluruhan, skala 1-10")
    GrLivArea: float = Field(..., gt=0, description="Luas area tinggal (sq ft)")
    GarageCars: int = Field(..., ge=0, description="Kapasitas garasi (jumlah mobil)")
    GarageArea: float = Field(..., ge=0)
    TotalBsmtSF: float = Field(..., ge=0, description="Luas basement (sq ft)")
    FullBath: int = Field(..., ge=0)
    YearBuilt: int = Field(..., ge=1800, le=2026)
    Neighborhood: str

    class Config:
        json_schema_extra = {
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1710,
                "GarageCars": 2,
                "GarageArea": 548,
                "TotalBsmtSF": 856,
                "FullBath": 2,
                "YearBuilt": 2003,
                "Neighborhood": "CollgCr"
            }
        }


class PredictionResponse(BaseModel):
    """Bentuk data yang dikembalikan API setelah prediksi berhasil."""
    predicted_price: float
    status: str = "success"


class BatchPredictionResponse(BaseModel):
    """Bentuk data yang dikembalikan setelah prediksi banyak baris (upload CSV)."""
    status: str = "success"
    n_rows: int
    predictions: List[float]