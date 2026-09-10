# 🏠 House Price Prediction — MLOps Pipeline

Sistem prediksi harga rumah menggunakan Machine Learning, dibangun dengan
menerapkan praktik **MLOps** (proses kerja profesional di industri data
science): kode modular, pelacakan eksperimen (MLflow), sistem logging,
layanan API (FastAPI), pemantauan sistem (Prometheus), dan kontainerisasi
(Docker).

Dataset: [House Prices - Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## 📁 Struktur Project

<img width="662" height="634" alt="image" src="https://github.com/user-attachments/assets/9f386798-7c91-4bd4-aa9e-cc7878cdbdc9" />



---

## 🧠 Machine Learning & MLflow

| Tahap | Detail |
|---|---|
| Dataset | House Prices (1460 baris, 80 kolom) |
| Model dibandingkan | Linear Regression, Random Forest, Gradient Boosting |
| Tracking | MLflow — parameter, metrik (RMSE, MAE, R²), artifact model |
| Model terbaik | **Gradient Boosting** (R² = 0.897) |
| Penyimpanan | Model terbaik disimpan otomatis ke `models/best_model.pkl` |

---

## 🚀 Cara Menjalankan

### 1. Setup lingkungan lokal
```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

### 2. Training model (+ pencatatan ke MLflow)
```bash
python -m src.models.train
```

Lihat hasil eksperimen di MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```
Buka `http://localhost:5001`

### 3. Menjalankan API secara lokal
```bash
uvicorn src.api.main:app --reload --port 8000
```
Buka `http://localhost:8000/docs` untuk mencoba endpoint lewat Swagger UI.

### 4. Menjalankan unit test
```bash
pytest tests/ -v
```

### 5. Menjalankan dengan Docker (FastAPI + Prometheus)
```bash
docker-compose up --build
```
- FastAPI: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`

Untuk menghentikan:
```bash
docker-compose down
```

---

## 📡 API Endpoints

| Method | Endpoint | Fungsi |
|---|---|---|
| GET | `/` | Mengecek status API |
| GET | `/health` | Mengecek kesehatan model |
| POST | `/predict` | Prediksi untuk satu rumah (validasi otomatis via Pydantic) |
| POST | `/predict-csv` | Prediksi untuk banyak rumah sekaligus lewat file CSV |
| GET | `/metrics` | Data metrik untuk dipantau Prometheus |

---

## 📊 Sistem Logging & Monitoring

- **Logging**: setiap tahap (pembersihan data, training, permintaan prediksi)
  dicatat secara terstruktur ke file di folder `logs/`, sehingga proses
  dapat ditelusuri jika terjadi masalah.
- **Monitoring**: Prometheus memantau performa API setiap 15 detik —
  jumlah request masuk dan waktu respons — melalui endpoint `/metrics`.

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `MLflow` · `FastAPI` · `Pydantic` ·
`Prometheus` · `Docker` · `Pytest`

---

## 👩‍💻 Author

**Zahwa Rizzi Ani** — Data Science & Machine Learning Bootcamp
