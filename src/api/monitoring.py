from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Counter = angka yang cuma bisa naik (total request masuk)
REQUEST_COUNT = Counter(
    "app_request_count",
    "Total jumlah request yang diterima API",
    ["endpoint", "method", "status"]
)

# Histogram = mencatat sebaran waktu respons (cepat/lambat)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Waktu yang dibutuhkan untuk memproses request",
    ["endpoint"]
)

# Counter khusus buat total prediksi yang berhasil
PREDICTION_COUNT = Counter(
    "app_prediction_count",
    "Total jumlah prediksi yang berhasil dilakukan"
)


def track_request(endpoint: str, method: str, status: str, duration: float):
    """Dipanggil setiap kali ada request masuk, untuk mencatat statistiknya."""
    REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)


def track_prediction():
    """Dipanggil setiap kali prediksi berhasil dilakukan."""
    PREDICTION_COUNT.inc()


def get_metrics():
    """Mengembalikan semua metrik dalam format yang bisa dibaca Prometheus."""
    return generate_latest(), CONTENT_TYPE_LATEST