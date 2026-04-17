import os

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").strip()
API_URL = f"{BASE_URL}/predict"
