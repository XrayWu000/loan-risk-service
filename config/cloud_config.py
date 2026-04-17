import os

from config.base_config import get_bool, get_int

GCP_UPLOAD_ENABLED = get_bool("GCP_UPLOAD_ENABLED", default=False)
GCP_SCHEDULER_ENABLED = get_bool("GCP_SCHEDULER_ENABLED", default=False)
GCP_UPLOAD_CRON = os.getenv("GCP_UPLOAD_CRON", "0 21 * * *").strip()
GCP_TIMEZONE = os.getenv("GCP_TIMEZONE", "Asia/Taipei").strip()
GCP_PROJECT = os.getenv("GCP_PROJECT", "").strip()
GCP_SERVICE_ACCOUNT_INFO = os.getenv("GCP_SERVICE_ACCOUNT_INFO", "").strip()
GCS_BUCKET = os.getenv("GCS_BUCKET", "").strip()
BQ_TABLE = os.getenv("BQ_TABLE", "").strip()
GCP_DAILY_PREFIX = os.getenv("GCP_DAILY_PREFIX", "loan-requests/daily").strip().strip("/")
GCP_MASTER_OBJECT = os.getenv("GCP_MASTER_OBJECT", "master/master_all.csv").strip()
GCP_LOCAL_RETENTION_DAYS = get_int("GCP_LOCAL_RETENTION_DAYS", default=0)
