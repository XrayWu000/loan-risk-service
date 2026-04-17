from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pandas as pd

from app.services.request_log_service import (
    LOG_COLUMNS,
    get_pending_upload_rows,
    mark_cases_as_processed,
    prune_processed_rows,
)
from config.cloud_config import (
    BQ_TABLE,
    GCP_DAILY_PREFIX,
    GCP_LOCAL_RETENTION_DAYS,
    GCP_MASTER_OBJECT,
    GCP_PROJECT,
    GCP_SCHEDULER_ENABLED,
    GCP_SERVICE_ACCOUNT_INFO,
    GCP_TIMEZONE,
    GCP_UPLOAD_CRON,
    GCP_UPLOAD_ENABLED,
    GCS_BUCKET,
)
from config.path_config import (
    LOCAL_LOG_PATH,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler

try:
    from apscheduler.schedulers.background import BackgroundScheduler as APSBackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:  # pragma: no cover - optional dependency for local development
    APSBackgroundScheduler = None
    CronTrigger = None

try:
    from google.api_core.exceptions import NotFound
    from google.cloud import bigquery, storage
    from google.oauth2 import service_account
except ImportError:  # pragma: no cover - optional dependency for local development
    NotFound = None
    bigquery = None
    service_account = None
    storage = None


@dataclass
class GCPUploadResult:
    status: str
    uploaded_rows: int = 0
    batch_id: str | None = None
    daily_object: str | None = None
    master_object: str | None = None
    bigquery_table: str | None = None
    processed_rows_marked: int = 0
    pruned_rows: int = 0
    skipped_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_scheduler: "BackgroundScheduler | None" = None


def _has_destination() -> bool:
    return bool(GCS_BUCKET or BQ_TABLE)


def is_gcp_upload_enabled() -> bool:
    return GCP_UPLOAD_ENABLED and _has_destination()


def get_gcp_runtime_status() -> dict[str, Any]:
    return {
        "upload_enabled": GCP_UPLOAD_ENABLED,
        "has_destination": _has_destination(),
        "scheduler_enabled": GCP_SCHEDULER_ENABLED,
        "scheduler_running": bool(_scheduler and _scheduler.running),
        "gcs_bucket": bool(GCS_BUCKET),
        "bigquery_table": bool(BQ_TABLE),
    }


def _build_credentials():
    if GCP_SERVICE_ACCOUNT_INFO:
        if service_account is None:
            raise RuntimeError(
                "google-cloud dependencies are not installed. Install google-cloud-storage and "
                "google-cloud-bigquery first."
            )
        info = json.loads(GCP_SERVICE_ACCOUNT_INFO)
        return service_account.Credentials.from_service_account_info(info)

    return None


def _get_storage_client():
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage is not installed. Install it or disable GCP upload locally."
        )

    credentials = _build_credentials()
    return storage.Client(project=GCP_PROJECT or None, credentials=credentials)


def _get_bigquery_client():
    if bigquery is None:
        raise RuntimeError(
            "google-cloud-bigquery is not installed. Install it or disable BigQuery sync locally."
        )

    credentials = _build_credentials()
    return bigquery.Client(project=GCP_PROJECT or None, credentials=credentials)


def _compute_batch_id(df: pd.DataFrame) -> str:
    case_ids = df["case_id"].fillna("").astype(str).sort_values().tolist()
    digest_source = "|".join(case_ids) or f"rows:{len(df)}"
    return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]


def _write_snapshot(df: pd.DataFrame) -> tuple[str, str]:
    export_df = df.copy()
    export_df = export_df.reindex(columns=LOG_COLUMNS)
    batch_id = _compute_batch_id(export_df)

    temp_dir = Path(tempfile.gettempdir())
    snapshot_path = temp_dir / f"loan_requests_{batch_id}.csv"
    export_df.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
    return str(snapshot_path), batch_id


def _daily_object_name(batch_id: str) -> str:
    date_prefix = datetime.now(ZoneInfo(GCP_TIMEZONE)).strftime("%Y%m%d")
    file_name = f"{Path(LOCAL_LOG_PATH).stem}_{batch_id}.csv"
    return f"{GCP_DAILY_PREFIX}/{date_prefix}/{file_name}".strip("/")


def _upload_daily_snapshot(snapshot_path: str, batch_id: str) -> str:
    client = _get_storage_client()
    bucket = client.bucket(GCS_BUCKET)
    object_name = _daily_object_name(batch_id)
    bucket.blob(object_name).upload_from_filename(snapshot_path, content_type="text/csv")
    return object_name


def _upload_master_csv(snapshot_df: pd.DataFrame, batch_id: str) -> str:
    client = _get_storage_client()
    bucket = client.bucket(GCS_BUCKET)
    master_blob = bucket.blob(GCP_MASTER_OBJECT)

    temp_master = Path(tempfile.gettempdir()) / f"loan_requests_master_{batch_id}.csv"

    try:
        if master_blob.exists():
            master_blob.download_to_filename(str(temp_master))
            master_df = pd.read_csv(temp_master, encoding="utf-8-sig")
            master_df = master_df.reindex(columns=LOG_COLUMNS)
            combined = pd.concat([master_df, snapshot_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["case_id"], keep="last")
        else:
            combined = snapshot_df.copy()

        combined = combined.reindex(columns=LOG_COLUMNS)
        combined.to_csv(temp_master, index=False, encoding="utf-8-sig")
        master_blob.upload_from_filename(str(temp_master), content_type="text/csv")
    finally:
        if temp_master.exists():
            temp_master.unlink()

    return GCP_MASTER_OBJECT


def _table_exists(client, table_id: str) -> bool:
    try:
        client.get_table(table_id)
        return True
    except Exception as exc:
        if NotFound is not None and isinstance(exc, NotFound):
            return False
        raise


def _load_csv_to_bigquery(client, table_id: str, snapshot_path: str, write_disposition: str) -> None:
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=write_disposition,
    )

    with open(snapshot_path, "rb") as csv_file:
        client.load_table_from_file(csv_file, table_id, job_config=job_config).result()


def _merge_bigquery_table(client, target_table: str, staging_table: str) -> None:
    update_clause = ", ".join(f"target.{column} = source.{column}" for column in LOG_COLUMNS)
    insert_columns = ", ".join(LOG_COLUMNS)
    insert_values = ", ".join(f"source.{column}" for column in LOG_COLUMNS)

    query = f"""
    MERGE `{target_table}` AS target
    USING `{staging_table}` AS source
    ON target.case_id = source.case_id
    WHEN MATCHED THEN
      UPDATE SET {update_clause}
    WHEN NOT MATCHED THEN
      INSERT ({insert_columns})
      VALUES ({insert_values})
    """
    client.query(query).result()


def _upsert_bigquery(snapshot_path: str, batch_id: str) -> str:
    client = _get_bigquery_client()
    table_id = BQ_TABLE

    if not _table_exists(client, table_id):
        _load_csv_to_bigquery(client, table_id, snapshot_path, bigquery.WriteDisposition.WRITE_APPEND)
        return table_id

    table_ref = bigquery.TableReference.from_string(table_id, default_project=GCP_PROJECT or None)
    staging_table_id = (
        f"{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}__staging_{batch_id}"
    )

    try:
        _load_csv_to_bigquery(
            client,
            staging_table_id,
            snapshot_path,
            bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        _merge_bigquery_table(client, table_id, staging_table_id)
    finally:
        client.delete_table(staging_table_id, not_found_ok=True)

    return table_id


def upload_pending_logs_to_gcp() -> dict[str, Any]:
    if not GCP_UPLOAD_ENABLED:
        return GCPUploadResult(status="skipped", skipped_reason="gcp_upload_disabled").to_dict()

    if not _has_destination():
        return GCPUploadResult(status="skipped", skipped_reason="no_gcp_destination").to_dict()

    pending_df = get_pending_upload_rows()
    if pending_df.empty:
        return GCPUploadResult(status="skipped", skipped_reason="no_pending_rows").to_dict()

    snapshot_path, batch_id = _write_snapshot(pending_df)
    result = GCPUploadResult(
        status="failed",
        uploaded_rows=len(pending_df),
        batch_id=batch_id,
        bigquery_table=BQ_TABLE or None,
        master_object=GCP_MASTER_OBJECT if GCS_BUCKET else None,
    )

    try:
        if GCS_BUCKET:
            result.daily_object = _upload_daily_snapshot(snapshot_path, batch_id)
            result.master_object = _upload_master_csv(pending_df, batch_id)

        if BQ_TABLE:
            _upsert_bigquery(snapshot_path, batch_id)

        result.processed_rows_marked = mark_cases_as_processed(pending_df["case_id"].tolist())
        result.pruned_rows = prune_processed_rows(GCP_LOCAL_RETENTION_DAYS)
        result.status = "success"
        return result.to_dict()
    except Exception as exc:
        logger.exception("GCP upload failed")
        result.errors.append(str(exc))
        return result.to_dict()
    finally:
        if os.path.exists(snapshot_path):
            os.remove(snapshot_path)


def start_gcp_upload_scheduler() -> bool:
    global _scheduler

    if _scheduler and _scheduler.running:
        return True

    if not GCP_SCHEDULER_ENABLED or not is_gcp_upload_enabled():
        logger.info(
            "GCP upload scheduler not started. enabled=%s destination=%s",
            GCP_SCHEDULER_ENABLED,
            _has_destination(),
        )
        return False

    if APSBackgroundScheduler is None or CronTrigger is None:
        logger.warning("APScheduler is not installed. Scheduler startup skipped.")
        return False

    try:
        timezone = ZoneInfo(GCP_TIMEZONE)
        trigger = CronTrigger.from_crontab(GCP_UPLOAD_CRON, timezone=timezone)
    except Exception:
        logger.exception(
            "Invalid scheduler configuration. cron=%s timezone=%s",
            GCP_UPLOAD_CRON,
            GCP_TIMEZONE,
        )
        return False

    _scheduler = APSBackgroundScheduler(timezone=timezone)
    _scheduler.add_job(
        upload_pending_logs_to_gcp,
        trigger=trigger,
        id="gcp_log_upload",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    _scheduler.start()
    logger.info(
        "GCP upload scheduler started with cron=%s timezone=%s",
        GCP_UPLOAD_CRON,
        GCP_TIMEZONE,
    )
    return True


def stop_gcp_upload_scheduler() -> None:
    global _scheduler

    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)

    _scheduler = None
