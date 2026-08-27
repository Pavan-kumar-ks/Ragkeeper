import time
import traceback
from datetime import datetime, timezone

from .ingest import run_ingestion


def run_scheduler(interval_hours: float) -> None:
    interval_s = interval_hours * 3600
    print(f"RAGKeeper scheduler started — syncing every {interval_hours}h. Ctrl+C to stop.")

    while True:
        started = datetime.now(timezone.utc).isoformat()
        print(f"\n[{started}] Running scheduled ingest...")
        try:
            run_ingestion(recreate=False)
        except Exception:
            print(f"[{started}] Scheduled ingest failed:")
            traceback.print_exc()

        print(f"Sleeping {interval_hours}h until next sync...")
        time.sleep(interval_s)
