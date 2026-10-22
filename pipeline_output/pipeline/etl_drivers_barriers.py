"""
ETL: Drivers & Barriers page
Copies the latest aggregated drivers/barriers table written by
table_analysis/03_driver_barrier_table_w_counts.ipynb into the app's data/ folder.
No respondent-level data is read.

Reads:   <DIR_FINAL_TABLES>/<COUNTRY>_drivers-barriers_table_<date>.csv
         (the grouped-non-users variant, *_grpd_nonusers_*, is skipped)
Writes:  <APP_DATA_DIR>/<COUNTRY>_drivers-barriers_table_<date>.csv
"""

import glob
import os
import shutil
import sys

# Allow running this file directly (python pipeline/etl_drivers_barriers.py), not only via run_pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import APP_DATA_DIR, DIR_FINAL_TABLES, COUNTRY


def run(df=None):
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    print("  [drivers_barriers] running...")

    pattern = os.path.join(DIR_FINAL_TABLES, f"{COUNTRY}_drivers-barriers_table_*.csv")
    candidates = sorted(p for p in glob.glob(pattern) if "_grpd_" not in os.path.basename(p))
    if not candidates:
        grouped = sorted(os.path.basename(p) for p in glob.glob(pattern) if "_grpd_" in os.path.basename(p))
        print(f"  [drivers_barriers] WARNING: no file matching {pattern} -- "
              f"run table_analysis/03_driver_barrier_table_w_counts.ipynb to the end, answering 0 "
              f"(don't group non-users) at its prompt.")
        if grouped:
            print(f"  Only the grouped-non-users version exists, which the app doesn't use: {grouped}")
        return

    # Dates are yy_mm_dd, so the last name in sort order is the newest.
    src = candidates[-1]
    dst = os.path.join(APP_DATA_DIR, os.path.basename(src))
    shutil.copyfile(src, dst)
    print(f"  Copied: {os.path.basename(src)} -> {APP_DATA_DIR}")
    print("  [drivers_barriers] done.")


if __name__ == "__main__":
    run()
