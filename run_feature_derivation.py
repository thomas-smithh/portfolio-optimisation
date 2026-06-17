"""
Crash-surviving runner for the feature-derivation pipeline.

Runs ``feature_derivation.main()`` under a background MemoryMonitor that
streams per-second RSS / system-memory samples to a CSV log.  Because the log
is flushed on every line, the memory trajectory survives even if the process
is OOM-killed (which on Linux can also take VS Code's kernel down with it).

Run it from a terminal so it is independent of the editor, e.g.::

    # foreground
    python run_feature_derivation.py --data-folder Data_01_04_2026

    # detached, survives terminal/VS Code closing:
    nohup python run_feature_derivation.py --data-folder Data_01_04_2026 \
        > Data_01_04_2026/_mem_log/stdout.log 2>&1 &

After a crash, inspect the newest log under <data_folder>/_mem_log/ — the last
rows show which phase was active and how memory climbed into the ceiling.
Use ``--max-tickers N`` for a fast bounded smoke test.
"""

from __future__ import annotations

import argparse
import datetime
import traceback

from _memlog import MemoryMonitor
import feature_derivation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", default="Data_01_04_2026")
    parser.add_argument("--min-market-cap", type=float, default=0.0)
    parser.add_argument("--inference-batch-min-date", default=None)
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Memory sampling interval in seconds.",
    )
    parser.add_argument(
        "--max-tickers", type=int, default=None,
        help="If set, limit the universe to the first N tickers (smoke test).",
    )
    parser.add_argument(
        "--log-dir", default="_mem_log",
        help="Directory for the memory log (kept outside the data folder so "
             "logging never depends on data-folder permissions).",
    )
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{args.log_dir}/memlog_{ts}.csv"
    monitor = MemoryMonitor(log_path, interval=args.interval).start()
    print(f"[runner] memory log -> {log_path}")

    # Optional bounded smoke test: monkeypatch get_stock_fundamentals to keep
    # only the first N tickers, so the whole pipeline runs end-to-end quickly
    # while still exercising every memory-sensitive stage.
    if args.max_tickers is not None:
        _orig = feature_derivation.get_stock_fundamentals

        def _limited(market_cap_lower_limit: float = 0.25):
            df = _orig(market_cap_lower_limit)
            keep = df["ticker"].drop_duplicates().head(args.max_tickers)
            return df[df["ticker"].isin(keep)].copy()

        feature_derivation.get_stock_fundamentals = _limited
        print(f"[runner] SMOKE TEST: limited to first {args.max_tickers} tickers")

    try:
        feature_derivation.main(
            min_market_cap=args.min_market_cap,
            inference_batch_min_date=args.inference_batch_min_date,
            data_folder=args.data_folder,
        )
        monitor.note("DONE_OK")
        print("[runner] pipeline completed successfully.")
    except BaseException:  # noqa: BLE001 - capture everything, incl. MemoryError
        monitor.note("CRASHED")
        traceback.print_exc()
        raise
    finally:
        monitor.stop()
        print(
            f"[runner] peak total RSS: {monitor.peak_total_mb / 1000:.1f} GB "
            f"during phase '{monitor.peak_phase}' (log: {log_path})"
        )


if __name__ == "__main__":
    main()
