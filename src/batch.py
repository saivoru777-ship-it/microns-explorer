"""Batch processing: download, format, and validate multiple neurons.

Supports resume capability, rate limiting, progress tracking, and error logging.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.discovery import _classify_broad as classify_broad
from src.fetch import fetch_neuron
from src.format import make_label, save_neuron_files
from src.validate import validate_neuron

logger = logging.getLogger(__name__)

PROGRESS_FILE = "batch_progress.json"
ERROR_LOG = "errors.log"
RATE_LIMIT_DELAY = 1.5  # seconds between neurons


def _load_progress(output_dir):
    """Load batch progress from JSON file."""
    progress_path = Path(output_dir) / PROGRESS_FILE
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "start_time": None}


def _save_progress(output_dir, progress):
    """Save batch progress to JSON file."""
    progress_path = Path(output_dir) / PROGRESS_FILE
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2, default=str)


def _log_error(output_dir, root_id, label, category, message):
    """Append error to error log."""
    error_path = Path(output_dir) / ERROR_LOG
    timestamp = datetime.now().isoformat()
    with open(error_path, "a") as f:
        f.write(f"{timestamp} | {root_id} | {label} | {category} | {message}\n")


def fetch_batch(client, catalog_df, output_dir, filters=None,
                max_neurons=None, resume=True, validate=True):
    """Download and format neurons matching filters. Resumable.

    Parameters
    ----------
    client : CAVEclient
    catalog_df : pd.DataFrame
        Neuron catalog from discovery.build_catalog().
    output_dir : str or Path
        Directory to save neuron files (data/neurons/).
    filters : dict, optional
        Filter criteria:
        - cell_type_broad: str or list of str
        - cell_type_fine: str or list of str
        - proofread_only: bool (default True)
        - root_ids: list of int (specific neurons)
    max_neurons : int, optional
        Maximum neurons to download.
    resume : bool
        If True, skip already-downloaded neurons.
    validate : bool
        If True, run QC after each download.

    Returns
    -------
    dict with 'completed', 'failed', 'skipped' counts and details.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply filters
    df = catalog_df.copy()
    if filters:
        if "proofread_only" in filters and filters["proofread_only"]:
            df = df[df["is_proofread"]]

        if "cell_type_broad" in filters:
            types = filters["cell_type_broad"]
            if isinstance(types, str):
                types = [types]
            df = df[df["cell_type_broad"].isin(types)]

        if "cell_type_fine" in filters:
            types = filters["cell_type_fine"]
            if isinstance(types, str):
                types = [types]
            df = df[df["cell_type_fine"].isin(types)]

        if "root_ids" in filters:
            df = df[df["root_id"].isin(filters["root_ids"])]

    if max_neurons is not None:
        df = df.head(max_neurons)

    logger.info(f"Batch: {len(df)} neurons to process")

    # Load progress for resume
    progress = _load_progress(output_dir) if resume else {
        "completed": [], "failed": [], "start_time": None
    }
    if progress["start_time"] is None:
        progress["start_time"] = datetime.now().isoformat()

    completed_ids = set(progress["completed"])
    failed_ids = set(progress["failed"])

    results = {
        "completed": [],
        "failed": [],
        "skipped": [],
        "qc_results": [],
    }

    timing_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching neurons"):
        root_id = int(row["root_id"])
        cell_type_broad = row["cell_type_broad"]
        cell_type_fine = row["cell_type_fine"]
        label = make_label(cell_type_broad, cell_type_fine, root_id)

        # Skip if already done
        if resume and root_id in completed_ids:
            results["skipped"].append(root_id)
            continue

        # Skip known failures (don't retry stale roots)
        if resume and root_id in failed_ids:
            results["skipped"].append(root_id)
            continue

        t_start = time.time()

        try:
            # Fetch
            fetch_result = fetch_neuron(client, root_id)
            if fetch_result is None:
                _log_error(output_dir, root_id, label, "fetch_failed",
                           "fetch_neuron returned None")
                progress["failed"].append(root_id)
                results["failed"].append(root_id)
                _save_progress(output_dir, progress)
                continue

            # Check partner coverage
            if fetch_result["partner_coverage"] < 0.50:
                logger.warning(
                    f"{label}: low partner coverage "
                    f"({fetch_result['partner_coverage']:.0%})"
                )

            # Save files
            save_neuron_files(label, root_id, fetch_result, output_dir)

            # Validate
            if validate:
                qc = validate_neuron(label, root_id, output_dir)
                results["qc_results"].append(qc)
                if not qc["passed"]:
                    logger.warning(
                        f"{label}: QC failed: {qc['failure_reasons']}"
                    )

            # Record success
            progress["completed"].append(root_id)
            results["completed"].append(root_id)
            _save_progress(output_dir, progress)

            # Timing
            elapsed = time.time() - t_start
            timing_records.append({
                "root_id": root_id,
                "label": label,
                "elapsed_s": elapsed,
                "n_synapses": fetch_result["n_synapses"],
                "n_partners": fetch_result["n_unique_partners"],
            })
            logger.info(f"  {label}: done in {elapsed:.1f}s")

        except Exception as e:
            _log_error(output_dir, root_id, label, "exception", str(e))
            progress["failed"].append(root_id)
            results["failed"].append(root_id)
            _save_progress(output_dir, progress)
            logger.error(f"  {label}: FAILED: {e}")

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

    # Save timing data
    if timing_records:
        timing_df = pd.DataFrame(timing_records)
        timing_path = output_dir / "throughput.json"
        timing_summary = {
            "n_neurons": len(timing_records),
            "mean_time_per_neuron_s": float(timing_df["elapsed_s"].mean()),
            "median_time_per_neuron_s": float(timing_df["elapsed_s"].median()),
            "total_time_s": float(timing_df["elapsed_s"].sum()),
            "mean_synapses_per_neuron": float(timing_df["n_synapses"].mean()),
            "extrapolated_2200_neurons_hours": float(
                timing_df["elapsed_s"].mean() * 2200 / 3600
            ),
            "per_neuron": timing_records,
        }
        with open(timing_path, "w") as f:
            json.dump(timing_summary, f, indent=2)

    # Summary
    logger.info(
        f"\nBatch complete: {len(results['completed'])} completed, "
        f"{len(results['failed'])} failed, "
        f"{len(results['skipped'])} skipped"
    )

    return results
