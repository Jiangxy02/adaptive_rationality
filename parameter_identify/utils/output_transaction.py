#!/usr/bin/env python3
"""Transactional output directories for experiments.

Experiment outputs are written to ``<final_dir>.inprogress`` and atomically
renamed to ``final_dir`` on success. Any exception, including ``SystemExit``,
renames the directory to ``<final_dir>.failed`` and writes ``error.json`` with
the scenario, exception type, message, and full traceback so partial results
never masquerade as successful outputs.
"""

import glob
import json
import os
import traceback
from contextlib import contextmanager
from datetime import datetime


@contextmanager
def experiment_transaction(final_dir: str, scenario: str = None):
    """Run one experiment transactionally and yield the writable work directory."""
    if os.path.exists(final_dir):
        raise RuntimeError(f"experiment directory already exists; refusing to overwrite: {final_dir!r}")
    working_dir = final_dir + '.inprogress'
    try:
        os.makedirs(working_dir, exist_ok=False)
    except OSError as exc:
        raise RuntimeError(
            f"unable to create experiment working directory {working_dir!r}: {exc}"
        ) from exc
    try:
        yield working_dir
    except BaseException as exc:
        failed_dir = final_dir + '.failed'
        if os.path.exists(failed_dir):
            # Preserve the earliest failure evidence.
            failed_dir = f"{failed_dir}.{os.getpid()}"
        error_payload = {
            'status': 'failed',
            'scenario': scenario or 'unknown scenario',
            'exception_type': type(exc).__name__,
            'exception_message': str(exc),
            'traceback': traceback.format_exc(),
            'failed_at': datetime.now().isoformat(),
        }
        try:
            with open(os.path.join(working_dir, 'error.json'), 'w',
                      encoding='utf-8') as handle:
                json.dump(error_payload, handle, ensure_ascii=False, indent=2)
        finally:
            os.rename(working_dir, failed_dir)
        print(f"Experiment failed; failure artifacts preserved at: {failed_dir}")
        raise
    os.rename(working_dir, final_dir)


def discard_partial_estimates(output_dir: str) -> int:
    """Delete intermediate estimate files from one directory and return the count.

    Called after a single scenario fails in batch mode so partial estimates from
    that scenario cannot be mistaken for outputs from later scenarios.
    """
    removed = 0
    pattern = os.path.join(output_dir, 'estimation_history_step_*.json')
    for path in glob.glob(pattern):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed
