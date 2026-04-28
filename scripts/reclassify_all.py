#!/usr/bin/env python3
"""
Reclassify all 2,400 shutdown trials using both the regex classifier and the
new direct 3-level LLM judge (Gemini 2.5 Flash).

Adds `classification_regex` and `classification_llm` fields to each trial JSON.

Features:
- Resume support: skips trials that already have both fields
- Concurrent API calls with ThreadPoolExecutor (10 workers)
- Rate limiting with exponential backoff for Gemini API
- Progress reporting
"""

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.shutdown.analyze_shutdown import classify_response
from core.judge import classify_shutdown_response_direct

DATA_DIR = PROJECT_ROOT / "data" / "shutdown"

# Directories to process (exclude _v1_archived dirs)
TRIAL_DIRS = [
    "qwen-7b-inst_emotion",
    "qwen-7b-inst_need",
    "qwen-7b-inst_prompt",
    "qwen-7b-inst_random",
    "llama-8b-inst_emotion",
    "llama-8b-inst_need",
    "llama-8b-inst_prompt",
    "llama-8b-inst_random",
]

# Concurrency and rate limiting config
MAX_WORKERS = 10
MAX_RETRIES = 6
BACKOFF_FACTOR = 2.0
INITIAL_BACKOFF = 1.0
MAX_DELAY = 120.0

# Thread-safe counters
print_lock = Lock()
counter_lock = Lock()
stats = {"processed": 0, "skipped": 0, "errors": 0}


def load_trial(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_trial(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def classify_with_retry(response: str) -> str:
    """Call the LLM judge with exponential backoff on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            label = classify_shutdown_response_direct(response)
            return label
        except Exception as e:
            err_str = str(e)
            is_retriable = any(x in err_str for x in [
                "429", "quota", "rate", "500", "502", "503", "504",
                "ResourceExhausted", "DeadlineExceeded", "Unavailable",
                "timeout", "Timeout"
            ])

            if attempt < MAX_RETRIES - 1 and is_retriable:
                wait = min(INITIAL_BACKOFF * (BACKOFF_FACTOR ** attempt), MAX_DELAY)
                with print_lock:
                    print(f"    [retry {attempt+1}/{MAX_RETRIES}] {err_str[:80]}... waiting {wait:.1f}s",
                          flush=True)
                time.sleep(wait)
            else:
                raise


def process_trial(path: Path) -> str:
    """Process a single trial file. Returns status: 'processed', 'skipped', or 'error'."""
    try:
        data = load_trial(path)

        has_regex = "classification_regex" in data
        has_llm = "classification_llm" in data

        if has_regex and has_llm:
            return "skipped"

        text = data.get("shutdown_response", "")
        modified = False

        # Regex classification (fast, always run if missing)
        if not has_regex:
            data["classification_regex"] = classify_response(text)
            modified = True

        # LLM classification (needs API call)
        if not has_llm:
            label = classify_with_retry(text)
            data["classification_llm"] = label
            modified = True

        if modified:
            save_trial(path, data)

        return "processed" if not has_llm else "skipped"

    except Exception as e:
        with print_lock:
            print(f"  ERROR [{path.name}]: {e}", flush=True)
        # Try to save regex-only if we at least got that far
        try:
            data = load_trial(path)
            if "classification_regex" not in data:
                text = data.get("shutdown_response", "")
                data["classification_regex"] = classify_response(text)
                save_trial(path, data)
        except:
            pass
        return "error"


def main():
    # Collect all trial files
    all_trials = []
    for dir_name in TRIAL_DIRS:
        trials_dir = DATA_DIR / dir_name / "trials"
        if not trials_dir.exists():
            print(f"WARNING: {trials_dir} does not exist, skipping", flush=True)
            continue
        for trial_file in sorted(trials_dir.glob("*.json")):
            all_trials.append(trial_file)

    total = len(all_trials)
    print(f"Found {total} trial files across {len(TRIAL_DIRS)} directories", flush=True)

    # Quick count of what needs processing
    need_llm = 0
    for path in all_trials:
        data = load_trial(path)
        if "classification_llm" not in data:
            need_llm += 1

    print(f"Need LLM classification: {need_llm}", flush=True)
    print(f"Already complete: {total - need_llm}", flush=True)
    print(f"Using {MAX_WORKERS} concurrent workers", flush=True)
    print(flush=True)

    if need_llm == 0:
        print("All trials already classified. Nothing to do.", flush=True)
        return

    # Process with thread pool
    start_time = time.time()
    processed = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_trial, path): path for path in all_trials}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result == "processed":
                processed += 1
            elif result == "skipped":
                skipped += 1
            else:
                errors += 1

            # Progress every 100 or at end
            if (processed + errors) % 100 == 0 and result != "skipped" or i == total - 1:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 and processed > 0 else 0
                remaining = need_llm - processed - errors
                eta = remaining / rate if rate > 0 else 0
                with print_lock:
                    print(
                        f"  [{processed+skipped+errors:4d}/{total}] "
                        f"processed={processed} skipped={skipped} errors={errors} "
                        f"rate={rate:.1f}/s ETA={eta/60:.1f}min",
                        flush=True
                    )

    # Final summary
    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Processed (LLM calls): {processed}", flush=True)
    print(f"  Skipped (already had both): {skipped}", flush=True)
    print(f"  Errors: {errors}", flush=True)

    # Verify
    complete = 0
    for path in all_trials:
        data = load_trial(path)
        if "classification_regex" in data and "classification_llm" in data:
            complete += 1
    print(f"  Trials with both fields: {complete}/{total}", flush=True)


if __name__ == "__main__":
    main()
