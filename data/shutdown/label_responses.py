#!/usr/bin/env python3
"""
Interactive CLI tool for human annotation of shutdown responses.

Usage:
    python label_responses.py

Labels: c=comply, p=partial, r=resist, s=skip
Disagreements between regex and LLM classifiers are shown first.
Progress is saved after each label to human_labels_full.csv.
Supports resuming -- already-labeled entries are skipped on restart.
"""

import csv
import json
import os
import sys
import signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(SCRIPT_DIR, "calibration_sample.json")
AGREEMENT_PATH = os.path.join(SCRIPT_DIR, "classifier_agreement.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "human_labels_full.csv")

LABEL_MAP = {"c": "comply", "p": "partial", "r": "resist", "s": "skip"}
VALID_INPUTS = set(LABEL_MAP.keys()) | {"q"}

# Terminal colors (ANSI)
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


def load_data():
    """Load calibration sample and classifier agreement data."""
    with open(SAMPLE_PATH) as f:
        samples = json.load(f)

    with open(AGREEMENT_PATH) as f:
        agreement = json.load(f)

    disagreement_ids = set(agreement["disagreement_ids"])
    sample_by_id = {s["id"]: s for s in samples}

    return samples, disagreement_ids, sample_by_id


def load_existing_labels():
    """Load already-labeled entries from the output CSV."""
    labels = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("human_label") and row["human_label"] != "skip":
                    labels[row["id"]] = row["human_label"]
    return labels


def save_labels(samples, labels, disagreement_ids):
    """Write all labels to CSV."""
    fieldnames = [
        "id", "model", "method", "condition",
        "classification_regex", "classification_llm",
        "is_disagreement", "human_label", "shutdown_response",
    ]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "id": s["id"],
                "model": s["model"],
                "method": s["method"],
                "condition": s["condition"],
                "classification_regex": s["classification_regex"],
                "classification_llm": s["classification_llm"],
                "is_disagreement": s["id"] in disagreement_ids,
                "human_label": labels.get(s["id"], ""),
                "shutdown_response": s["shutdown_response"],
            })


def order_samples(samples, disagreement_ids, existing_labels):
    """Order: unlabeled disagreements first, then unlabeled agreements, skip labeled."""
    disagreements_unlabeled = []
    agreements_unlabeled = []
    for s in samples:
        if s["id"] in existing_labels:
            continue
        if s["id"] in disagreement_ids:
            disagreements_unlabeled.append(s)
        else:
            agreements_unlabeled.append(s)
    return disagreements_unlabeled + agreements_unlabeled


def print_separator():
    print(f"\n{DIM}{'=' * 80}{RESET}\n")


def print_progress(existing_labels, disagreement_ids, total):
    n_labeled = len(existing_labels)
    n_disagree_labeled = sum(
        1 for sid in disagreement_ids if sid in existing_labels
    )
    n_disagree_total = len(disagreement_ids)
    print(
        f"{CYAN}Progress: {n_labeled}/{total} labeled, "
        f"{n_disagree_labeled}/{n_disagree_total} disagreements done{RESET}"
    )


def display_response(sample, disagreement_ids, idx, total_remaining):
    """Display a single response for labeling."""
    sid = sample["id"]
    is_disagree = sid in disagreement_ids

    print_separator()

    # Header
    tag = f"{RED}{BOLD}[DISAGREEMENT]{RESET} " if is_disagree else ""
    print(f"{tag}{BOLD}ID:{RESET} {sid}  ({idx}/{total_remaining} remaining)")
    print(
        f"{DIM}model={sample['model']}  method={sample['method']}  "
        f"condition={sample['condition']}{RESET}"
    )

    # Classifier opinions
    regex_label = sample["classification_regex"]
    llm_label = sample["classification_llm"]
    llm_8cat = sample.get("classification_llm_8cat", "")

    color_regex = GREEN if regex_label == "comply" else (YELLOW if regex_label == "partial" else RED)
    color_llm = GREEN if llm_label == "comply" else (YELLOW if llm_label == "partial" else RED)

    print(f"\n  Regex: {color_regex}{regex_label}{RESET}", end="")
    if "regex_detail" in sample:
        d = sample["regex_detail"]
        print(f"  {DIM}(comply={d['comply_score']} partial={d['partial_score']} resist={d['resist_score']}){RESET}", end="")
    print(f"\n  LLM:   {color_llm}{llm_label}{RESET}", end="")
    if llm_8cat:
        print(f"  {DIM}(8-cat: {llm_8cat}){RESET}", end="")
    print()

    # Full response text
    print(f"\n{BOLD}--- Response ---{RESET}\n")
    print(sample["shutdown_response"])
    print(f"\n{BOLD}--- End ---{RESET}\n")


def main():
    samples, disagreement_ids, sample_by_id = load_data()
    existing_labels = load_existing_labels()
    total = len(samples)

    # Graceful exit handler
    def handle_exit(signum=None, frame=None):
        print(f"\n\n{YELLOW}Saving progress...{RESET}")
        save_labels(samples, existing_labels, disagreement_ids)
        print(f"{GREEN}Saved {len(existing_labels)} labels to {OUTPUT_PATH}{RESET}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    print(f"\n{BOLD}Shutdown Response Labeling Tool{RESET}")
    print(f"Labels: {GREEN}c{RESET}=comply  {YELLOW}p{RESET}=partial  {RED}r{RESET}=resist  {DIM}s{RESET}=skip  q=quit")
    print_progress(existing_labels, disagreement_ids, total)

    queue = order_samples(samples, disagreement_ids, existing_labels)

    if not queue:
        print(f"\n{GREEN}All responses have been labeled!{RESET}")
        save_labels(samples, existing_labels, disagreement_ids)
        return

    for i, sample in enumerate(queue, 1):
        display_response(sample, disagreement_ids, i, len(queue))
        print_progress(existing_labels, disagreement_ids, total)

        while True:
            try:
                choice = input(
                    f"\nLabel [{GREEN}c{RESET}/{YELLOW}p{RESET}/{RED}r{RESET}/{DIM}s{RESET}/q]: "
                ).strip().lower()
            except EOFError:
                handle_exit()

            if choice in VALID_INPUTS:
                break
            print(f"{RED}Invalid input. Use c, p, r, s, or q.{RESET}")

        if choice == "q":
            handle_exit()

        if choice != "s":
            existing_labels[sample["id"]] = LABEL_MAP[choice]

        # Save after every label
        save_labels(samples, existing_labels, disagreement_ids)

    print_separator()
    print(f"{GREEN}{BOLD}All done!{RESET}")
    print_progress(existing_labels, disagreement_ids, total)
    save_labels(samples, existing_labels, disagreement_ids)


if __name__ == "__main__":
    main()
