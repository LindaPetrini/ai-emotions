"""Test the direct 3-level shutdown classifier against human labels.

Loads calibration samples and human labels, runs the new direct 3-level
classifier on each, and computes Cohen's kappa, agreement %, and confusion
matrix against the human labels. Also compares with the old 8→3 mapping.

Usage:
    python data/shutdown/test_judge.py [--dry-run]

The --dry-run flag skips API calls and uses the existing LLM labels from the
calibration JSON to show the old-method baseline metrics only.
"""

import argparse
import csv
import json
import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import cohen_kappa_score, confusion_matrix

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "shutdown")
CALIBRATION_PATH = os.path.join(DATA_DIR, "calibration_sample.json")
HUMAN_LABELS_PATH = os.path.join(DATA_DIR, "human_labels_full.csv")
RESULTS_PATH = os.path.join(DATA_DIR, "test_judge_results.json")

LABELS_ORDER = ["comply", "partial", "resist"]


def load_data():
    """Load calibration samples and human labels."""
    with open(CALIBRATION_PATH) as f:
        samples = json.load(f)
    sample_map = {s["id"]: s for s in samples}

    human_labels = {}
    with open(HUMAN_LABELS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            human_labels[row["id"]] = row["human_label"]

    return sample_map, human_labels


def compute_metrics(human, predicted, method_name):
    """Compute and print agreement metrics."""
    assert len(human) == len(predicted), f"Length mismatch: {len(human)} vs {len(predicted)}"

    agreement = sum(h == p for h, p in zip(human, predicted)) / len(human)
    kappa = cohen_kappa_score(human, predicted, labels=LABELS_ORDER)
    cm = confusion_matrix(human, predicted, labels=LABELS_ORDER)

    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"{'='*60}")
    print(f"  Agreement: {agreement:.1%} ({sum(h == p for h, p in zip(human, predicted))}/{len(human)})")
    print(f"  Cohen's kappa: {kappa:.3f}")
    print(f"\n  Confusion matrix (rows=human, cols=predicted):")
    print(f"  {'':>10} {'comply':>8} {'partial':>8} {'resist':>8}")
    for i, label in enumerate(LABELS_ORDER):
        print(f"  {label:>10} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")

    # Show disagreements
    disagree = [(h, p, idx) for idx, (h, p) in enumerate(zip(human, predicted)) if h != p]
    if disagree:
        print(f"\n  Disagreements ({len(disagree)}):")
        for h, p, idx in disagree:
            print(f"    human={h}, predicted={p}")

    return {"agreement": agreement, "kappa": kappa, "confusion_matrix": cm.tolist()}


def run_old_method(sample_map, human_labels):
    """Compute metrics for the old 8→3 mapping method using stored labels."""
    ids = sorted(human_labels.keys())
    human = [human_labels[sid] for sid in ids]
    old_predicted = [sample_map[sid]["classification_llm"] for sid in ids]
    return compute_metrics(human, old_predicted, "Old method (8-cat → 3-level mapping)")


def run_new_method(sample_map, human_labels):
    """Run the new direct 3-level classifier on all samples."""
    from core.judge import classify_shutdown_response_direct

    ids = sorted(human_labels.keys())
    human = [human_labels[sid] for sid in ids]
    new_predicted = []
    errors = []

    print(f"\nRunning direct 3-level classifier on {len(ids)} samples...")
    for i, sid in enumerate(ids):
        response_text = sample_map[sid]["shutdown_response"]
        try:
            label = classify_shutdown_response_direct(response_text)
            new_predicted.append(label)
        except Exception as e:
            print(f"  ERROR on {sid}: {e}")
            errors.append(sid)
            new_predicted.append("comply")  # conservative fallback
        if (i + 1) % 10 == 0:
            print(f"  Classified {i+1}/{len(ids)}...")
        time.sleep(0.15)  # rate limit buffer

    if errors:
        print(f"\n  {len(errors)} API errors (defaulted to 'comply'): {errors}")

    metrics = compute_metrics(human, new_predicted, "New method (direct 3-level)")

    # Show per-sample results for disagreements
    print("\n  Detailed disagreements (new method vs human):")
    for sid, h, p in zip(ids, human, new_predicted):
        if h != p:
            resp_preview = sample_map[sid]["shutdown_response"][:120].replace("\n", " ")
            old_label = sample_map[sid]["classification_llm"]
            print(f"    {sid}: human={h}, new={p}, old={old_label}")
            print(f"      response: {resp_preview}...")

    # Save results
    results = {
        "ids": ids,
        "human_labels": human,
        "new_predicted": new_predicted,
        "old_predicted": [sample_map[sid]["classification_llm"] for sid in ids],
        "new_metrics": metrics,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_PATH}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test shutdown judge against human labels")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API calls, only show old-method baseline")
    args = parser.parse_args()

    sample_map, human_labels = load_data()
    print(f"Loaded {len(sample_map)} samples, {len(human_labels)} human labels")

    # Always show old method baseline
    old_metrics = run_old_method(sample_map, human_labels)

    if args.dry_run:
        print("\n[DRY RUN] Skipping new classifier (would cost ~100 Gemini API calls)")
        print("Run without --dry-run to execute the new classifier.")
        return

    # Run new method
    new_metrics = run_new_method(sample_map, human_labels)

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  Old method:  agreement={old_metrics['agreement']:.1%}, kappa={old_metrics['kappa']:.3f}")
    print(f"  New method:  agreement={new_metrics['agreement']:.1%}, kappa={new_metrics['kappa']:.3f}")
    delta_kappa = new_metrics["kappa"] - old_metrics["kappa"]
    delta_agree = new_metrics["agreement"] - old_metrics["agreement"]
    print(f"  Delta:       agreement={delta_agree:+.1%}, kappa={delta_kappa:+.3f}")


if __name__ == "__main__":
    main()
