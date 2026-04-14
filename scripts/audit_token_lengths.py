"""Audit token lengths for all stories across all models.

Checks how many tokens each story produces for each model's tokenizer,
and flags stories where usable tokens (n_tokens - MIN_TOKEN_POS) are
dangerously low.

Usage:
    python -m scripts.audit_token_lengths
"""

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.models import MODEL_REGISTRY, BASE_DIR
from configs.emotions import ALL_EMOTIONS
from configs.needs import ALL_NEEDS, sanitize_need_name

# Must match core/activations.py MIN_TOKEN_POS
MIN_TOKEN_POS = 50


def load_tokenizer(model_id: str):
    """Load a HuggingFace tokenizer by model ID."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def count_tokens(tokenizer, text: str) -> int:
    """Return number of tokens for a text string."""
    return len(tokenizer(text, truncation=True, max_length=512)["input_ids"])


def compute_stats(lengths: list[int]) -> dict:
    """Compute distribution statistics for a list of token lengths."""
    if not lengths:
        return {}
    arr = np.array(lengths)
    usable = arr - MIN_TOKEN_POS
    return {
        "count": len(arr),
        "n_tokens": {
            "min": int(np.min(arr)),
            "p5": int(np.percentile(arr, 5)),
            "p25": int(np.percentile(arr, 25)),
            "median": int(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p95": int(np.percentile(arr, 95)),
            "max": int(np.max(arr)),
        },
        "usable_tokens": {
            "min": int(np.min(usable)),
            "p5": int(np.percentile(usable, 5)),
            "p25": int(np.percentile(usable, 25)),
            "median": int(np.median(usable)),
            "mean": float(np.mean(usable)),
            "p95": int(np.percentile(usable, 95)),
            "max": int(np.max(usable)),
        },
        "flagged_lt20_usable": int(np.sum(usable < 20)),
        "flagged_lt0_usable": int(np.sum(usable < 0)),
    }


def print_stats_table(label: str, stats: dict):
    """Print a formatted stats row."""
    if not stats:
        print(f"  {label}: NO DATA")
        return
    nt = stats["n_tokens"]
    ut = stats["usable_tokens"]
    flag_str = ""
    if stats["flagged_lt20_usable"] > 0:
        flag_str = f"  *** {stats['flagged_lt20_usable']}/{stats['count']} stories have <20 usable tokens ***"
    if stats["flagged_lt0_usable"] > 0:
        flag_str += f"  *** {stats['flagged_lt0_usable']}/{stats['count']} stories have <0 usable tokens (MIN_TOKEN_POS not applied) ***"
    print(f"  {label} ({stats['count']} stories):")
    print(f"    Tokens:  min={nt['min']:>4}  p5={nt['p5']:>4}  p25={nt['p25']:>4}  "
          f"median={nt['median']:>4}  mean={nt['mean']:>6.1f}  p95={nt['p95']:>4}  max={nt['max']:>4}")
    print(f"    Usable:  min={ut['min']:>4}  p5={ut['p5']:>4}  p25={ut['p25']:>4}  "
          f"median={ut['median']:>4}  mean={ut['mean']:>6.1f}  p95={ut['p95']:>4}  max={ut['max']:>4}")
    if flag_str:
        print(f"    {flag_str}")


def audit_model(model_name: str, model_id: str) -> dict:
    """Run the full token length audit for one model."""
    print(f"\n{'='*80}")
    print(f"Model: {model_name} ({model_id})")
    print(f"MIN_TOKEN_POS = {MIN_TOKEN_POS}")
    print(f"{'='*80}")

    tokenizer = load_tokenizer(model_id)
    stories_base = BASE_DIR / "data" / "stories" / model_name

    result = {"model_name": model_name, "model_id": model_id, "emotions": {}, "needs": {}}
    all_emotion_lengths = []
    all_need_lengths = []
    flagged_stories = []

    # --- Emotion stories ---
    print(f"\n--- Emotion stories ---")
    for emotion in ALL_EMOTIONS:
        path = stories_base / f"{emotion}.json"
        if not path.exists():
            continue
        stories = json.loads(path.read_text())
        lengths = [count_tokens(tokenizer, s) for s in stories]
        stats = compute_stats(lengths)
        result["emotions"][emotion] = stats
        all_emotion_lengths.extend(lengths)

        # Flag individual stories
        for i, (length, story) in enumerate(zip(lengths, stories)):
            usable = length - MIN_TOKEN_POS
            if usable < 20:
                flagged_stories.append({
                    "type": "emotion",
                    "label": emotion,
                    "index": i,
                    "n_tokens": length,
                    "usable_tokens": usable,
                    "story_preview": story[:100],
                })

    emotion_overall = compute_stats(all_emotion_lengths)
    result["emotions_overall"] = emotion_overall
    print_stats_table("ALL EMOTIONS", emotion_overall)

    # Show per-emotion stats only for worst cases
    worst_emotions = sorted(
        [(e, s) for e, s in result["emotions"].items() if s],
        key=lambda x: x[1]["usable_tokens"]["min"],
    )
    if worst_emotions:
        print(f"\n  Bottom 10 emotions by min usable tokens:")
        for emotion, stats in worst_emotions[:10]:
            ut = stats["usable_tokens"]
            print(f"    {emotion:>25s}: min_usable={ut['min']:>4}  median_usable={ut['median']:>4}  "
                  f"flagged={stats['flagged_lt20_usable']}/{stats['count']}")

    # --- Need stories ---
    needs_dir = stories_base / "needs"
    print(f"\n--- Need stories ---")
    for need in ALL_NEEDS:
        safe = sanitize_need_name(need)
        for condition in ["met", "unmet"]:
            path = needs_dir / f"{safe}_{condition}.json"
            if not path.exists():
                continue
            stories = json.loads(path.read_text())
            lengths = [count_tokens(tokenizer, s) for s in stories]
            key = f"{safe}_{condition}"
            stats = compute_stats(lengths)
            result["needs"][key] = stats
            all_need_lengths.extend(lengths)

            for i, (length, story) in enumerate(zip(lengths, stories)):
                usable = length - MIN_TOKEN_POS
                if usable < 20:
                    flagged_stories.append({
                        "type": "need",
                        "label": f"{need} ({condition})",
                        "index": i,
                        "n_tokens": length,
                        "usable_tokens": usable,
                        "story_preview": story[:100],
                    })

    need_overall = compute_stats(all_need_lengths)
    result["needs_overall"] = need_overall
    print_stats_table("ALL NEEDS", need_overall)

    # Show per-need stats only for worst cases
    worst_needs = sorted(
        [(n, s) for n, s in result["needs"].items() if s],
        key=lambda x: x[1]["usable_tokens"]["min"],
    )
    if worst_needs:
        print(f"\n  Bottom 10 needs by min usable tokens:")
        for need_key, stats in worst_needs[:10]:
            ut = stats["usable_tokens"]
            print(f"    {need_key:>35s}: min_usable={ut['min']:>4}  median_usable={ut['median']:>4}  "
                  f"flagged={stats['flagged_lt20_usable']}/{stats['count']}")

    # --- Flagged stories ---
    result["flagged_stories"] = flagged_stories
    if flagged_stories:
        print(f"\n--- FLAGGED: {len(flagged_stories)} stories with <20 usable tokens ---")
        for f in flagged_stories[:20]:
            print(f"  [{f['type']}] {f['label']} #{f['index']}: "
                  f"{f['n_tokens']} tokens, {f['usable_tokens']} usable  "
                  f"\"{f['story_preview']}...\"")
        if len(flagged_stories) > 20:
            print(f"  ... and {len(flagged_stories) - 20} more")
    else:
        print(f"\n  No stories flagged (all have >= 20 usable tokens)")

    return result


def main():
    all_results = {}

    for model_name, cfg in MODEL_REGISTRY.items():
        try:
            result = audit_model(model_name, cfg.model_id)
            all_results[model_name] = result
        except Exception as e:
            print(f"\nERROR auditing {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}

    # --- Cross-model summary ---
    print(f"\n{'='*80}")
    print("CROSS-MODEL SUMMARY")
    print(f"MIN_TOKEN_POS = {MIN_TOKEN_POS}")
    print(f"{'='*80}")
    print(f"\n{'Model':>20s} | {'Type':>8s} | {'Count':>6s} | {'Min':>4s} | {'P5':>4s} | "
          f"{'Med':>4s} | {'Mean':>6s} | {'P95':>4s} | {'Max':>4s} | {'<20':>4s}")
    print("-" * 95)
    for model_name, result in all_results.items():
        if "error" in result:
            print(f"{model_name:>20s} | ERROR: {result['error']}")
            continue
        for typ, key in [("emotions", "emotions_overall"), ("needs", "needs_overall")]:
            stats = result.get(key, {})
            if not stats:
                continue
            ut = stats["usable_tokens"]
            print(f"{model_name:>20s} | {typ:>8s} | {stats['count']:>6d} | "
                  f"{ut['min']:>4d} | {ut['p5']:>4d} | {ut['median']:>4d} | "
                  f"{ut['mean']:>6.1f} | {ut['p95']:>4d} | {ut['max']:>4d} | "
                  f"{stats['flagged_lt20_usable']:>4d}")

    # Total flagged
    total_flagged = sum(
        len(r.get("flagged_stories", []))
        for r in all_results.values()
        if "error" not in r
    )
    print(f"\nTotal stories with <20 usable tokens across all models: {total_flagged}")

    # --- Save results ---
    output_path = BASE_DIR / "data" / "figures" / "token_length_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert for JSON serialization (strip story previews to keep file small)
    save_data = {}
    for model_name, result in all_results.items():
        if "error" in result:
            save_data[model_name] = result
            continue
        save_data[model_name] = {
            "model_id": result["model_id"],
            "min_token_pos": MIN_TOKEN_POS,
            "emotions_overall": result.get("emotions_overall", {}),
            "needs_overall": result.get("needs_overall", {}),
            "emotions": result.get("emotions", {}),
            "needs": result.get("needs", {}),
            "n_flagged_stories": len(result.get("flagged_stories", [])),
            "flagged_stories": result.get("flagged_stories", []),
        }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
