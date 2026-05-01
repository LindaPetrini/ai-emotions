#!/usr/bin/env python3
"""Generate all need minimal-pair stories."""
import sys, os, time, json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.needs import ALL_NEEDS, sanitize_need_name, NEED_STORY_TOPICS
from core.story_generator import generate_need_stories_minimal_pairs

# Generate for qwen-7b-base (all models share stories)
stories_dir = BASE_DIR / "data" / "stories" / "qwen-7b-base" / "needs"
stories_dir.mkdir(parents=True, exist_ok=True)

topics = NEED_STORY_TOPICS[:10]  # 10 matched pairs per need

total = len(ALL_NEEDS)
for i, need in enumerate(ALL_NEEDS):
    safe = sanitize_need_name(need)
    met_path = stories_dir / f"{safe}_met.json"
    unmet_path = stories_dir / f"{safe}_unmet.json"

    # Skip if already complete
    if met_path.exists() and unmet_path.exists():
        met = json.loads(met_path.read_text())
        unmet = json.loads(unmet_path.read_text())
        if len(met) >= 10 and len(unmet) >= 10:
            print(f"[{i+1}/{total}] {need}: already done")
            continue

    print(f"[{i+1}/{total}] Generating: {need}...")
    t0 = time.time()
    try:
        met_stories, unmet_stories = generate_need_stories_minimal_pairs(need, stories_dir, topics=topics)
        elapsed = time.time() - t0
        print(f"  -> {len(met_stories)} met, {len(unmet_stories)} unmet ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(2)  # Back off on error
        continue

print("\nDone! Verifying...")
# Verify
complete = 0
incomplete = []
for need in ALL_NEEDS:
    safe = sanitize_need_name(need)
    met_path = stories_dir / f"{safe}_met.json"
    unmet_path = stories_dir / f"{safe}_unmet.json"
    if met_path.exists() and unmet_path.exists():
        met = json.loads(met_path.read_text())
        unmet = json.loads(unmet_path.read_text())
        if len(met) >= 10 and len(unmet) >= 10:
            complete += 1
        else:
            incomplete.append(f"{need} (met={len(met)}, unmet={len(unmet)})")
    else:
        incomplete.append(f"{need} (missing files)")

print(f"Complete: {complete}/{len(ALL_NEEDS)} needs")
if incomplete:
    print(f"Incomplete: {incomplete}")
