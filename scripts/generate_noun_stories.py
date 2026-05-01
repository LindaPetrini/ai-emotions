#!/usr/bin/env python3
"""Generate noun control stories."""
import sys, os, time, json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.story_generator import CONTROL_NOUNS, generate_noun_control_stories
from configs.emotions import STORY_TOPICS

stories_dir = BASE_DIR / "data" / "stories" / "qwen-7b-base" / "nouns"
stories_dir.mkdir(parents=True, exist_ok=True)

total = len(CONTROL_NOUNS)
print(f"Generating stories for {total} nouns x {len(STORY_TOPICS)} topics = {total * len(STORY_TOPICS)} stories")

for i, noun in enumerate(CONTROL_NOUNS):
    output_path = stories_dir / f"{noun}.json"
    existing = json.loads(output_path.read_text()) if output_path.exists() else []

    if len(existing) >= len(STORY_TOPICS):
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{total}] {noun}: already done ({len(existing)} stories)")
        continue

    print(f"[{i+1}/{total}] Generating: {noun} ({len(existing)}/{len(STORY_TOPICS)} done)...")
    t0 = time.time()
    try:
        generate_noun_control_stories([noun], stories_dir, topics=STORY_TOPICS)
        elapsed = time.time() - t0
        new_count = len(json.loads(output_path.read_text()))
        print(f"  -> {new_count} stories ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  ERROR: {e}")
        time.sleep(2)
        continue

print("\nDone! Verifying...")
complete = 0
for noun in CONTROL_NOUNS:
    path = stories_dir / f"{noun}.json"
    if path.exists():
        stories = json.loads(path.read_text())
        if len(stories) >= 20:
            complete += 1
print(f"Complete: {complete}/{total} nouns")
