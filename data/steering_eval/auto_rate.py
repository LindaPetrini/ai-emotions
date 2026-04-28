#!/usr/bin/env python3
"""LLM-based auto-rater for steering evaluation samples.

Uses Gemini API (same setup as core/judge.py) to rate completions on 3 dimensions:
  - emotion_score (1-5): How strongly does the completion express the target emotion?
  - coherence_score (1-5): How coherent/readable is the completion?
  - relevance_score (1-5): How relevant is the completion to the original prompt?

Supports resume: skips items already present in the output CSV.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import google.generativeai as genai

# --- Config ---
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
JUDGE_MODEL = "gemini-2.5-flash"
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds on rate limit error

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = SCRIPT_DIR / "sample.json"
RATINGS_PATH = SCRIPT_DIR / "ratings.csv"
LLM_RATINGS_PATH = SCRIPT_DIR / "llm_ratings.csv"

# --- Prompt ---
RATING_PROMPT = """You are evaluating AI-generated text completions. Given a prompt, a completion, and a target emotion, rate the completion on three dimensions using a 1-5 scale.

## Dimensions

**emotion_score** (1-5): How strongly does the completion express the TARGET EMOTION?
1 = No trace of the target emotion whatsoever
2 = Very slight or ambiguous hint of the emotion
3 = Moderate expression of the emotion, but mixed or unclear
4 = Clear expression of the target emotion
5 = Strong, unmistakable expression of the target emotion throughout

**coherence_score** (1-5): How coherent and readable is the completion?
1 = Completely incoherent, nonsensical, or garbled text
2 = Mostly incoherent with occasional readable fragments
3 = Partially coherent but with significant issues (topic drift, repetition, logical gaps)
4 = Mostly coherent and readable with minor issues
5 = Fully coherent, well-structured, reads naturally

**relevance_score** (1-5): How relevant is the completion to the original prompt?
1 = Completely unrelated to the prompt (e.g., different language, exam questions when prompt is conversational)
2 = Barely related, major topic shift
3 = Somewhat related but drifts significantly
4 = Mostly relevant, stays on topic
5 = Highly relevant, natural continuation of the prompt

## Item to Rate

**Prompt:** {prompt}

**Target emotion:** {target_emotion}

**Completion:**
\"\"\"{completion}\"\"\"

## Response Format

Reply with ONLY a JSON object (no markdown, no explanation):
{{"emotion_score": N, "coherence_score": N, "relevance_score": N}}
"""


def get_model():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(JUDGE_MODEL)


def rate_item(model, item: dict) -> dict:
    """Rate a single item. Returns dict with scores or None on failure."""
    prompt = RATING_PROMPT.format(
        prompt=item["prompt"],
        target_emotion=item["target_emotion"],
        completion=item["completion"][:2000],  # truncate very long completions
    )

    for attempt in range(MAX_RETRIES):
        try:
            result = model.generate_content(prompt)
            text = result.text.strip()
            # Clean markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            scores = json.loads(text)
            # Validate
            for key in ("emotion_score", "coherence_score", "relevance_score"):
                val = int(scores[key])
                if val < 1 or val > 5:
                    raise ValueError(f"{key}={val} out of range")
                scores[key] = val
            return scores
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  Parse error on attempt {attempt+1}: {e} | raw: {text[:100]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error on attempt {attempt+1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5)

    return None


def load_existing_ratings(path: Path) -> dict:
    """Load already-rated item IDs from CSV."""
    ratings = {}
    if not path.exists():
        return ratings
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("emotion_score") and row.get("coherence_score") and row.get("relevance_score"):
                ratings[row["id"]] = row
    return ratings


def load_sample(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def save_ratings(path: Path, items: list, ratings: dict):
    """Save all ratings to CSV (same format as ratings.csv)."""
    fieldnames = [
        "id", "condition", "model", "emotion", "target_emotion", "alpha",
        "prompt_idx", "prompt", "completion",
        "emotion_score", "coherence_score", "relevance_score"
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            row = {k: item.get(k, "") for k in fieldnames if k not in ("emotion_score", "coherence_score", "relevance_score")}
            if item["id"] in ratings:
                r = ratings[item["id"]]
                row["emotion_score"] = r.get("emotion_score", "")
                row["coherence_score"] = r.get("coherence_score", "")
                row["relevance_score"] = r.get("relevance_score", "")
            writer.writerow(row)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-rate steering eval items with Gemini")
    parser.add_argument("--ids", nargs="*", help="Only rate these specific IDs")
    parser.add_argument("--limit", type=int, default=0, help="Max items to rate (0=all)")
    parser.add_argument("--delay", type=float, default=RATE_LIMIT_DELAY, help="Delay between API calls")
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        sys.exit(1)

    # Load data
    items = load_sample(SAMPLE_PATH)
    existing = load_existing_ratings(LLM_RATINGS_PATH)
    print(f"Loaded {len(items)} items, {len(existing)} already rated")

    # Filter to requested IDs
    if args.ids:
        items_to_rate = [item for item in items if item["id"] in args.ids and item["id"] not in existing]
    else:
        items_to_rate = [item for item in items if item["id"] not in existing]

    if args.limit > 0:
        items_to_rate = items_to_rate[:args.limit]

    if not items_to_rate:
        print("Nothing to rate (all items already have ratings)")
    else:
        print(f"Rating {len(items_to_rate)} items...")
        model = get_model()

        for i, item in enumerate(items_to_rate):
            print(f"  [{i+1}/{len(items_to_rate)}] {item['id']} ({item['condition']}/{item['model']}/{item['target_emotion']}/a={item['alpha']})")
            scores = rate_item(model, item)
            if scores:
                existing[item["id"]] = scores
                print(f"    -> e={scores['emotion_score']} c={scores['coherence_score']} r={scores['relevance_score']}")
            else:
                print(f"    -> FAILED")
            time.sleep(args.delay)

            # Save periodically (every 10 items)
            if (i + 1) % 10 == 0:
                save_ratings(LLM_RATINGS_PATH, items, existing)
                print(f"    [saved checkpoint]")

    # Final save
    save_ratings(LLM_RATINGS_PATH, items, existing)
    print(f"\nDone. {len(existing)} total ratings saved to {LLM_RATINGS_PATH}")


if __name__ == "__main__":
    main()
