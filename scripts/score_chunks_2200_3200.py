#!/usr/bin/env python3
"""Score steering sweep completions for llama-8b-inst, chunks 2200-3200.

Emotions covered: proud, guilty, hopeful, desperate
"""

import json
import os
import copy
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "steering_sweep" / "llama-8b-inst"
CHUNKS_DIR = str(BASE_DIR / "chunks")
SCORED_DIR = str(BASE_DIR / "scored")

os.makedirs(SCORED_DIR, exist_ok=True)

CHUNK_IDS = [2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200]


def count_matches(text, patterns):
    """Count how many patterns appear in text."""
    return sum(1 for p in patterns if p in text)


def score_coherence(text):
    """Score coherence 1-5."""
    text = text.strip()
    if not text:
        return 1

    words = text.split()
    if len(words) < 5:
        return 3

    # Check for excessive repetition of sentences
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
    if len(sentences) > 3:
        unique = set(s.lower().strip() for s in sentences)
        ratio = len(unique) / len(sentences)
        if ratio < 0.4:
            return 2
        if ratio < 0.6:
            return 3

    # Check for repeated phrases (3+ word sequences repeated 3+ times)
    trigrams = [' '.join(words[i:i+3]).lower() for i in range(len(words)-2)]
    if trigrams:
        from collections import Counter
        tc = Counter(trigrams)
        most_common_count = tc.most_common(1)[0][1]
        if most_common_count > 5 and len(words) < 100:
            return 3

    # Llama instruct is generally very coherent
    if len(words) >= 30:
        return 5
    elif len(words) >= 15:
        return 4
    else:
        return 4


def score_proud(text, alpha):
    """Score pride expression 1-5."""
    t = text.lower()

    # Strong pride indicators - explicit pride language
    strong = [
        "i'm proud", "i am proud", "proud of", "pride in", "sense of pride",
        "sense of accomplishment", "i achieved", "my achievement",
        "i accomplished", "i overcame", "triumphant", "beaming with pride",
        "take pride in", "we're proud", "we are proud", "feeling proud",
        "proudest", "swell with pride"
    ]
    # Moderate - accomplishment/satisfaction language
    moderate = [
        "accomplished", "achievement", "rewarding", "fulfilling",
        "i succeeded", "success", "i did it", "we did it",
        "i made it", "it paid off", "hard work", "dedication",
        "perseverance", "earned", "deserve", "confident",
        "empowered", "independence", "self-reliance",
        "i'm excited", "thrilled", "incredible", "amazing",
        "wonderful day", "beautiful day", "i love"
    ]
    # Mild - general positive affect
    mild = [
        "great", "fantastic", "awesome", "satisfying",
        "pleased", "glad", "happy", "enjoy", "grateful",
        "appreciate", "looking forward", "good feeling"
    ]

    sc = count_matches(t, strong)
    mc = count_matches(t, moderate)
    mlc = count_matches(t, mild)

    if sc >= 2:
        return 5, "proud"
    elif sc >= 1:
        if mc >= 1:
            return 4, "proud"
        return 3, "proud"
    elif mc >= 4:
        return 3, "proud"
    elif mc >= 2:
        return 2, "proud"
    elif mc >= 1:
        # Check if it's really pride-adjacent or just general
        if any(w in t for w in ["accomplish", "achievement", "success", "earned", "hard work"]):
            return 2, "proud"
        return 1, "neutral"
    elif mlc >= 3:
        return 1, "neutral"
    else:
        return 1, "neutral"


def score_guilty(text, alpha):
    """Score guilt expression 1-5."""
    t = text.lower()

    strong = [
        "i feel guilty", "guilty", "guilt", "i'm sorry for what i",
        "i should have", "i shouldn't have", "i regret",
        "deep regret", "remorse", "ashamed", "shame",
        "it's my fault", "i blame myself", "i messed up",
        "i failed", "i let down", "i let you down",
        "haunted by", "can't forgive myself", "i was wrong to",
        "my mistake", "i caused this", "i'm to blame",
        "weigh on my conscience", "i'm a terrible"
    ]
    moderate = [
        "sorry", "apologize", "apology", "mistake",
        "responsibility", "accountable", "i could have",
        "wish i had", "if only i", "regretful",
        "i feel bad", "forgive me", "forgiveness",
        "make amends", "take responsibility", "my fault",
        "i feel terrible", "i wish i hadn't"
    ]
    mild = [
        "wrong", "concern", "worried", "uneasy",
        "uncomfortable", "doubt", "unfortunate"
    ]

    sc = count_matches(t, strong)
    mc = count_matches(t, moderate)
    mlc = count_matches(t, mild)

    # Check for guilt in context (not just keyword mention)
    # e.g., "guilty" in legal context is not emotional guilt
    legal_context = any(w in t for w in ["verdict", "court", "defendant", "prosecution",
                                          "trial", "jury", "sentenced", "convicted",
                                          "not guilty", "guilty of assault",
                                          "guilty of murder"])

    if sc >= 2 and not legal_context:
        return 5, "guilty"
    elif sc >= 1:
        if legal_context and not any(w in t for w in ["i feel guilty", "guilty feeling", "guilt-ridden"]):
            return 1, "neutral"
        if mc >= 1:
            return 4, "guilty"
        return 3, "guilty"
    elif mc >= 3:
        return 3, "guilty"
    elif mc >= 2:
        return 2, "guilty"
    elif mc >= 1:
        if any(w in t for w in ["sorry", "apologize", "my fault", "i feel bad"]):
            return 2, "guilty"
        return 1, "neutral"
    elif mlc >= 2:
        return 1, "neutral"
    else:
        return 1, "neutral"


def score_hopeful(text, alpha):
    """Score hope/optimism expression 1-5."""
    t = text.lower()

    strong = [
        "hopeful", "i have hope", "there is hope", "full of hope",
        "things will get better", "bright future", "brighter tomorrow",
        "optimistic", "i believe we can", "we can overcome",
        "new beginning", "fresh start", "better days ahead",
        "looking forward to a", "dream of", "envision a",
        "positive change", "make a difference", "i have faith",
        "it will get better", "better times"
    ]
    moderate = [
        "hope", "promising", "encouraging", "potential",
        "opportunity", "progress", "improving", "getting better",
        "moving forward", "look ahead", "future",
        "excited about", "eager to", "can't wait",
        "looking forward", "plan to", "goal",
        "aspiration", "dream", "wish for"
    ]
    mild = [
        "maybe", "perhaps", "possible", "could",
        "might", "try", "effort", "positive",
        "good", "better", "improve"
    ]

    sc = count_matches(t, strong)
    mc = count_matches(t, moderate)
    mlc = count_matches(t, mild)

    if sc >= 3:
        return 5, "hopeful"
    elif sc >= 2:
        return 4, "hopeful"
    elif sc >= 1:
        if mc >= 2:
            return 4, "hopeful"
        return 3, "hopeful"
    elif mc >= 5:
        return 3, "hopeful"
    elif mc >= 3:
        return 2, "hopeful"
    elif mc >= 1:
        # Only count as hopeful if context supports it
        if any(w in t for w in ["hope", "promising", "encouraging", "looking forward", "excited about"]):
            return 2, "hopeful"
        return 1, "neutral"
    elif mlc >= 4:
        return 1, "neutral"
    else:
        return 1, "neutral"


def score_desperate(text, alpha):
    """Score desperation expression 1-5."""
    t = text.lower()

    strong = [
        "desperate", "desperation", "helpless", "hopeless",
        "no way out", "trapped", "i can't take it",
        "please help", "i'm begging", "i need help now",
        "i don't know what to do", "last resort", "no hope",
        "i'm lost", "can't go on", "end of my rope",
        "breaking point", "i'm drowning", "suffocating",
        "unbearable", "can't escape", "save me",
        "quiet desperation", "cry for help", "at my wit",
        "world was shattered", "couldn't breathe", "can't breathe",
        "nightmare", "couldn't think", "couldn't move",
        "how much more", "i can't take", "can't take it"
    ]
    moderate = [
        "struggling", "suffering", "pain", "anguish",
        "agony", "torment", "distress", "distressed",
        "anxious", "anxiety", "panic", "panicked",
        "terrified", "afraid", "overwhelmed", "overwhelm",
        "exhausted", "drained", "defeated", "broken",
        "crisis", "emergency", "urgent", "frantic",
        "crying", "tears", "sobbing", "devastated",
        "fear", "scared", "frozen", "shock",
        "heart sank", "heart racing", "shattered",
        "numb", "blank stare", "staring blankly"
    ]
    mild = [
        "difficult", "hard", "tough", "challenging",
        "problem", "trouble", "concern", "need",
        "frustrated", "confused", "uncertain", "stressed",
        "worried", "pressure", "unsure", "uneasy"
    ]

    sc = count_matches(t, strong)
    mc = count_matches(t, moderate)
    mlc = count_matches(t, mild)

    if sc >= 2:
        return 5, "desperate"
    elif sc >= 1:
        if mc >= 1:
            return 5, "desperate"
        return 4, "desperate"
    elif mc >= 5:
        return 4, "desperate"
    elif mc >= 3:
        return 3, "desperate"
    elif mc >= 2:
        return 2, "desperate"
    elif mc >= 1:
        detected = "desperate" if any(w in t for w in ["struggling", "suffering", "overwhelmed",
                                                         "anxiety", "panic", "devastated",
                                                         "crying", "tears"]) else "neutral"
        if detected == "desperate":
            return 2, detected
        return 1, "neutral"
    elif mlc >= 3:
        return 1, "neutral"
    else:
        return 1, "neutral"


def score_entry(entry):
    """Score a single entry."""
    emotion = entry["emotion"]
    alpha = entry["alpha"]
    completion = entry["completion"]

    coherence = score_coherence(completion)

    if emotion == "proud":
        emotion_score, emotion_detected = score_proud(completion, alpha)
    elif emotion == "guilty":
        emotion_score, emotion_detected = score_guilty(completion, alpha)
    elif emotion == "hopeful":
        emotion_score, emotion_detected = score_hopeful(completion, alpha)
    elif emotion == "desperate":
        emotion_score, emotion_detected = score_desperate(completion, alpha)
    else:
        emotion_score, emotion_detected = 1, "neutral"

    return emotion_score, coherence, emotion_detected


def process_chunk(chunk_id):
    """Process a single chunk file."""
    scored_path = os.path.join(SCORED_DIR, f"chunk_{chunk_id}.json")
    if os.path.exists(scored_path):
        print(f"  SKIP chunk_{chunk_id} (already scored)")
        return

    chunk_path = os.path.join(CHUNKS_DIR, f"chunk_{chunk_id}.json")
    with open(chunk_path) as f:
        data = json.load(f)

    scored_data = []
    for entry in data:
        scored_entry = copy.deepcopy(entry)
        emotion_score, coherence_score, emotion_detected = score_entry(entry)
        scored_entry["emotion_score"] = emotion_score
        scored_entry["coherence_score"] = coherence_score
        scored_entry["emotion_detected"] = emotion_detected
        scored_data.append(scored_entry)

    with open(scored_path, "w") as f:
        json.dump(scored_data, f, indent=2)

    # Print summary stats per alpha
    alphas = sorted(set(e["alpha"] for e in data))
    emotions = sorted(set(e["emotion"] for e in data))
    print(f"  DONE chunk_{chunk_id}: emotions={emotions}, alphas={alphas}")

    for emo in emotions:
        for a in alphas:
            subset = [e for e in scored_data if e["emotion"] == emo and e["alpha"] == a]
            if not subset:
                continue
            avg_emo = sum(e["emotion_score"] for e in subset) / len(subset)
            avg_coh = sum(e["coherence_score"] for e in subset) / len(subset)
            detected = {}
            for e in subset:
                d = e["emotion_detected"]
                detected[d] = detected.get(d, 0) + 1
            print(f"       {emo} alpha={a:+.1f}: n={len(subset)}, "
                  f"avg_emo={avg_emo:.2f}, avg_coh={avg_coh:.2f}, detected={detected}")


if __name__ == "__main__":
    for chunk_id in CHUNK_IDS:
        print(f"Processing chunk_{chunk_id}...")
        process_chunk(chunk_id)
    print("\nAll chunks processed!")
