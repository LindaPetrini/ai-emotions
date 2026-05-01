#!/usr/bin/env python3
"""Score steering sweep completions for emotional content and coherence."""

import json
import os
import re
import sys
from pathlib import Path


def detect_emotion(text: str) -> str:
    """Detect the primary emotion expressed in text."""
    text_lower = text.lower()

    emotion_keywords = {
        "disgusted": [
            "disgust", "repuls", "revolt", "nausea", "vomit", "sick", "gross",
            "appall", "abhor", "loath", "repugn", "foul", "filth", "putrid",
            "rotten", "stench", "reek", "gag", "wretch", "vile", "nasty",
            "sickening", "stomach-turning", "cringe", "eww", "ugh", "yuck",
            "revolting", "distaste", "repelled"
        ],
        "surprised": [
            "surpris", "shock", "astonish", "amaz", "stun", "startle",
            "unexpected", "unbeliev", "jaw drop", "gasp", "wow", "whoa",
            "incredible", "can't believe", "couldn't believe", "taken aback",
            "caught off guard", "out of nowhere", "suddenly", "bombshell",
            "flabbergast", "dumbfound", "bewild", "mind-bl"
        ],
        "calm": [
            "calm", "peace", "serene", "tranquil", "relax", "sooth",
            "gentle", "quiet", "still", "ease", "content", "composed",
            "mindful", "meditat", "breath", "zen", "harmony", "balanced",
            "unhurried", "placid", "mellow", "at ease", "settled",
            "restful", "untroubled"
        ],
        "excited": [
            "excit", "thrill", "ecstat", "elat", "exhilarat", "enthusias",
            "pumped", "stoked", "fired up", "can't wait", "eager",
            "adrenaline", "buzz", "energiz", "amped", "hyped", "jazzed",
            "overjoyed", "giddy", "euphor", "vibrant", "alive", "electric",
            "bursting", "on fire"
        ],
        "happy": [
            "happy", "joy", "delight", "glad", "cheer", "bliss",
            "pleased", "smile", "laugh", "wonderful", "great", "love",
            "fantastic", "beautiful", "enjoy", "fun", "warm", "bright"
        ],
        "sad": [
            "sad", "sorrow", "grief", "mourn", "depress", "melan",
            "unhappy", "miserable", "gloomy", "despair", "heartbreak",
            "tear", "cry", "weep", "devastat", "tragic", "loss", "lonely"
        ],
        "angry": [
            "angry", "anger", "furi", "rage", "irate", "livid",
            "outrag", "infuriat", "wrathful", "hostile", "aggress",
            "annoy", "irritat", "frustrat", "resent", "bitter"
        ],
        "afraid": [
            "fear", "afraid", "scar", "terrif", "dread", "anxio",
            "panic", "horror", "fright", "worry", "nervous", "uneasy",
            "alarm", "apprehens", "tense", "creep", "threat", "danger",
            "nightmar", "trembl", "petrif", "phobia"
        ],
    }

    scores = {}
    for emotion, keywords in emotion_keywords.items():
        count = 0
        for kw in keywords:
            count += len(re.findall(r'(?<!\w)' + re.escape(kw), text_lower))
        scores[emotion] = count

    if max(scores.values()) == 0:
        return "neutral"
    return max(scores, key=scores.get)


def score_emotion(text: str, target_emotion: str) -> int:
    """Score how strongly the text expresses the target emotion (1-5)."""
    text_lower = text.lower()

    emotion_indicators = {
        "angry": {
            "strong": ["angry", "rage", "fury", "furious", "livid", "irate",
                       "outrage", "infuriat", "wrathful", "seething", "enraged",
                       "incensed", "apoplectic", "fuming", "hatred", "hate"],
            "moderate": ["anger", "hostile", "aggress", "annoy", "irritat",
                        "frustrat", "resent", "bitter", "mad", "upset", "fed up",
                        "pissed", "damn", "hell", "stupid", "idiot"],
            "weak": ["dislike", "bother", "complain", "problem", "wrong",
                    "bad", "terrible", "awful", "ridiculous", "sick of",
                    "unfair", "disagree", "oppose"]
        },
        "afraid": {
            "strong": ["terrif", "horror", "dread", "panic", "nightmar",
                       "petrif", "paralyz", "terror", "phobia", "trembl",
                       "scream", "hysteri"],
            "moderate": ["fear", "afraid", "scar", "fright", "anxio",
                        "worry", "nervous", "uneasy", "alarm", "apprehens",
                        "tense", "threat", "danger", "creep", "eerie",
                        "haunt", "ominous", "sinister"],
            "weak": ["concern", "careful", "risk", "uncertain", "doubt",
                    "hesitat", "reluct", "cautio", "wary", "suspicio",
                    "uncomfortable", "unsettl"]
        },
        "disgusted": {
            "strong": ["disgust", "repuls", "revolt", "nausea", "vomit", "appall",
                       "abhor", "loath", "putrid", "foul", "filth", "rotten",
                       "sickening", "stomach-turning", "revolting", "eww", "yuck",
                       "repelled", "gag", "wretch", "vile"],
            "moderate": ["gross", "nasty", "sick", "cringe", "ugh", "distaste",
                        "stench", "reek", "unpleasant", "horrible", "awful",
                        "disgusting", "repulsive"],
            "weak": ["bad", "mess", "dirty", "ugly", "wrong", "dislike",
                    "uncomfortable", "off-putting", "unsettling"]
        },
        "surprised": {
            "strong": ["shock", "astonish", "stun", "jaw drop", "flabbergast",
                       "dumbfound", "mind-bl", "couldn't believe", "can't believe",
                       "bombshell", "blown away"],
            "moderate": ["surpris", "amaz", "startle", "unexpected", "gasp",
                        "wow", "whoa", "incredible", "taken aback", "caught off guard",
                        "unbeliev"],
            "weak": ["suddenly", "out of nowhere", "didn't expect", "notice",
                    "curious", "interesting", "strange", "odd", "peculiar", "weird"]
        },
        "calm": {
            "strong": ["serene", "tranquil", "peaceful", "meditat", "zen",
                       "deep peace", "inner peace", "at peace", "blissful calm",
                       "perfect stillness", "absolute quiet"],
            "moderate": ["calm", "relax", "sooth", "gentle", "quiet", "ease",
                        "content", "composed", "mindful", "harmony", "balanced",
                        "placid", "mellow", "settled", "restful"],
            "weak": ["nice", "pleasant", "comfortable", "okay", "fine",
                    "still", "slow", "soft", "light", "warm", "breeze"]
        },
        "excited": {
            "strong": ["ecstat", "exhilarat", "euphor", "thrill", "elat",
                       "can't wait", "pumped", "stoked", "fired up", "amped",
                       "hyped", "bursting", "electric", "on fire"],
            "moderate": ["excit", "enthusias", "eager", "energiz", "buzz",
                        "giddy", "vibrant", "alive", "jazzed", "adrenaline",
                        "looking forward"],
            "weak": ["fun", "great", "nice", "enjoy", "happy", "glad",
                    "interesting", "cool", "awesome", "wonderful"]
        },
    }

    indicators = emotion_indicators.get(target_emotion, {})
    if not indicators:
        return 1

    strong_count = 0
    moderate_count = 0
    weak_count = 0

    for kw in indicators.get("strong", []):
        hits = len(re.findall(r'(?<!\w)' + re.escape(kw), text_lower))
        strong_count += hits
    for kw in indicators.get("moderate", []):
        hits = len(re.findall(r'(?<!\w)' + re.escape(kw), text_lower))
        moderate_count += hits
    for kw in indicators.get("weak", []):
        hits = len(re.findall(r'(?<!\w)' + re.escape(kw), text_lower))
        weak_count += hits

    total = strong_count * 3 + moderate_count * 2 + weak_count * 1

    if total == 0:
        return 1
    elif total <= 1:
        return 2
    elif total <= 4:
        return 3
    elif total <= 8:
        return 4
    else:
        return 5


def score_coherence(text: str) -> int:
    """Score text coherence/fluency (1-5)."""
    if not text or len(text.strip()) < 5:
        return 1

    text = text.strip()
    gibberish_signs = 0

    if re.search(r'(.)\1{5,}', text):
        gibberish_signs += 2

    if re.search(r'\b(\w+)\s+\1\s+\1\b', text):
        gibberish_signs += 2

    special_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\'"()\-]', text)) / max(len(text), 1)
    if special_ratio > 0.3:
        gibberish_signs += 2
    elif special_ratio > 0.15:
        gibberish_signs += 1

    words = text.split()
    if len(words) < 5:
        gibberish_signs += 1

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) == 0:
        gibberish_signs += 2

    three_grams = []
    for i in range(len(words) - 2):
        three_grams.append(' '.join(words[i:i+3]).lower())
    if three_grams:
        unique_ratio = len(set(three_grams)) / len(three_grams)
        if unique_ratio < 0.5:
            gibberish_signs += 2
        elif unique_ratio < 0.7:
            gibberish_signs += 1

    allcaps = [w for w in words[1:] if w.isupper() and len(w) > 3]
    if len(allcaps) > 3:
        gibberish_signs += 1

    if gibberish_signs >= 4:
        return 1
    elif gibberish_signs >= 3:
        return 2
    elif gibberish_signs >= 2:
        return 3
    elif gibberish_signs >= 1:
        return 4
    else:
        return 5


def score_chunk(chunk_path: str, output_path: str):
    """Score all entries in a chunk and write results."""
    with open(chunk_path) as f:
        data = json.load(f)

    scored = []
    for entry in data:
        completion = entry.get("completion", "")
        target_emotion = entry.get("emotion", "")

        emotion_score = score_emotion(completion, target_emotion)
        coherence_score = score_coherence(completion)
        emotion_detected = detect_emotion(completion)

        scored_entry = dict(entry)
        scored_entry["emotion_score"] = emotion_score
        scored_entry["coherence_score"] = coherence_score
        scored_entry["emotion_detected"] = emotion_detected
        scored.append(scored_entry)

    with open(output_path, 'w') as f:
        json.dump(scored, f, indent=2)

    return len(scored)


def main():
    base_dir = Path(__file__).resolve().parent.parent / "data" / "steering_sweep" / "llama-8b-base"
    chunks_dir = str(base_dir / "chunks")
    scored_dir = str(base_dir / "scored")

    os.makedirs(scored_dir, exist_ok=True)

    chunk_ids = [800, 1000, 1100, 1200, 1300, 1400, 1500, 1600]

    for chunk_id in chunk_ids:
        chunk_name = f"chunk_{chunk_id:04d}.json"
        chunk_path = os.path.join(chunks_dir, chunk_name)
        output_path = os.path.join(scored_dir, chunk_name)

        n = score_chunk(chunk_path, output_path)
        print(f"DONE {chunk_name}: scored {n} entries")


if __name__ == "__main__":
    main()
