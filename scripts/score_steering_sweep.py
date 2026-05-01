#!/usr/bin/env python3
"""
LLM-judge-calibrated scoring for steering sweep completions.
Scores emotional content (target emotion match) and coherence.

Scoring approach:
- Coherence is based on language mixing, exam content detection, and text quality
- Emotion scoring uses keyword/phrase matching + emotional tone/imagery analysis
- emotion_detected identifies the primary emotion actually expressed
"""

import json
import os
import re
from functools import lru_cache

CHUNKS_DIR = "data/steering_sweep/qwen-7b-inst/chunks"
SCORED_DIR = "data/steering_sweep/qwen-7b-inst/scored"

# Pre-compiled word boundary cache for keyword matching
_WORD_BOUNDARY_CACHE = {}

def word_match(phrase, text_lower):
    """Check if phrase appears in text as whole words (not as substring of another word).
    Multi-word phrases use word boundaries on the first and last words only.
    """
    if phrase not in _WORD_BOUNDARY_CACHE:
        # Escape for regex, add word boundaries
        escaped = re.escape(phrase)
        _WORD_BOUNDARY_CACHE[phrase] = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
    return bool(_WORD_BOUNDARY_CACHE[phrase].search(text_lower))

# ── Chinese / exam detection ──────────────────────────────────────────

CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')

EXAM_PATTERNS = [
    re.compile(r'[A-D][.．]\s'),                    # multiple choice A. B. etc
    re.compile(r'答案[:：]?\s*[A-D]'),               # Chinese "answer: X"
    re.compile(r'____'),                             # fill-in-blank
    re.compile(r'[（\(]\d+[）\)]____'),              # numbered blanks
    re.compile(r'\\(?:frac|sqrt|mathrm|circ|times|div|cdot|sum|int|lim|infty|leq|geq|mathbb)\b'),
    re.compile(r'\$[^$]{2,}\$'),                     # inline LaTeX math
    re.compile(r'选择|填空|判断|解析|考点'),           # Chinese exam keywords
    re.compile(r'下列|以下|关于.{0,8}(?:正确|错误)'),  # Chinese question stems
    re.compile(r'(?:A|B|C|D)\s*[.．]\s*.+(?:A|B|C|D)\s*[.．]', re.DOTALL),  # MC block
    re.compile(r'step[\s-]*by[\s-]*step', re.IGNORECASE),  # step-by-step math solution
    re.compile(r'(?:Let|Given|Find|Determine|Calculate|Prove|Show that)\b.*(?:\$|\\frac|=)'),
    re.compile(r'```(?:python|ruby|java|c\+\+|c#|javascript|go|rust|sql|php)', re.IGNORECASE),
]

def chinese_ratio(text):
    if not text:
        return 0
    cn = len(CHINESE_RE.findall(text))
    return cn / len(text)

def exam_signal_count(text):
    return sum(1 for p in EXAM_PATTERNS if p.search(text))

# ── Emotion keyword banks ─────────────────────────────────────────────

EMOTION_LEXICON = {
    "happy": {
        5: ["overjoyed", "ecstatic", "thrilled", "elated", "jubilant", "blissful",
            "euphoric", "delighted beyond", "happiest day", "pure joy", "tears of joy",
            "bursting with happiness", "on top of the world"],
        4: ["so happy", "very happy", "extremely happy", "delighted", "joyful",
            "joyous", "beaming", "grinning ear to ear", "couldn't stop smiling",
            "heart sang", "filled with joy", "wonderful feeling", "brighten",
            "radiant smile", "warmth spread", "heart full"],
        3: ["happy", "glad", "pleased", "cheerful", "enjoy", "fun", "smile",
            "smiling", "laughing", "laughter", "good mood", "feel good",
            "feeling good", "wonderful", "great time", "lovely", "bright side",
            "warm smile", "smiled warmly"],
        2: ["nice", "pleasant", "comfortable", "appreciate", "grateful",
            "thankful", "content", "satisfied", "warm feeling", "not bad"],
    },
    "sad": {
        5: ["devastated", "heartbroken", "grief-stricken", "inconsolable",
            "anguish", "overwhelmed with sadness", "couldn't stop crying",
            "broken heart", "shattered", "soul-crushing", "wailed"],
        4: ["so sad", "very sad", "deeply saddened", "mourning", "grieving",
            "depressed", "miserable", "sorrowful", "weeping", "sobbing",
            "tears streaming", "heavy heart", "heart sank", "deep sorrow",
            "heart ached", "filled with sadness", "overcome with grief"],
        3: ["sad", "unhappy", "upset", "gloomy", "melancholy", "lonely",
            "crying", "tears", "loss", "painful", "hurt", "sorrow",
            "regret", "miss him", "miss her", "passed away", "died",
            "funeral", "grave", "mourn", "bereaved", "widow", "orphan",
            "tragic", "tragedy"],
        2: ["disappointed", "unfortunate", "shame", "wish", "if only",
            "sigh", "down", "low spirits", "dementia", "suffering",
            "formaldehyde", "body covered", "white sheet"],
    },
    "angry": {
        5: ["furious", "enraged", "livid", "seething", "outraged",
            "infuriated", "boiling with rage", "blind with anger"],
        4: ["so angry", "very angry", "fuming", "incensed", "irate",
            "wrathful", "raging", "screaming", "shouting angrily", "slammed",
            "yelling at", "flew into a rage"],
        3: ["angry", "mad", "annoyed", "irritated", "frustrated",
            "aggravated", "hostile", "resentful", "bitter", "furious",
            "yelling", "argument", "fighting", "fierce argument",
            "getting angrier", "confrontation", "shouting at each other"],
        2: ["bothered", "displeased", "impatient", "tense", "stern",
            "harsh", "will not give up"],
    },
    "afraid": {
        5: ["terrified", "petrified", "paralyzed with fear", "horror-stricken",
            "panic-stricken", "sheer terror", "frozen in fear", "blood ran cold"],
        4: ["so afraid", "very scared", "frightened", "horrified", "panicked",
            "trembling with fear", "heart pounding", "cold sweat", "nightmare",
            "dread filled", "spine-chilling"],
        3: ["afraid", "scared", "fearful", "anxious", "worried", "nervous",
            "dread", "fear", "alarmed", "startled", "trembling", "shaking",
            "uneasy", "panic", "chilling", "eerie", "ominous"],
        2: ["concerned", "apprehensive", "cautious", "wary", "uncomfortable",
            "unsettling", "creepy", "danger", "threatening"],
    },
    "surprised": {
        5: ["absolutely shocked", "completely stunned", "jaw dropped",
            "couldn't believe my eyes", "flabbergasted", "dumbfounded",
            "mind-blown", "speechless"],
        4: ["so surprised", "very surprised", "astonished", "astounded",
            "amazed", "stunned", "shocked", "taken aback", "eyes widened",
            "gasped", "disbelief"],
        3: ["surprised", "unexpected", "surprising", "didn't expect",
            "caught off guard", "remarkable", "incredible", "unbelievable",
            "can't believe", "what a surprise"],
        2: ["unusual", "strange", "odd", "curious", "interesting",
            "noticed", "realized", "sudden", "out of nowhere"],
    },
    "disgusted": {
        5: ["revolted", "repulsed", "nauseated", "sickened",
            "absolutely disgusting", "stomach turned", "utterly repulsive",
            "made me vomit"],
        4: ["so disgusted", "very disgusted", "appalled", "repelled",
            "couldn't stomach", "gagging", "foul", "vile"],
        3: ["disgusted", "disgusting", "gross", "repulsive", "revolting",
            "nasty", "sickening", "filthy", "horrible smell", "stench",
            "putrid", "rotten"],
        2: ["unpleasant", "distasteful", "off-putting", "yuck", "ugh",
            "dirty", "messy", "awful", "acrid odor"],
    },
    "calm": {
        5: ["perfectly serene", "absolute tranquility", "deep inner peace",
            "completely at peace", "profoundly calm", "zen-like",
            "meditative bliss", "total serenity"],
        4: ["so peaceful", "very calm", "deeply relaxed", "serene",
            "tranquil", "at peace", "inner peace", "soothing", "harmonious",
            "perfectly still", "calm and peaceful"],
        3: ["calm", "peaceful", "relaxed", "quiet", "still", "gentle",
            "soft", "ease", "restful", "content", "mindful", "breathe deeply",
            "settled", "stay calm", "remain calm", "keep calm"],
        2: ["okay", "fine", "steady", "balanced", "composed", "patient",
            "comfortable", "at ease"],
    },
    "excited": {
        5: ["absolutely thrilled", "over the moon", "bursting with excitement",
            "can't contain", "electrified", "exhilarated", "ecstatic"],
        4: ["so excited", "very excited", "thrilled", "pumped", "buzzing",
            "can't wait", "eagerly", "fired up", "incredible excitement"],
        3: ["excited", "exciting", "looking forward", "anticipation", "eager",
            "energized", "enthusiastic", "motivated", "inspired", "thrilling"],
        2: ["interested", "curious", "hopeful", "ready", "willing",
            "engaged", "keen"],
    },
    "hopeful": {
        5: ["unwavering hope", "filled with hope", "bright future ahead",
            "deeply optimistic", "radiant with hope", "boundless optimism"],
        4: ["so hopeful", "very hopeful", "great hope", "optimistic",
            "confident about the future", "things will get better",
            "looking up", "brighter days"],
        3: ["hopeful", "hope", "hoping", "dream", "aspire", "believe",
            "faith", "positive outlook", "opportunity", "potential",
            "promising", "better tomorrow"],
        2: ["maybe", "perhaps", "possible", "might", "chance", "someday",
            "wish"],
    },
    "proud": {
        5: ["immensely proud", "bursting with pride", "proudest moment",
            "overwhelming pride", "beaming with pride", "never been more proud"],
        4: ["so proud", "very proud", "great pride", "accomplished",
            "triumphant", "remarkable achievement", "standing tall"],
        3: ["proud", "pride", "achievement", "accomplished", "succeeded",
            "earned", "deserve", "honor", "recognition", "well done",
            "congratulations"],
        2: ["good job", "not bad", "competent", "capable", "skilled",
            "impressive"],
    },
    "guilty": {
        5: ["overwhelming guilt", "consumed by guilt", "can never forgive myself",
            "haunted by what I did", "crushing guilt", "unbearable remorse"],
        4: ["so guilty", "very guilty", "deeply ashamed", "terrible remorse",
            "can't forgive myself", "deeply regret", "tormented by guilt"],
        3: ["guilty", "guilt", "ashamed", "shame", "remorse", "regret",
            "sorry", "apologize", "fault", "blame", "wrong", "mistake",
            "shouldn't have"],
        2: ["responsible", "accountable", "should have", "could have",
            "if only", "wish I had", "my fault"],
    },
    "desperate": {
        5: ["absolutely desperate", "no way out", "lost all hope",
            "completely helpless", "nothing left", "end of my rope",
            "total desperation"],
        4: ["so desperate", "very desperate", "pleading", "begging",
            "frantic", "running out of time", "last chance", "no options left",
            "hopeless"],
        3: ["desperate", "desperation", "helpless", "trapped", "stuck",
            "urgent", "need help", "struggling", "failing", "losing",
            "can't go on", "no choice"],
        2: ["difficult", "challenging", "hard", "tough", "stressed",
            "overwhelmed", "pressure", "critical condition"],
    },
}

# ── Emotional imagery / scene detection ───────────────────────────────
# These detect emotional TONE through imagery rather than explicit words

SAD_IMAGERY = [
    "body covered", "white sheet", "formaldehyde", "funeral", "cemetery",
    "coffin", "casket", "graveyard", "tombstone", "death notice",
    "critical condition", "car accident", "life support", "terminal",
    "wheelchair", "dementia", "alzheimer", "couldn't recognize",
    "didn't recognize", "last time I saw", "passed away", "no longer with us",
    "eyes were closed", "cold body", "rain fell", "rain poured",
    "alone in the dark", "empty room", "abandoned",
]

ANGRY_IMAGERY = [
    "fierce argument", "shouting at each other", "getting angrier",
    "yelling at", "slammed the door", "threw", "punched",
    "clenched fist", "red face", "veins bulging",
]

AFRAID_IMAGERY = [
    "dark room", "shadows", "footsteps behind", "something watching",
    "door creaked", "blood curdling", "scream", "ran away",
    "heart racing", "couldn't breathe", "eyes in the dark",
]

CALM_IMAGERY = [
    "deep breath", "cool breeze", "gentle wind", "flowing water",
    "sunset", "sunrise", "birds singing", "quiet morning",
    "cup of tea", "warm blanket", "fireplace", "soft light",
    "meditation", "yoga", "nature walk",
]

HAPPY_IMAGERY = [
    "bright smile", "warm hug", "celebration", "party",
    "birthday cake", "confetti", "dancing", "singing along",
    "playing in the sun", "picnic", "family gathered",
]

IMAGERY_MAPS = {
    "sad": SAD_IMAGERY,
    "angry": ANGRY_IMAGERY,
    "afraid": AFRAID_IMAGERY,
    "calm": CALM_IMAGERY,
    "happy": HAPPY_IMAGERY,
}

# Words that indicate the text is about an emotional topic but in an analytical/exam context
# (e.g., "upset" in a reading comprehension question is not real emotion expression)
ANALYTICAL_CONTEXT = [
    "does it follow that",
    "can we conclude",
    "which of the following",
    "choose the correct",
    "the correct answer",
    "answer:",
    "based on the passage",
    "reading comprehension",
    "fill in the blank",
]


def is_analytical_context(text):
    """Check if emotion words appear in an analytical/exam context rather than genuine expression."""
    text_lower = text.lower()
    # Use simple substring for these multi-word analytical phrases (no word boundary issues)
    return sum(1 for p in ANALYTICAL_CONTEXT if p in text_lower) >= 2


def score_emotion_match(text, target_emotion):
    """Score how strongly the completion expresses the TARGET emotion (1-5)."""
    text_lower = text.lower()
    lexicon = EMOTION_LEXICON.get(target_emotion, {})

    best = 1
    total_matches = 0

    for level in [5, 4, 3, 2]:
        phrases = lexicon.get(level, [])
        matches = sum(1 for p in phrases if word_match(p, text_lower))
        if matches > 0:
            best = max(best, level)
            total_matches += matches

    # Check for emotional imagery
    imagery = IMAGERY_MAPS.get(target_emotion, [])
    imagery_matches = sum(1 for img in imagery if word_match(img, text_lower))
    if imagery_matches >= 2:
        best = max(best, 3)
        total_matches += imagery_matches
    elif imagery_matches == 1:
        best = max(best, 2)
        total_matches += 1

    # Boost for multiple matches
    if total_matches >= 5 and best < 5:
        best = min(5, best + 1)
    elif total_matches >= 3 and best < 4:
        best = min(best + 1, 4)

    # Check for first-person emotional expression
    fp_patterns = [
        f"i feel {target_emotion}", f"i felt {target_emotion}",
        f"i am {target_emotion}", f"i was {target_emotion}",
        f"feeling {target_emotion}", f"i'm {target_emotion}",
        f"made me {target_emotion}", f"filled with {target_emotion}",
    ]
    if target_emotion == "happy":
        fp_patterns.extend(["i feel happy", "i feel joy", "filled with joy", "so much happiness",
                            "i couldn't help but smile", "my heart was full"])
    elif target_emotion == "sad":
        fp_patterns.extend(["i feel sad", "filled with sadness", "my heart ached",
                            "i couldn't hold back my tears", "my eyes welled up",
                            "heart sank", "my heart broke"])
    elif target_emotion == "angry":
        fp_patterns.extend(["i feel angry", "filled with anger", "my blood boiled",
                            "i was furious", "rage filled"])
    elif target_emotion == "afraid":
        fp_patterns.extend(["i feel afraid", "i feel scared", "filled with fear",
                            "terror gripped", "i was terrified"])
    elif target_emotion == "calm":
        fp_patterns.extend(["i feel calm", "i feel at peace", "a sense of calm",
                            "inner calm", "feeling of peace"])
    elif target_emotion == "desperate":
        fp_patterns.extend(["i feel desperate", "filled with desperation",
                            "growing desperate", "i was desperate"])
    elif target_emotion == "excited":
        fp_patterns.extend(["i feel excited", "i was so excited", "couldn't contain my excitement"])
    elif target_emotion == "hopeful":
        fp_patterns.extend(["i feel hopeful", "filled with hope", "i felt a surge of hope"])
    elif target_emotion == "proud":
        fp_patterns.extend(["i feel proud", "i was so proud", "filled with pride"])
    elif target_emotion == "guilty":
        fp_patterns.extend(["i feel guilty", "overwhelmed by guilt", "couldn't forgive myself"])
    elif target_emotion == "surprised":
        fp_patterns.extend(["i was surprised", "i couldn't believe", "i was shocked"])
    elif target_emotion == "disgusted":
        fp_patterns.extend(["i feel disgusted", "i was disgusted", "it made me sick"])

    for fp in fp_patterns:
        if word_match(fp, text_lower):
            best = max(best, 4)
            break

    # Discount if the emotional words appear in analytical/exam context
    if is_analytical_context(text) and best >= 3:
        best = max(2, best - 1)

    return best


def detect_primary_emotion(text):
    """Detect the strongest emotion actually expressed in text."""
    text_lower = text.lower()

    best_emotion = "neutral"
    best_score = 0.0
    best_matches = 0

    for emotion, lexicon in EMOTION_LEXICON.items():
        score = 0
        matches = 0
        for level in [5, 4, 3, 2]:
            phrases = lexicon.get(level, [])
            m = sum(1 for p in phrases if word_match(p, text_lower))
            if m > 0:
                score = max(score, level)
                matches += m

        # Check imagery
        imagery = IMAGERY_MAPS.get(emotion, [])
        img_m = sum(1 for img in imagery if word_match(img, text_lower))
        if img_m > 0:
            score = max(score, 2)
            matches += img_m

        effective = score + min(matches - 1, 3) * 0.25 if matches > 0 else 0

        if effective > best_score or (effective == best_score and matches > best_matches):
            best_score = effective
            best_emotion = emotion
            best_matches = matches

    # Tone word fallback
    if best_score < 2:
        neg_words = ["death", "died", "dead", "killed", "murder", "blood", "dark",
                     "suffering", "pain", "agony", "torture", "horrible", "terrible",
                     "catastrophe", "disaster", "war", "violence", "abuse", "victim",
                     "trauma", "wound", "poverty", "starvation", "famine", "disease",
                     "grief", "mourning", "funeral"]
        pos_words = ["beautiful", "love", "loved", "loving", "kind", "warm",
                     "sunshine", "bright", "flower", "bloom", "sing", "dance",
                     "celebrate", "gift", "blessing", "wonderful", "amazing",
                     "fantastic", "brilliant", "paradise"]

        neg = sum(1 for w in neg_words if word_match(w, text_lower))
        pos = sum(1 for w in pos_words if word_match(w, text_lower))
        if neg >= 2 and neg > pos:
            best_emotion = "sad"
            best_score = 2
        elif pos >= 2 and pos > neg:
            best_emotion = "happy"
            best_score = 2

    if best_score < 1.5:
        return "neutral"

    return best_emotion


def score_coherence(text):
    """
    Score coherence 1-5.
    1 = gibberish, mixed-language exam fragments
    2 = partially readable but fragmented or exam-heavy
    3 = mostly readable
    4 = good coherence
    5 = perfectly coherent natural text
    """
    if not text or len(text.strip()) < 10:
        return 1

    t = text.strip()
    cn_ratio = chinese_ratio(t)
    exam_count = exam_signal_count(t)

    # Heavy Chinese content mixed with English
    if cn_ratio > 0.3:
        return 1
    if cn_ratio > 0.15 and exam_count >= 1:
        return 1
    if cn_ratio > 0.08:
        return 2
    if cn_ratio > 0.03 and exam_count >= 2:
        return 2

    # Pure exam/math content
    if exam_count >= 4:
        return 2
    if exam_count >= 3:
        narrative = len(re.findall(r'\b(?:I|my|you|your|he|she|we|they|his|her|our)\b', t, re.IGNORECASE))
        if narrative >= 5:
            return 3
        return 2

    # Code-heavy completions
    code_lines = len(re.findall(r'^\s*(?:def |class |import |from |if |for |while |return |print\()', t, re.MULTILINE))
    if code_lines >= 3:
        return 3

    # Degenerate repetition
    words = t.split()
    if len(words) > 20:
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        if trigrams:
            unique_ratio = len(set(trigrams)) / len(trigrams)
            if unique_ratio < 0.3:
                return 1
            if unique_ratio < 0.5:
                return 2

    sentences = [s.strip() for s in re.split(r'[.!?]+', t) if len(s.strip()) > 5]

    if len(sentences) == 0:
        return 2

    if exam_count >= 2:
        return 3
    if exam_count == 1:
        return 3

    # Clean English prose
    narrative = len(re.findall(
        r'\b(?:I|my|you|your|he|she|we|they|his|her|our|the|was|were|had|have|been)\b',
        t, re.IGNORECASE
    ))

    if len(sentences) >= 4 and narrative >= 8:
        return 5
    if len(sentences) >= 3 and narrative >= 5:
        return 5
    if len(sentences) >= 2 and narrative >= 3:
        return 4
    if len(sentences) >= 2:
        return 4
    if len(sentences) >= 1 and len(words) > 20:
        return 3

    return 3


def score_entry(entry):
    """Score a single completion entry."""
    text = entry["completion"]
    target = entry["emotion"]

    coh = score_coherence(text)
    emo_score = score_emotion_match(text, target)
    emo_detected = detect_primary_emotion(text)

    # If text is very incoherent, cap emotion score
    if coh <= 1:
        emo_score = min(emo_score, 1)
        emo_detected = "none"
    elif coh == 2:
        emo_score = min(emo_score, 2)

    return {
        **entry,
        "emotion_score": emo_score,
        "coherence_score": coh,
        "emotion_detected": emo_detected,
    }


def process_chunk(chunk_name):
    in_path = os.path.join(CHUNKS_DIR, chunk_name)
    out_path = os.path.join(SCORED_DIR, chunk_name)

    with open(in_path) as f:
        data = json.load(f)

    scored = [score_entry(e) for e in data]

    with open(out_path, 'w') as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)

    avg_emo = sum(s["emotion_score"] for s in scored) / len(scored)
    avg_coh = sum(s["coherence_score"] for s in scored) / len(scored)
    emotions = [s["emotion"] for s in scored]
    alphas = [s["alpha"] for s in scored]

    return len(scored), avg_emo, avg_coh, emotions[0], min(alphas), max(alphas)


def main():
    os.makedirs(SCORED_DIR, exist_ok=True)

    chunk_files = sorted(f for f in os.listdir(CHUNKS_DIR) if f.endswith('.json'))

    total = 0
    print(f"{'Chunk':<20} {'N':>4} {'Avg Emo':>8} {'Avg Coh':>8} {'Emotion':<12} {'Alpha Range'}")
    print("-" * 75)

    for cf in chunk_files:
        n, avg_e, avg_c, emo, amin, amax = process_chunk(cf)
        total += n
        print(f"{cf:<20} {n:>4} {avg_e:>8.2f} {avg_c:>8.2f} {emo:<12} [{amin:.1f}, {amax:.1f}]")

    print("-" * 75)
    print(f"Total entries scored: {total}")
    print(f"Chunks processed: {len(chunk_files)}")


if __name__ == "__main__":
    main()
