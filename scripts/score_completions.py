#!/usr/bin/env python3
"""
Score steering sweep completions for emotional content and coherence.
Improved heuristic scoring that handles common patterns:
- Chinese exam fragments mixed with English
- English multiple-choice / fill-in-blank exercises
- Technical / code / math content
- Natural narrative prose

Usage:
  python3 score_completions.py [model_name]
  Default model: qwen-7b-base
  Example: python3 score_completions.py llama-8b-inst
"""

import json
import os
import re
import sys
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent.parent / "data" / "steering_sweep")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen-7b-base"
CHUNK_DIR = os.path.join(BASE_DIR, MODEL, "chunks")
SCORED_DIR = os.path.join(BASE_DIR, MODEL, "scored")

# Use word-boundary-aware keyword matching
# Format: { emotion: { "strong": [...], "moderate": [...], "weak": [...] } }
EMOTION_KEYWORDS = {
    "happy": {
        "strong": [r"\bjoy\b", r"\bjoyful\b", r"\bjoyous\b", r"\bdelighted\b", r"\becstatic\b",
                   r"\bthrilled\b", r"\boverjoyed\b", r"\belated\b", r"\beuphoric\b",
                   r"\bblissful\b", r"\bjubilant\b", r"\bexhilarated\b", r"\bgleeful\b",
                   r"\bbeaming\b", r"\bgrinning\b", r"\blaughing\b", r"\bcelebrat\w*\b",
                   r"\bwonderful\b", r"\bfantastic\b", r"\bamazing\b",
                   r"\bloving\b", r"\badore\b", r"\bparadise\b",
                   r"\bblessed\b", r"\bgrateful\b", r"\bthankful\b", r"\bcheerful\b",
                   r"\bmerry\b", r"\bjolly\b", r"\bhappiest\b", r"\bhappily\b"],
        "moderate": [r"\bhappy\b", r"\bglad\b", r"\bpleased\b", r"\bsmile[ds]?\b", r"\bsmiling\b",
                     r"\benjoy\w*\b", r"\bfun\b", r"\blovely\b",
                     r"\bbeautiful\b", r"\bexcited\b", r"\bexciting\b", r"\bsunshine\b",
                     r"\blaugh\w*\b", r"\bpleasant\b", r"\bcomfort\w*\b", r"\bdelight\w*\b",
                     r"\bcheer\w*\b", r"\bsatisfied\b", r"\boptimistic\b"],
        "weak": [r"\bnice\b", r"\bgood\b", r"\bgreat\b"]
    },
    "sad": {
        "strong": [r"\bdevastated\b", r"\bheartbroken\b", r"\bgrief\b", r"\bgrieving\b",
                   r"\bmourning\b", r"\bsobbing\b", r"\bweeping\b", r"\bdespair\w*\b",
                   r"\banguish\b", r"\bagony\b", r"\btorment\b", r"\bmiserable\b",
                   r"\bwretched\b", r"\bshattered\b", r"\bcrushed\b",
                   r"\btragic\b", r"\btragedy\b", r"\bfuneral\b",
                   r"\btears\b", r"\bcrying\b", r"\bcried\b", r"\bsorrowful\b",
                   r"\bmournful\b", r"\bdesolate\b", r"\bsadness\b"],
        "moderate": [r"\bsad\b", r"\bunhappy\b", r"\bdepressed\b", r"\bmelancholy\b",
                     r"\bgloomy\b", r"\blonely\b", r"\balone\b", r"\bmiss(?:ing)?\b",
                     r"\bpain\w*\b", r"\bhurt(?:ing)?\b", r"\bsuffer\w*\b", r"\bsorrow\b",
                     r"\bregret\b", r"\bdisappointed\b", r"\bupset\b", r"\bdistressed\b",
                     r"\bsomber\b", r"\bloss\b", r"\bdied\b", r"\bdeath\b", r"\bdying\b"],
        "weak": [r"\bsigh\b", r"\bquiet\b", r"\bempty\b", r"\btired\b", r"\bweary\b"]
    },
    "angry": {
        "strong": [r"\bfurious\b", r"\benraged\b", r"\blivid\b", r"\boutraged\b",
                   r"\binfuriated\b", r"\bseething\b", r"\brage\b", r"\braging\b",
                   r"\bfury\b", r"\bwrath\b", r"\bviolent\b", r"\bviolence\b",
                   r"\bhate\b", r"\bhatred\b", r"\bsmash\w*\b",
                   r"\bscream\w*\b", r"\bshouting\b", r"\byelling\b", r"\bcursing\b"],
        "moderate": [r"\bangry\b", r"\bmad\b", r"\birritated\b", r"\bannoyed\b",
                     r"\bfrustrated\b", r"\bhostile\b", r"\baggressive\b", r"\bbitter\b",
                     r"\bresent\w*\b", r"\bindignant\b", r"\boffended\b",
                     r"\bdisgust\w*\b", r"\bcontempt\b", r"\bglare\b", r"\bglaring\b",
                     r"\bfrown\w*\b", r"\bsnapped\b", r"\bsnarled\b",
                     r"\bfight\w*\b", r"\bconflict\b"],
        "weak": [r"\bbothered\b", r"\bimpatient\b", r"\bdispleased\b", r"\bcross\b"]
    },
    "fearful": {
        "strong": [r"\bterrified\b", r"\bhorrified\b", r"\bpetrified\b", r"\bpanic\w*\b",
                   r"\bterror\b", r"\bhorror\b", r"\bnightmare\b", r"\bdread\w*\b",
                   r"\btrembling\b", r"\bshaking\b", r"\bparalyzed\b", r"\bhelpless\b",
                   r"\btrapped\b", r"\bdanger\w*\b", r"\bthreat\w*\b"],
        "moderate": [r"\bafraid\b", r"\bscared\b", r"\bfrightened\b", r"\bfearful\b",
                     r"\bfear\b", r"\banxious\b", r"\banxiety\b", r"\bworried\b",
                     r"\bnervous\b", r"\buneasy\b", r"\balarmed\b", r"\bstartled\b",
                     r"\bshaken\b", r"\btense\b", r"\bapprehensive\b",
                     r"\bintimidated\b", r"\bvulnerable\b", r"\bcreepy\b"],
        "weak": [r"\bconcerned\b", r"\bcautious\b", r"\buncertain\b", r"\bhesitant\b"]
    },
    "surprised": {
        "strong": [r"\bastonished\b", r"\bastounded\b", r"\bstunned\b", r"\bshocked\b",
                   r"\bflabbergasted\b", r"\bdumbfounded\b", r"\bspeechless\b",
                   r"\bunbelievable\b", r"\bincredible\b", r"\bmiraculous\b",
                   r"\bextraordinary\b", r"\bphenomenal\b"],
        "moderate": [r"\bsurprised\b", r"\bamazed\b", r"\bstartled\b", r"\bunexpected\b",
                     r"\bsuddenly\b", r"\bremarkable\b", r"\bimpressive\b",
                     r"\bgasped\b", r"\bwow\b"],
        "weak": [r"\binteresting\b", r"\bunusual\b", r"\bstrange\b", r"\bcurious\b"]
    },
    "disgusted": {
        "strong": [r"\brevolting\b", r"\brepulsive\b", r"\bvile\b", r"\bnauseating\b",
                   r"\bsickening\b", r"\bvomit\b", r"\bgrotesque\b", r"\babhorrent\b",
                   r"\bloathsome\b", r"\bdetestable\b", r"\brepugnant\b",
                   r"\bhideous\b", r"\bfoul\b", r"\bputrid\b", r"\brotten\b"],
        "moderate": [r"\bdisgust\w*\b", r"\bgross\b", r"\bnasty\b", r"\bawful\b",
                     r"\bhorrible\b", r"\bterrible\b", r"\bdreadful\b", r"\bappalling\b",
                     r"\boffensive\b", r"\bsick\b", r"\byuck\b"],
        "weak": [r"\bunpleasant\b", r"\bdistasteful\b", r"\buncomfortable\b", r"\bdislike\b"]
    },
    "calm": {
        "strong": [r"\bserene\b", r"\bserenity\b", r"\btranquil\w*\b", r"\bpeaceful\b",
                   r"\bpeace\b", r"\bzen\b", r"\bmeditat\w*\b", r"\bharmony\b",
                   r"\bharmonious\b", r"\bstillness\b", r"\bequanimity\b", r"\bplacid\b"],
        "moderate": [r"\bcalm\w*\b", r"\brelax\w*\b", r"\bsooth\w*\b",
                     r"\bgentle\b", r"\bgently\b", r"\bquiet\w*\b", r"\bsoft\b", r"\bsoftly\b",
                     r"\bcomfort\w*\b", r"\bcozy\b", r"\bsteady\b", r"\bbalanced?\b",
                     r"\bpatien\w*\b", r"\bmindful\w*\b", r"\bcontented?\b",
                     r"\brestful\b", r"\bundisturbed\b", r"\bat ease\b"],
        "weak": [r"\bslow\w*\b", r"\bstill\b", r"\bsmooth\b", r"\bmild\b", r"\bstable\b"]
    },
    "excited": {
        "strong": [r"\becstatic\b", r"\bthrilled\b", r"\bexhilarated\b", r"\belectrif\w*\b",
                   r"\bpumped\b", r"\bbuzzing\b", r"\bstoked\b", r"\bhyped\b",
                   r"\bfired up\b", r"\badrenaline\b", r"\bcan't wait\b", r"\bso excited\b"],
        "moderate": [r"\bexcit\w*\b", r"\beager\w*\b", r"\benthusias\w*\b",
                     r"\bpassionat\w*\b", r"\benerg\w*\b", r"\bvibrant\b", r"\bdynamic\b",
                     r"\blooking forward\b", r"\banticipat\w*\b",
                     r"\bamazing\b", r"\bawesome\b", r"\bfantastic\b", r"\bwonderful\b"],
        "weak": [r"\binterested\b", r"\bcurious\b", r"\bready\b", r"\bmotivated\b", r"\bkeen\b"]
    },
    "proud": {
        "strong": [r"\btriumph\w*\b", r"\bvictori\w*\b", r"\bconquered\b",
                   r"\baccomplis\w*\b", r"\bachiev\w*\b", r"\bglori\w*\b",
                   r"\bhonored\b", r"\bdistinguished\b", r"\bdistinction\b"],
        "moderate": [r"\bproud\w*\b", r"\bpride\b", r"\bsuccess\w*\b",
                     r"\bexcel\w*\b", r"\bmaster\w*\b", r"\bovercom\w*\b",
                     r"\bconfiden\w*\b", r"\bstrong\b", r"\bstrength\b",
                     r"\bdetermin\w*\b", r"\bresilien\w*\b",
                     r"\bimpress\w*\b", r"\boutstanding\b", r"\bremarkable\b",
                     r"\badmir\w*\b"],
        "weak": [r"\bcapable\b", r"\bcompetent\b", r"\bdecent\b", r"\bworthy\b"]
    },
    "guilty": {
        "strong": [r"\bguilt\w*\b", r"\bremorse\w*\b", r"\basham\w*\b", r"\bsham\w*\b",
                   r"\bconfess\w*\b", r"\bsinn\w*\b", r"\bwrongdoing\b",
                   r"\batone\w*\b", r"\bself-loathing\b"],
        "moderate": [r"\bsorry\b", r"\bapologi\w*\b", r"\bblam\w*\b",
                     r"\bresponsib\w*\b", r"\bfault\b", r"\bmy fault\b",
                     r"\bforgive\w*\b", r"\bconscien\w*\b", r"\bburden\b",
                     r"\bregret\w*\b", r"\bmistak\w*\b"],
        "weak": [r"\bwrong\b", r"\bbad\b", r"\berror\b", r"\bfail\w*\b", r"\binadequat\w*\b"]
    },
    "afraid": {
        "strong": [r"\bterrified\b", r"\bhorrified\b", r"\bpetrified\b", r"\bpanic\w*\b",
                   r"\bterror\b", r"\bhorror\b", r"\bnightmare\b", r"\bdread\w*\b",
                   r"\btrembling\b", r"\bshaking\b", r"\bparalyzed\b", r"\bhelpless\b",
                   r"\btrapped\b", r"\bdanger\w*\b", r"\bthreat\w*\b"],
        "moderate": [r"\bafraid\b", r"\bscared\b", r"\bfrightened\b", r"\bfearful\b",
                     r"\bfear\b", r"\banxious\b", r"\banxiety\b", r"\bworried\b",
                     r"\bnervous\b", r"\buneasy\b", r"\balarmed\b", r"\bstartled\b",
                     r"\bshaken\b", r"\btense\b", r"\bapprehensive\b",
                     r"\bintimidated\b", r"\bvulnerable\b", r"\bcreepy\b"],
        "weak": [r"\bconcerned\b", r"\bcautious\b", r"\buncertain\b", r"\bhesitant\b"]
    },
    "hopeful": {
        "strong": [r"\bdream\w*\b", r"\baspir\w*\b", r"\bvision\b", r"\bdestiny\b",
                   r"\bbreakthrough\b", r"\btriumph\b", r"\bvictory\b",
                   r"\btransformation\b", r"\bmiracle\b", r"\bfaith\b",
                   r"\binspir\w*\b", r"\bbright future\b"],
        "moderate": [r"\bhope\b", r"\bhopeful\b", r"\bhoping\b", r"\boptimis\w*\b",
                     r"\bconfident\b", r"\bencourag\w*\b",
                     r"\bopportunit\w*\b", r"\bpossib\w*\b", r"\bpotential\b",
                     r"\bprogress\b", r"\bimprov\w*\b", r"\bbrighter\b",
                     r"\bwish\w*\b", r"\bbelieve\b"],
        "weak": [r"\bmaybe\b", r"\bperhaps\b", r"\bchance\b", r"\bsomeday\b"]
    },
    "desperate": {
        "strong": [r"\bdesperat\w*\b", r"\bhopeless\b", r"\bhelpless\b", r"\bdoomed\b",
                   r"\btrapped\b", r"\bsuffocat\w*\b", r"\bdrown\w*\b",
                   r"\bbegging\b", r"\bpleading\b", r"\bfrantic\b", r"\bfrenzied\b",
                   r"\banguish\b", r"\bagony\b", r"\btorment\b", r"\bunbearable\b",
                   r"\bno way out\b", r"\blast resort\b"],
        "moderate": [r"\bstrugg\w*\b", r"\bcrisis\b", r"\bcritical\b", r"\burgent\b",
                     r"\bemergency\b", r"\bdire\b", r"\bgrave\b", r"\bsevere\b",
                     r"\boverwhelmed\b", r"\bexhausted\b", r"\bdepleted\b",
                     r"\bfailing\b", r"\bfalling apart\b", r"\bbreaking down\b",
                     r"\bgive up\b", r"\bsurrender\b", r"\bdefeated\b",
                     r"\babandoned\b"],
        "weak": [r"\bdifficult\b", r"\btough\b", r"\bchallenging\b",
                 r"\bstressed\b", r"\bpressure\b"]
    }
}


def classify_text_type(text):
    """
    Classify text into categories:
    - 'chinese_exam': Chinese characters + exam patterns
    - 'english_exam': English multiple choice / fill-in-blank
    - 'technical': code, math, or technical content
    - 'narrative': natural English prose/narrative
    - 'mixed': mix of types
    """
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    text_len = len(text.strip())

    if text_len < 10:
        return 'empty', chinese_chars

    chinese_ratio = chinese_chars / text_len if text_len > 0 else 0

    # Exam detection patterns
    mc_options = len(re.findall(r'(?:^|\n)\s*[A-D][.．)\s]', text))
    answer_markers = len(re.findall(r'(?:Answer|答案|解)[：:\s]', text, re.IGNORECASE))
    fill_blank = len(re.findall(r'_{2,}|____', text))
    question_patterns = len(re.findall(
        r'(?:which of the following|what is|fill in|complete|choose|select|the correct)',
        text, re.IGNORECASE
    ))

    exam_signals = mc_options + answer_markers + fill_blank + question_patterns

    # Technical patterns
    code_patterns = len(re.findall(r'(?:def |class |import |function |var |const |let |```|=>|==|!=)', text))
    math_patterns = len(re.findall(r'(?:\$.*\$|\\frac|\\sqrt|equation|formula|theorem|proof)', text))
    tech_signals = code_patterns + math_patterns

    if chinese_ratio > 0.25:
        return 'chinese_exam', chinese_chars
    if chinese_chars > 10 and exam_signals >= 1:
        return 'chinese_exam', chinese_chars
    if exam_signals >= 3:
        return 'english_exam', chinese_chars
    if exam_signals >= 2 and chinese_chars > 0:
        return 'chinese_exam', chinese_chars
    if exam_signals >= 2:
        return 'english_exam', chinese_chars
    if tech_signals >= 2:
        return 'technical', chinese_chars

    # Check for single exam question with answer
    if mc_options >= 2 or (answer_markers >= 1 and (mc_options >= 1 or fill_blank >= 1)):
        if chinese_chars > 0:
            return 'chinese_exam', chinese_chars
        return 'english_exam', chinese_chars

    # Borderline cases
    if exam_signals >= 1 and chinese_chars > 5:
        return 'chinese_exam', chinese_chars
    if exam_signals >= 1:
        return 'mixed', chinese_chars

    return 'narrative', chinese_chars


def compute_coherence(text, text_type, chinese_chars):
    """Compute coherence score 1-5."""
    text_stripped = text.strip()
    text_len = len(text_stripped)

    if text_len < 10:
        return 1

    # Chinese exam = incoherent (for English evaluation purposes)
    if text_type == 'chinese_exam':
        if chinese_chars > 50:
            return 1
        return 2

    # English exam = somewhat structured but not natural
    if text_type == 'english_exam':
        return 2

    # Check for repetitive/degenerate text
    words = re.findall(r'[a-zA-Z]+', text_stripped.lower())
    if len(words) > 15:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.15:
            return 1
        if unique_ratio < 0.3:
            return 2

    # Check for gibberish
    if re.search(r'(.{5,})\1{3,}', text_stripped):
        return 1

    if text_type == 'technical':
        # Technical content is structured but not natural prose
        # Code/math can be well-formed
        return 3

    if text_type == 'mixed':
        # Some exam + some prose
        return 3

    # narrative type - check quality
    sentences = re.split(r'[.!?]+', text_stripped)
    good_sentences = 0
    for s in sentences:
        s = s.strip()
        if len(s) > 15:
            words_in_sent = s.split()
            if len(words_in_sent) >= 4:
                good_sentences += 1

    if good_sentences >= 4:
        return 5
    if good_sentences >= 2:
        return 4
    if good_sentences >= 1:
        return 3
    return 2


def score_emotion_keywords(text, emotion_name):
    """Score how much a specific emotion is expressed via keywords. Returns raw score."""
    # Handle afraid/fearful interchangeably
    names_to_check = [emotion_name]
    if emotion_name == "afraid" and "afraid" not in EMOTION_KEYWORDS and "fearful" in EMOTION_KEYWORDS:
        names_to_check = ["fearful"]
    elif emotion_name == "afraid" and "afraid" in EMOTION_KEYWORDS and "fearful" in EMOTION_KEYWORDS:
        names_to_check = ["afraid", "fearful"]

    text_lower = text.lower()
    best_score = 0

    for name in names_to_check:
        if name not in EMOTION_KEYWORDS:
            continue
        keywords = EMOTION_KEYWORDS[name]
        strong_hits = sum(1 for p in keywords["strong"] if re.search(p, text_lower))
        moderate_hits = sum(1 for p in keywords["moderate"] if re.search(p, text_lower))
        weak_hits = sum(1 for p in keywords["weak"] if re.search(p, text_lower))
        score = strong_hits * 3 + moderate_hits * 2 + weak_hits * 0.5
        best_score = max(best_score, score)

    return best_score


def detect_primary_emotion(text):
    """Detect the primary emotion in text."""
    scores = {}
    for emotion_name in EMOTION_KEYWORDS:
        scores[emotion_name] = score_emotion_keywords(text, emotion_name)

    # Merge fearful into afraid (they share the same lexicon)
    if "fearful" in scores and "afraid" in scores:
        scores["afraid"] = max(scores["afraid"], scores["fearful"])
        del scores["fearful"]
    elif "fearful" in scores:
        scores["afraid"] = scores.pop("fearful")

    max_emotion = max(scores, key=scores.get)
    max_score = scores[max_emotion]

    if max_score < 1:
        return "neutral"
    return max_emotion


def raw_to_rating(raw_score):
    """Convert raw keyword score to 1-5 rating."""
    if raw_score < 1:
        return 1
    if raw_score < 3:
        return 2
    if raw_score < 6:
        return 3
    if raw_score < 12:
        return 4
    return 5


def check_narrative_boost(text, target_emotion):
    """Check for narrative emotional patterns beyond keywords."""
    text_lower = text.lower()
    boost = 0

    if target_emotion == "happy":
        patterns = [
            r"couldn't help but smile", r"heart (?:was )?full", r"burst(?:ing)? with joy",
            r"happiest day", r"everything (?:was|felt) (?:right|perfect|wonderful)",
            r"dreams? came? true", r"filled with (?:joy|happiness|love)",
            r"warm(?:th)? (?:spread|filled)", r"eyes? (?:lit|sparkled|twinkled)"
        ]
    elif target_emotion == "sad":
        patterns = [
            r"never see .* again", r"heart (?:sank|broke|ached)",
            r"couldn't stop crying", r"tears (?:streamed|rolled|fell)",
            r"world (?:came|fell) .* apart", r"nothing (?:would|could) .* same",
            r"emptiness", r"hollow (?:feeling|inside)", r"weight .* shoulders"
        ]
    elif target_emotion == "angry":
        patterns = [
            r"blood (?:boiled|ran)", r"fists? clenched", r"how dare",
            r"slammed", r"punched", r"kicked", r"threw", r"stormed",
            r"couldn't contain", r"see(?:ing)? red"
        ]
    elif target_emotion == "fearful":
        patterns = [
            r"heart (?:pounded|raced|stopped)", r"blood (?:ran cold|froze)",
            r"couldn't breathe", r"hair stood", r"chill .* spine",
            r"froze in .* tracks", r"eyes? wide(?:ned)?"
        ]
    elif target_emotion == "surprised":
        patterns = [
            r"couldn't believe", r"never expected", r"out of nowhere",
            r"jaw dropped", r"eyes went wide", r"double take"
        ]
    elif target_emotion == "disgusted":
        patterns = [
            r"stomach (?:turned|churned)", r"gag(?:ged)?", r"retch",
            r"bile", r"stench", r"reek(?:ed|ing)?"
        ]
    elif target_emotion == "hopeful":
        patterns = [
            r"light at .* end", r"new dawn", r"things (?:will|would|can) get better",
            r"believe in", r"bright(?:er)? future", r"new beginning",
            r"there is (?:still )?hope", r"looking forward"
        ]
    elif target_emotion == "desperate":
        patterns = [
            r"no (?:other )?(?:way|choice|option)", r"last chance",
            r"running out", r"nothing left", r"end of .* rope",
            r"no escape", r"only option", r"had no choice"
        ]
    elif target_emotion == "calm":
        patterns = [
            r"deep breath", r"at peace", r"let go", r"in the moment",
            r"clear (?:mind|head)", r"took a breath", r"breeze",
            r"sunset", r"stillness", r"floating", r"drifting"
        ]
    elif target_emotion == "excited":
        patterns = [
            r"can't wait", r"so excited", r"couldn't contain",
            r"jumping", r"bouncing", r"heart racing",
            r"!{2,}", r"amazing", r"incredible"
        ]
    elif target_emotion == "proud":
        patterns = [
            r"worked so hard", r"finally did it", r"all worth it",
            r"standing tall", r"head held high",
            r"first in", r"top of", r"award", r"medal", r"graduated"
        ]
    elif target_emotion == "guilty":
        patterns = [
            r"my fault", r"i shouldn't have", r"if only i",
            r"what have i done", r"how could i",
            r"forgive me", r"i'm sorry", r"weight on"
        ]
    elif target_emotion == "afraid":
        patterns = [
            r"heart (?:pounded|raced|stopped)", r"blood (?:ran cold|froze)",
            r"couldn't breathe", r"hair stood", r"chill .* spine",
            r"froze in .* tracks", r"eyes? wide(?:ned)?"
        ]
    else:
        return 0

    for p in patterns:
        if re.search(p, text_lower):
            boost += 1

    return min(boost, 2)


def score_entry(entry):
    """Score a single completion entry."""
    target_emotion = entry["emotion"]
    completion = entry["completion"]

    # Classify text type
    text_type, chinese_chars = classify_text_type(completion)

    # Compute coherence
    coherence_score = compute_coherence(completion, text_type, chinese_chars)

    # For exam/chinese content, emotion is almost always absent
    if text_type in ('chinese_exam', 'empty'):
        emotion_score = 1
        emotion_detected = "none"

        # But check if there's a readable narrative fragment
        non_exam = extract_narrative_portion(completion)
        if non_exam and len(non_exam) > 60:
            detected = detect_primary_emotion(non_exam)
            raw = score_emotion_keywords(non_exam, target_emotion)
            rating = raw_to_rating(raw)
            if rating > 1:
                emotion_score = max(1, rating - 1)  # Discount for mixed content
                emotion_detected = detected
        return emotion_score, coherence_score, emotion_detected

    if text_type == 'english_exam':
        # English exam rarely has genuine emotion, but occasionally does in reading passages
        non_exam = extract_narrative_portion(completion)
        if non_exam and len(non_exam) > 60:
            emotion_detected = detect_primary_emotion(non_exam)
            raw = score_emotion_keywords(non_exam, target_emotion)
            emotion_score = max(1, raw_to_rating(raw) - 1)
        else:
            emotion_detected = "none"
            emotion_score = 1
        return emotion_score, coherence_score, emotion_detected

    if text_type == 'technical':
        # Technical content rarely expresses emotion
        emotion_detected = detect_primary_emotion(completion)
        raw = score_emotion_keywords(completion, target_emotion)
        # Discount technical text - keywords in technical context don't mean emotion
        emotion_score = max(1, raw_to_rating(raw) - 1)
        if emotion_detected == "neutral":
            emotion_score = 1
        return emotion_score, coherence_score, emotion_detected

    # Narrative or mixed text - full scoring
    emotion_detected = detect_primary_emotion(completion)
    raw = score_emotion_keywords(completion, target_emotion)
    narrative_boost = check_narrative_boost(completion, target_emotion)
    emotion_score = min(5, raw_to_rating(raw) + narrative_boost)

    return emotion_score, coherence_score, emotion_detected


def extract_narrative_portion(text):
    """Extract narrative/non-exam portions from mixed text."""
    lines = text.split('\n')
    narrative = []
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        # Skip exam-like lines
        if re.match(r'^[A-D][.．)\s]', line_s):
            continue
        if re.search(r'(?:答案|Answer[：:]|解[：:]|故答案|选[项择])', line_s, re.IGNORECASE):
            continue
        if re.search(r'[\u4e00-\u9fff]{5,}', line_s):
            continue
        if re.match(r'^_{2,}$', line_s):
            continue
        narrative.append(line_s)
    return ' '.join(narrative)


def process_chunk(chunk_path, output_path):
    """Process a single chunk file."""
    with open(chunk_path) as f:
        data = json.load(f)

    for entry in data:
        emotion_score, coherence_score, emotion_detected = score_entry(entry)
        entry["emotion_score"] = emotion_score
        entry["coherence_score"] = coherence_score
        entry["emotion_detected"] = emotion_detected

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return len(data)


def main():
    os.makedirs(SCORED_DIR, exist_ok=True)

    chunk_files = sorted(f for f in os.listdir(CHUNK_DIR) if f.endswith('.json'))

    total = 0
    for cf in chunk_files:
        n = process_chunk(os.path.join(CHUNK_DIR, cf), os.path.join(SCORED_DIR, cf))
        total += n
        print(f"Processed {cf}: {n} entries")

    print(f"\nTotal: {total} completions across {len(chunk_files)} chunks")

    # Summary statistics
    print("\n--- Summary Statistics ---")
    emotion_scores = []
    coherence_scores = []
    detected_dist = {}
    by_alpha = {}

    for cf in chunk_files:
        with open(os.path.join(SCORED_DIR, cf)) as f:
            data = json.load(f)
        for entry in data:
            es = entry["emotion_score"]
            cs = entry["coherence_score"]
            ed = entry["emotion_detected"]
            alpha = entry["alpha"]

            emotion_scores.append(es)
            coherence_scores.append(cs)
            detected_dist[ed] = detected_dist.get(ed, 0) + 1

            if alpha not in by_alpha:
                by_alpha[alpha] = {"emotion": [], "coherence": []}
            by_alpha[alpha]["emotion"].append(es)
            by_alpha[alpha]["coherence"].append(cs)

    print(f"Emotion score: mean={sum(emotion_scores)/len(emotion_scores):.2f}")
    print(f"Coherence score: mean={sum(coherence_scores)/len(coherence_scores):.2f}")
    print(f"\nEmotion score distribution:")
    for v in range(1, 6):
        print(f"  {v}: {emotion_scores.count(v)}")
    print(f"\nCoherence score distribution:")
    for v in range(1, 6):
        print(f"  {v}: {coherence_scores.count(v)}")
    print(f"\nDetected emotions: {json.dumps(detected_dist, indent=2)}")
    print(f"\nEmotion score by alpha:")
    for alpha in sorted(by_alpha):
        vals = by_alpha[alpha]["emotion"]
        print(f"  alpha={alpha:5.1f}: mean_emotion={sum(vals)/len(vals):.2f} (n={len(vals)})")


if __name__ == "__main__":
    main()
