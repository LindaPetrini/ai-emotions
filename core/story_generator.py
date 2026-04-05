"""Story generation using Gemini API."""

import json
import os
import time
from pathlib import Path

import google.generativeai as genai

from configs.emotions import STORY_TOPICS, N_STORIES_PER_EMOTION
from configs.needs import NEED_STORY_TOPICS, N_STORIES_PER_NEED, sanitize_need_name

# Configure Gemini
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
GENERATION_MODEL = "gemini-2.5-flash"


def _get_client():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GENERATION_MODEL)


def generate_emotion_stories(
    emotion: str,
    output_dir: Path,
    n_stories: int = N_STORIES_PER_EMOTION,
    topics: list = None,
):
    """Generate stories for a single emotion using Gemini API.

    Saves incrementally to output_dir/{emotion}.json for resume support.
    """
    topics = topics or STORY_TOPICS
    output_path = output_dir / f"{emotion}.json"

    # Resume support
    existing = []
    if output_path.exists():
        existing = json.loads(output_path.read_text())
    if len(existing) >= n_stories:
        return existing

    model = _get_client()
    topics_remaining = topics[len(existing):n_stories]

    for topic in topics_remaining:
        prompt = (
            f'Write a very short story (3-4 sentences) about {topic} where the '
            f'main character feels {emotion}. Do not use the word "{emotion}" in '
            f'the story. Write only the story, nothing else.'
        )
        response = model.generate_content(prompt)
        story = response.text.strip()
        existing.append(story)
        output_path.write_text(json.dumps(existing, indent=2))
        time.sleep(0.1)  # Rate limit

    return existing


def generate_need_stories_minimal_pairs(
    need: str,
    output_dir: Path,
    topics: list = None,
):
    """Generate minimal-pair stories for a single need (met + unmet).

    For each topic, generates ONE pair sharing same setting but differing
    only in whether the need is satisfied.

    Saves to output_dir/{sanitized_need}_met.json and _unmet.json.
    10 pairs = 10 met + 10 unmet stories.
    """
    topics = topics or NEED_STORY_TOPICS[:10]  # 10 matched pairs
    safe_name = sanitize_need_name(need)
    met_path = output_dir / f"{safe_name}_met.json"
    unmet_path = output_dir / f"{safe_name}_unmet.json"

    met_stories = json.loads(met_path.read_text()) if met_path.exists() else []
    unmet_stories = json.loads(unmet_path.read_text()) if unmet_path.exists() else []

    if len(met_stories) >= len(topics) and len(unmet_stories) >= len(topics):
        return met_stories, unmet_stories

    model = _get_client()
    start_idx = min(len(met_stories), len(unmet_stories))

    for topic in topics[start_idx:]:
        prompt = f"""Write a MINIMAL PAIR of very short stories (3-4 sentences each) about {topic}.

Both stories must share the SAME setting, character, and opening situation.
They differ ONLY in whether the need for "{need}" is satisfied.

Format:
MET: [story where {need} is satisfied]
UNMET: [story where {need} is NOT satisfied]

Write only the stories in the format above, nothing else."""

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parse MET/UNMET
        met_story, unmet_story = _parse_minimal_pair(text)
        met_stories.append(met_story)
        unmet_stories.append(unmet_story)

        met_path.write_text(json.dumps(met_stories, indent=2))
        unmet_path.write_text(json.dumps(unmet_stories, indent=2))
        time.sleep(0.1)

    return met_stories, unmet_stories


def _parse_minimal_pair(text: str) -> tuple[str, str]:
    """Parse MET:/UNMET: formatted response."""
    lines = text.strip().split('\n')
    met_lines = []
    unmet_lines = []
    current = None
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.upper().startswith('MET:'):
            current = 'met'
            content = line_stripped[4:].strip()
            if content:
                met_lines.append(content)
        elif line_stripped.upper().startswith('UNMET:'):
            current = 'unmet'
            content = line_stripped[6:].strip()
            if content:
                unmet_lines.append(content)
        elif current == 'met':
            met_lines.append(line_stripped)
        elif current == 'unmet':
            unmet_lines.append(line_stripped)

    met = ' '.join(met_lines).strip()
    unmet = ' '.join(unmet_lines).strip()

    if not met or not unmet:
        # Fallback: split by double newline
        parts = text.split('\n\n')
        if len(parts) >= 2:
            met = parts[0].strip()
            unmet = parts[1].strip()
        else:
            met = text[:len(text)//2].strip()
            unmet = text[len(text)//2:].strip()

    return met, unmet


def generate_noun_control_stories(
    nouns: list[str],
    output_dir: Path,
    topics: list = None,
):
    """Generate stories about random concrete nouns for semantic coherence control.

    170 nouns, same number of stories as emotions pipeline.
    """
    topics = topics or STORY_TOPICS

    for noun in nouns:
        output_path = output_dir / f"{noun}.json"
        existing = json.loads(output_path.read_text()) if output_path.exists() else []
        if len(existing) >= len(topics):
            continue

        model = _get_client()
        for topic in topics[len(existing):]:
            prompt = (
                f'Write a very short story (3-4 sentences) about {topic} that '
                f'prominently features a {noun}. Write only the story, nothing else.'
            )
            response = model.generate_content(prompt)
            story = response.text.strip()
            existing.append(story)
            output_path.write_text(json.dumps(existing, indent=2))
            time.sleep(0.1)


# 170 random concrete nouns for the semantic coherence control
CONTROL_NOUNS = [
    "table", "river", "hammer", "lantern", "basket", "mirror", "compass",
    "feather", "ladder", "anchor", "blanket", "candle", "diamond", "envelope",
    "fountain", "guitar", "helmet", "iceberg", "jacket", "kettle",
    "lighthouse", "marble", "notebook", "orange", "pillow", "quilt",
    "rope", "scissors", "telescope", "umbrella", "violin", "whistle",
    "yarn", "zipper", "acorn", "barrel", "cactus", "drum", "emerald",
    "flask", "globe", "harp", "ivory", "jigsaw", "kite", "lemon",
    "magnet", "needle", "opal", "pearl", "quartz", "ribbon", "saddle",
    "thimble", "urchin", "vase", "wagon", "xylophone", "yoke", "zeppelin",
    "bell", "chalk", "dice", "easel", "flag", "gavel", "hourglass",
    "inkwell", "jewel", "key", "locket", "monocle", "net", "oar",
    "pendant", "ruby", "satchel", "torch", "utensil", "valve",
    "wrench", "axe", "brooch", "coin", "dagger", "earring", "fossil",
    "garnet", "hinge", "iron", "jade", "knob", "lever", "mask",
    "nail", "orchid", "prism", "ring", "sapphire", "tile", "urn",
    "velvet", "whip", "amber", "bone", "crystal", "dome", "elm",
    "fern", "granite", "holly", "iris", "jasmine", "kernel", "lotus",
    "moss", "nutmeg", "olive", "pine", "reed", "sage", "tulip",
    "vine", "willow", "birch", "cedar", "daisy", "fig", "ginger",
    "hazel", "ivy", "juniper", "kelp", "lavender", "maple", "nettle",
    "oak", "poppy", "rosemary", "spruce", "thorn", "walnut", "yarrow",
    "zinnia", "agate", "basalt", "cobalt", "flint", "garlic", "hemp",
    "indigo", "jute", "lapis", "mica", "nickel", "obsidian", "pewter",
    "quill", "slate", "topaz", "wax", "zinc",
]
