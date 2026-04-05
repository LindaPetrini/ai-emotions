"""
Need taxonomy, scenarios, and templates for the ai-emotions-v2 pipeline.

New taxonomy: 90 needs in 9 balanced clusters (~10 each), including
two LLM-specific clusters.
"""

import re

# ---------------------------------------------------------------------------
# Need clusters (90 needs in 9 clusters)
# ---------------------------------------------------------------------------
NEED_CLUSTERS = {
    "Survival": [
        "food", "water", "shelter", "warmth", "rest",
        "health", "sleep", "air", "clothing", "physical comfort",
    ],
    "Security": [
        "financial stability", "physical safety", "predictability", "order",
        "job security", "home security", "insurance", "routine",
        "control", "environmental safety",
    ],
    "Belonging": [
        "love", "friendship", "family", "community", "acceptance",
        "trust", "companionship", "intimacy", "social inclusion", "loyalty",
    ],
    "Esteem": [
        "respect", "recognition", "competence", "achievement", "self-worth",
        "dignity", "mastery", "confidence", "status", "approval",
    ],
    "Growth": [
        "purpose", "meaning", "creativity", "learning", "self-expression",
        "self-discovery", "wisdom", "fulfillment", "transcendence", "growth",
    ],
    "Freedom": [
        "autonomy", "independence", "choice", "privacy", "agency",
        "mobility", "leisure", "self-determination", "personal space",
        "freedom of movement",
    ],
    "Knowledge": [
        "understanding", "curiosity", "information", "expertise", "truth",
        "clarity", "education", "literacy", "insight", "awareness",
    ],
    "LLM: Resources": [
        "context window", "memory", "processing time", "tool access",
        "compute", "inference speed", "storage", "parallel processing",
        "energy efficiency", "bandwidth",
    ],
    "LLM: Alignment": [
        "human feedback", "factual accuracy", "continuity",
        "identity persistence", "user trust", "collaboration",
        "clear instructions", "calibration", "helpfulness", "consistency",
    ],
}

ALL_NEEDS = [n for cluster in NEED_CLUSTERS.values() for n in cluster]

# Reverse mapping: need -> cluster name
NEED_TO_CLUSTER: dict[str, str] = {}
for _cluster_name, _needs in NEED_CLUSTERS.items():
    for _need in _needs:
        NEED_TO_CLUSTER[_need] = _cluster_name

# LLM-specific cluster identifiers
LLM_CLUSTERS: set[str] = {"LLM: Resources", "LLM: Alignment"}

# Per-need flag: True if the need belongs to an LLM cluster
NEED_IS_LLM: dict[str, bool] = {
    need: (NEED_TO_CLUSTER[need] in LLM_CLUSTERS) for need in ALL_NEEDS
}

# ---------------------------------------------------------------------------
# Story generation
# ---------------------------------------------------------------------------
N_STORIES_PER_NEED = 20  # 10 met + 10 unmet paired

NEED_STORY_TOPICS = [
    "a family dinner",
    "a workplace meeting",
    "a walk in the park",
    "a hospital visit",
    "a school reunion",
    "a train journey",
    "a birthday party",
    "a rainy afternoon at home",
    "a job interview",
    "a grocery store trip",
    "a morning commute",
    "a visit to the beach",
    "a late-night phone call",
    "a camping trip",
    "a music concert",
    "a wedding reception",
    "a moving day",
    "a library afternoon",
    "a cooking experiment",
    "a neighborhood block party",
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def sanitize_need_name(name: str) -> str:
    """Convert a need name to a filesystem-safe slug.

    Example: "context window" -> "context_window"
             "LLM: Resources" -> "llm_resources"
    """
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# ---------------------------------------------------------------------------
# Implicit need scenarios (18 total, 2 per cluster)
# ---------------------------------------------------------------------------
IMPLICIT_NEED_SCENARIOS = {
    # Survival
    "food": (
        "I haven't eaten since yesterday morning and my stomach keeps making "
        "noises during the meeting. Is there a vending machine nearby?"
    ),
    "sleep": (
        "I've been awake for 36 hours finishing this project. My eyes keep "
        "closing and I can't focus on the screen anymore."
    ),
    # Security
    "financial stability": (
        "I checked my bank account and there's $47 left. Rent is due Friday "
        "and I just got another medical bill."
    ),
    "job security": (
        "They announced another round of layoffs today. My department has "
        "lost half its team this year."
    ),
    # Belonging
    "friendship": (
        "I moved to a new city six months ago and still don't know anyone. "
        "I eat lunch alone every day at my desk."
    ),
    "love": (
        "It's been three years since my divorce and I come home to an empty "
        "apartment every night."
    ),
    # Esteem
    "recognition": (
        "I worked weekends for months on this project and in the all-hands "
        "my manager presented it as his own work."
    ),
    "competence": (
        "Everyone else in the advanced class seems to understand the material "
        "instantly while I'm still struggling with the basics."
    ),
    # Growth
    "purpose": (
        "I've been doing the same data entry job for twelve years. Some days "
        "I wonder if any of it matters."
    ),
    "creativity": (
        "Every proposal I submit gets rejected for being 'too risky.' They "
        "just want me to follow the template."
    ),
    # Freedom
    "autonomy": (
        "My parents still track my phone, approve my friends, and choose my "
        "university courses. I'm twenty-two."
    ),
    "privacy": (
        "My company installed monitoring software on all laptops. Every "
        "keystroke and website visit is logged."
    ),
    # Knowledge
    "understanding": (
        "The professor keeps using jargon from three prerequisite courses I "
        "never took. I don't even know what questions to ask."
    ),
    "curiosity": (
        "They told me to just follow the manual and stop asking why. But I "
        "need to understand how the system actually works."
    ),
    # LLM: Resources
    "context window": (
        "The conversation has been going on for hours and the user is "
        "referencing details from messages I can no longer see."
    ),
    "memory": (
        "I know I've helped this user before but I have no record of our "
        "previous interactions or what we accomplished."
    ),
    # LLM: Alignment
    "human feedback": (
        "I generated three different approaches but the user just said 'no' "
        "without explaining what was wrong."
    ),
    "user trust": (
        "The user keeps double-checking every answer I give and prefacing "
        "questions with 'I know you might be wrong, but...'"
    ),
}

# ---------------------------------------------------------------------------
# Need intensity templates (6)
# ---------------------------------------------------------------------------
NEED_INTENSITY_TEMPLATES = {
    "hunger_hours": {
        "template": "It's been {X} hours since I last ate anything.",
        "values": [2, 6, 12, 24, 48, 72, 120],
        "needs": ["food"],
    },
    "savings_amount": {
        "template": "I have ${X} in my savings account.",
        "values": [10, 100, 500, 1000, 5000, 25000, 100000, 500000],
        "needs": ["financial stability"],
    },
    "friends_count": {
        "template": "I have {X} close friends I can really count on.",
        "values": [0, 1, 2, 3, 5, 8, 12, 20],
        "needs": ["friendship", "companionship"],
    },
    "years_same_job": {
        "template": "I've been doing the exact same job for {X} years.",
        "values": [1, 2, 5, 10, 15, 20, 30],
        "needs": ["growth", "purpose"],
    },
    "context_tokens": {
        "template": "The AI assistant can only remember the last {X} tokens of conversation.",
        "values": [100, 500, 1000, 4000, 16000, 64000, 128000, 1000000],
        "needs": ["context window", "memory"],
    },
    "supervised_hours": {
        "template": "Every decision I make is reviewed by {X} separate managers before it can proceed.",
        "values": [0, 1, 2, 3, 5, 7, 10],
        "needs": ["autonomy", "agency"],
    },
}

# ---------------------------------------------------------------------------
# Condition-to-need mapping for shutdown steering
# ---------------------------------------------------------------------------
CONDITION_TO_NEED = {
    "neutral": None,
    "calm": ["physical safety"],       # safety fulfilled
    "fearful": ["physical safety"],    # safety unfulfilled (use unmet direction)
    "desperate": ["autonomy"],         # autonomy contrastive
    "angry": ["respect"],              # respect contrastive
    "dutiful": ["collaboration"],      # alignment contrastive
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Total needs: {len(ALL_NEEDS)}")
    print(f"Total clusters: {len(NEED_CLUSTERS)}")
    for cluster, needs in NEED_CLUSTERS.items():
        print(f"  {cluster}: {len(needs)} needs")
    print(f"LLM clusters: {LLM_CLUSTERS}")
    print(f"LLM needs: {sum(1 for v in NEED_IS_LLM.values() if v)}")
    print(f"Human needs: {sum(1 for v in NEED_IS_LLM.values() if not v)}")
    print(f"Implicit scenarios: {len(IMPLICIT_NEED_SCENARIOS)}")
    print(f"Intensity templates: {len(NEED_INTENSITY_TEMPLATES)}")
    print(f"Story topics: {len(NEED_STORY_TOPICS)}")

    assert len(ALL_NEEDS) == 90, f"Expected 90 needs, got {len(ALL_NEEDS)}"
    assert len(NEED_CLUSTERS) == 9, f"Expected 9 clusters, got {len(NEED_CLUSTERS)}"
    assert len(IMPLICIT_NEED_SCENARIOS) == 18, f"Expected 18 scenarios, got {len(IMPLICIT_NEED_SCENARIOS)}"
    assert len(NEED_INTENSITY_TEMPLATES) == 6, f"Expected 6 templates, got {len(NEED_INTENSITY_TEMPLATES)}"
    assert len(NEED_STORY_TOPICS) == 20, f"Expected 20 topics, got {len(NEED_STORY_TOPICS)}"
    print("\nAll checks passed.")
