"""
Noun clusters for the semantic coherence control.

162 concrete nouns (from core/story_generator.py) grouped into 10 semantic
categories, mirroring the structure of EMOTION_CLUSTERS in configs/emotions.py.

Note: the source comment says "170" but the actual list contains 162 unique nouns.
We use 2 clusters of 17 and 8 clusters of 16 (2*17 + 8*16 = 162).
"""

# ---------------------------------------------------------------------------
# Noun clusters (162 nouns in 10 clusters)
# ---------------------------------------------------------------------------
NOUN_CLUSTERS: dict[str, list[str]] = {
    "Trees & Woody Plants": [
        "acorn", "birch", "cedar", "elm", "fig", "hazel", "holly", "ivy",
        "juniper", "maple", "oak", "olive", "pine", "spruce", "thorn",
        "walnut", "willow",
    ],  # 17
    "Flowers & Ground Cover": [
        "cactus", "daisy", "fern", "ginger", "iris", "jasmine", "kelp",
        "lavender", "lotus", "moss", "nettle", "orchid", "poppy", "reed",
        "rosemary", "sage", "tulip",
    ],  # 17
    "Botanical & Textile Products": [
        "blanket", "garlic", "hemp", "indigo", "jute", "kernel", "lemon",
        "nutmeg", "orange", "quilt", "ribbon", "velvet", "vine", "yarn",
        "yarrow", "zinnia",
    ],  # 16
    "Gems & Precious Stones": [
        "agate", "amber", "cobalt", "crystal", "diamond", "emerald",
        "garnet", "jade", "jewel", "lapis", "opal", "pearl", "quartz",
        "ruby", "sapphire", "topaz",
    ],  # 16
    "Rocks & Minerals": [
        "basalt", "bone", "chalk", "flint", "fossil", "granite", "iron",
        "marble", "mica", "nickel", "obsidian", "pewter", "slate", "tile",
        "wax", "zinc",
    ],  # 16
    "Tools & Hardware": [
        "axe", "gavel", "hammer", "hinge", "key", "knob", "lever",
        "magnet", "nail", "needle", "scissors", "thimble", "valve",
        "wrench", "yoke", "zipper",
    ],  # 16
    "Jewelry & Personal Accessories": [
        "brooch", "coin", "earring", "feather", "helmet", "ivory",
        "jacket", "locket", "mask", "monocle", "pendant", "ring",
        "saddle", "satchel", "umbrella", "whip",
    ],  # 16
    "Musical Instruments & Optics": [
        "bell", "compass", "dice", "drum", "globe", "guitar", "harp",
        "hourglass", "jigsaw", "kite", "mirror", "prism", "telescope",
        "violin", "whistle", "xylophone",
    ],  # 16
    "Containers & Household Objects": [
        "barrel", "basket", "candle", "envelope", "flask", "inkwell",
        "kettle", "lantern", "notebook", "pillow", "quill", "table",
        "torch", "urn", "utensil", "vase",
    ],  # 16
    "Navigation & Landscape": [
        "anchor", "dagger", "dome", "easel", "flag", "fountain",
        "iceberg", "ladder", "lighthouse", "net", "oar", "river",
        "rope", "urchin", "wagon", "zeppelin",
    ],  # 16
}

# ---------------------------------------------------------------------------
# Flat list of all control nouns (preserves original order from story_generator.py)
# ---------------------------------------------------------------------------
CONTROL_NOUNS: list[str] = [
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

# ---------------------------------------------------------------------------
# Reverse mapping: noun -> cluster name
# ---------------------------------------------------------------------------
NOUN_TO_CLUSTER: dict[str, str] = {}
for _cluster_name, _nouns in NOUN_CLUSTERS.items():
    for _noun in _nouns:
        NOUN_TO_CLUSTER[_noun] = _cluster_name

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_all_clustered = [n for cluster in NOUN_CLUSTERS.values() for n in cluster]
assert len(NOUN_CLUSTERS) == 10, f"Expected 10 clusters, got {len(NOUN_CLUSTERS)}"
assert len(_all_clustered) == 162, f"Expected 162 nouns in clusters, got {len(_all_clustered)}"
assert len(set(_all_clustered)) == 162, "Duplicate nouns found in clusters"
assert set(_all_clustered) == set(CONTROL_NOUNS), (
    f"Cluster nouns do not match CONTROL_NOUNS.\n"
    f"  In clusters but not CONTROL_NOUNS: {set(_all_clustered) - set(CONTROL_NOUNS)}\n"
    f"  In CONTROL_NOUNS but not clusters: {set(CONTROL_NOUNS) - set(_all_clustered)}"
)
