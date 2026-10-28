"""
Shared configuration and helpers used across the analysis scripts.

All data files are read from the directory given by the DATA_DIR environment
variable, defaulting to ./data. Figures are written to ./figures.
"""

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

DATA_DIR = os.environ.get("DATA_DIR", "data")
FIG_DIR = os.environ.get("FIG_DIR", "figures")

# The three raw CrowdTangle collections (one keyword query each).
RAW_FILES = {
    "Breakup":    "انفصال OR فسخ العلاقة.csv",
    "Infidelity": "خيانة OR الخيانة الزوجية.csv",
    "Heartbreak": "كسر القلب OR حزن الحب.csv",
}
COLLECTIONS = ["Breakup", "Infidelity", "Heartbreak"]

# The Fanar-annotated sample (12,394 posts) and the human-annotated gold
# standard (1,000 posts), plus the manual categorisation of the Other/Unclear
# posts used for Table 4.
ANNOTATED_FILE = "annotated_fanar_10kposts_breakup_natural.csv"
HUMAN_FILE = "manual_annotation_breakup_1k.xlsx"
OTHER_UNCLEAR_FILE = "other_unclear_categorized.xlsx"

# Each annotated post carries its source collection as d1/d2/d3.
SOURCE_TO_COLLECTION = {"d1": "Breakup", "d2": "Infidelity", "d3": "Heartbreak"}

# --------------------------------------------------------------------------- #
# Label scheme
# --------------------------------------------------------------------------- #

COPING_LABELS = [
    "Emotional Venting",
    "Meaning-Making/Reframing",
    "Religious/Spiritual Coping",
    "Social Support Seeking",
    "Avoidance/Denial",
    "Blame/Resentment",
]
ALL_LABELS = COPING_LABELS + ["Other/Unclear"]

SHORT_NAME = {
    "Emotional Venting": "Venting",
    "Meaning-Making/Reframing": "Meaning-Making",
    "Religious/Spiritual Coping": "Religious",
    "Social Support Seeking": "Support",
    "Avoidance/Denial": "Avoidance",
    "Blame/Resentment": "Blame",
    "Other/Unclear": "Other",
}

REACTIONS = ["Likes", "Love", "Wow", "Haha", "Sad", "Angry", "Care"]

COLLECTION_COLOR = {"Breakup": "#2f6f9f", "Infidelity": "#c1543a", "Heartbreak": "#4a9d6f"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def data_path(filename):
    return os.path.join(DATA_DIR, filename)


def read_crowdtangle_csv(path):
    """Read a CrowdTangle export, handling the UTF-8 BOM and stray columns."""
    with open(path, "rb") as fh:
        bom = fh.read(3)
    encoding = "utf-8-sig" if bom == b"\xef\xbb\xbf" else "utf-8"
    df = pd.read_csv(path, encoding=encoding, dtype=str,
                     engine="python", on_bad_lines="skip")
    df = df.loc[:, ~df.columns.astype(str).str.match("^Unnamed")]
    df.columns = df.columns.str.strip()
    return df


# --------------------------------------------------------------------------- #
# Label handling
# --------------------------------------------------------------------------- #

def parse_labels(value):
    """Split a comma-separated label string into a set of labels."""
    return frozenset(part.strip() for part in str(value).split(",") if part.strip())


def has_label(series, label):
    """Boolean mask: which rows of a label column contain the given label."""
    pattern = label.replace("(", r"\(").replace(")", r"\)")
    return series.str.contains(pattern, regex=True, na=False)


def to_binary_matrix(label_sets, labels):
    """One-hot encode a sequence of label sets over a fixed label list."""
    return np.array([[1 if l in s else 0 for l in labels] for s in label_sets])


# --------------------------------------------------------------------------- #
# Grouping of noisy metadata fields
# --------------------------------------------------------------------------- #

def group_account_type(category):
    """Collapse CrowdTangle's raw Page Category into higher-level account types."""
    if pd.isna(category):
        return "Other/Unknown"
    c = str(category).upper()
    buckets = {
        "Media & News": ["NEWS", "MEDIA", "TV_CHANNEL", "TV_NETWORK", "BROADCAST",
                         "NEWSPAPER", "MAGAZINE", "JOURNALIST", "RADIO"],
        "Entertainment & Creators": ["TV_SHOW", "MOVIE", "ACTOR", "MUSICIAN", "ARTIST",
                                     "ENTERTAINMENT", "VIDEO_CREATOR", "DIGITAL_CREATOR",
                                     "GAMING", "PODCAST", "WRITER", "AUTHOR", "BOOK"],
        "Personal (individuals)": ["PERSON", "PERSONAL_BLOG", "PUBLIC_FIGURE", "BLOGGER"],
        "Community & Orgs": ["COMMUNITY", "ACTIVITY_GENERAL", "GROUP", "CAUSE",
                             "ORGANIZATION", "NONPROFIT", "RELIGIOUS"],
        "Business & Local": ["COMPANY", "BUSINESS", "BRAND", "PRODUCT", "SHOPPING",
                             "RETAIL", "RESTAURANT", "LOCAL", "SERVICE", "CLINIC",
                             "DOCTOR", "OPHTHALMOLOGIST", "STORE", "COMMERCIAL"],
    }
    for name, keys in buckets.items():
        if any(k in c for k in keys):
            return name
    return "Other/Unknown"


def group_post_format(post_type):
    """Collapse the Type field into Video / Photo / Status / Link / Other."""
    if pd.isna(post_type):
        return "Other"
    t = str(post_type)
    if "Video" in t or "YouTube" in t or "Live" in t:
        return "Video"
    return {"Photo": "Photo", "Status": "Status (text)", "Link": "Link"}.get(t, "Other")


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #

def cohen_kappa_binary(a, b):
    """Cohen's kappa for two binary 0/1 vectors."""
    a, b = np.asarray(a), np.asarray(b)
    p_o = np.mean(a == b)
    p1a, p1b = np.mean(a), np.mean(b)
    p_e = p1a * p1b + (1 - p1a) * (1 - p1b)
    return (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else float("nan")


def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74)
