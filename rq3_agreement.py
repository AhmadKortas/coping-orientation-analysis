"""
RQ3 -- Reliability of the Fanar annotations against a human expert.

Reads the 1,000-post human-annotated gold standard (which also carries Fanar's
labels) and produces the per-label agreement table (Table 5), the overall
agreement figures, the Other/Unclear disagreement analysis, the codeable-subset
agreement, and the primary-label confusion figure (Figure 5).

    python rq3_agreement.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

import common as c

HUMAN_COL = "Human_annotation(Aisha)"
MODEL_COL = "llm_label"


def load_gold():
    df = pd.read_excel(c.data_path(c.HUMAN_FILE))
    human = df[HUMAN_COL].apply(c.parse_labels)
    model = df[MODEL_COL].apply(c.parse_labels)
    return df, human, model


def overall_and_per_label(human, model):
    n = len(human)
    exact = np.mean([h == m for h, m in zip(human, model)])
    jaccard = np.mean([len(h & m) / len(h | m) if (h | m) else 1.0
                       for h, m in zip(human, model)])
    print(f"\nN = {n:,}")
    print(f"  exact set match   {100 * exact:.1f}%")
    print(f"  mean Jaccard      {jaccard:.3f}")
    print(f"  mean labels/post  human={np.mean([len(s) for s in human]):.2f}  "
          f"model={np.mean([len(s) for s in model]):.2f}")

    Hb = c.to_binary_matrix(human, c.ALL_LABELS)
    Mb = c.to_binary_matrix(model, c.ALL_LABELS)
    print("\nPer-label agreement (Table 5, Full column)")
    print(f"  {'label':28} {'Human%':>7} {'Fanar%':>7} {'kappa':>7}")
    kappa_full = {}
    for i, l in enumerate(c.ALL_LABELS):
        kappa_full[l] = c.cohen_kappa_binary(Hb[:, i], Mb[:, i])
        print(f"  {l:28} {100 * Hb[:, i].mean():7.1f} "
              f"{100 * Mb[:, i].mean():7.1f} {kappa_full[l]:7.3f}")
    return kappa_full


def first_label(value):
    """The first label in a comma-separated label string (the primary label)."""
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return parts[0] if parts else None


def other_unclear_disagreement(df):
    # Two views of the same posts -- those the human marked Other/Unclear:
    #  - multi-label frequency: how often the model assigns each label, counting
    #    all labels per post (can exceed the number of posts);
    #  - primary label: the model's first label, which is what the confusion
    #    figure (Figure 5) shows and what the paper text reports.
    human_primary = df[HUMAN_COL].apply(first_label)
    model_primary = df[MODEL_COL].apply(first_label)
    mask = df[HUMAN_COL].apply(lambda v: "Other/Unclear" in c.parse_labels(v))
    n = int(mask.sum())

    print("\nModel labels on the posts the human marked Other/Unclear "
          "(multi-label frequency)")
    freq = Counter(l for v in df.loc[mask, MODEL_COL] for l in c.parse_labels(v))
    for l, count in freq.most_common():
        print(f"  {l:28} {count:4d}  ({100 * count / n:.1f}%)")

    print("\nModel PRIMARY label on the same posts "
          "(matches Figure 5 and the paper text)")
    primary_mask = human_primary == "Other/Unclear"
    pfreq = Counter(model_primary[primary_mask])
    m = int(primary_mask.sum())
    for l in c.ALL_LABELS:
        if pfreq.get(l, 0):
            print(f"  {l:28} {pfreq[l]:4d}  ({100 * pfreq[l] / m:.1f}%)")


def codeable_subset(human, model):
    keep = [i for i in range(len(human))
            if "Other/Unclear" not in human.iloc[i]
            and "Other/Unclear" not in model.iloc[i]]
    Hs = [human.iloc[i] for i in keep]
    Ms = [model.iloc[i] for i in keep]
    exact = np.mean([h == m for h, m in zip(Hs, Ms)])
    jaccard = np.mean([len(h & m) / len(h | m) if (h | m) else 1.0
                       for h, m in zip(Hs, Ms)])
    Hb = c.to_binary_matrix(Hs, c.COPING_LABELS)
    Mb = c.to_binary_matrix(Ms, c.COPING_LABELS)
    kappa_sub = {l: c.cohen_kappa_binary(Hb[:, i], Mb[:, i])
                 for i, l in enumerate(c.COPING_LABELS)}
    print(f"\nCodeable subset (Table 5, Codeable column), n={len(keep)}")
    print(f"  exact set match   {100 * exact:.1f}%")
    print(f"  mean Jaccard      {jaccard:.3f}")
    for l in c.COPING_LABELS:
        print(f"  kappa  {l:28} {kappa_sub[l]:.3f}")
    macro_full_note = np.mean([c.cohen_kappa_binary(
        c.to_binary_matrix(human, c.COPING_LABELS)[:, i],
        c.to_binary_matrix(model, c.COPING_LABELS)[:, i]) for i in range(len(c.COPING_LABELS))])
    macro_sub = np.mean([kappa_sub[l] for l in c.COPING_LABELS])
    print(f"  macro kappa (6 coping labels)  full={macro_full_note:.3f}  subset={macro_sub:.3f}")


def confusion_figure(df):
    c.ensure_fig_dir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "figure.dpi": 300})

    def primary(value):
        parts = [p.strip() for p in str(value).split(",") if p.strip()]
        return parts[0] if parts else None

    hp = df[HUMAN_COL].apply(primary)
    mp = df[MODEL_COL].apply(primary)
    M = np.zeros((len(c.ALL_LABELS), len(c.ALL_LABELS)))
    for h, m in zip(hp, mp):
        M[c.ALL_LABELS.index(h), c.ALL_LABELS.index(m)] += 1
    row_norm = M / M.sum(axis=1, keepdims=True) * 100

    labs = [c.SHORT_NAME[l] for l in c.ALL_LABELS]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(7)); ax.set_xticklabels(labs, rotation=35, ha="right")
    ax.set_yticks(range(7)); ax.set_yticklabels(labs)
    ax.set_xlabel("Fanar primary label"); ax.set_ylabel("Human primary label")
    ax.set_title("Primary-label agreement (row-normalised %)")
    for i in range(7):
        for j in range(7):
            if row_norm[i, j] >= 1:
                ax.text(j, i, f"{row_norm[i, j]:.0f}", ha="center", va="center",
                        fontsize=8.5, color="white" if row_norm[i, j] > 55 else "#222")
    fig.colorbar(im, fraction=0.046, pad=0.04).set_label("% of human row")
    fig.tight_layout(); fig.savefig(f"{c.FIG_DIR}/fig_rq3_confusion.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    c.section("RQ3  Human-model agreement")
    df, human, model = load_gold()
    overall_and_per_label(human, model)
    other_unclear_disagreement(df)
    codeable_subset(human, model)
    confusion_figure(df)
    print(f"\nFigure written to ./{c.FIG_DIR}/fig_rq3_confusion.png")