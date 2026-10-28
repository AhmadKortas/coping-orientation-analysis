"""
RQ2 -- Coping-orientation distribution and audience engagement.

Reads the Fanar-annotated sample (12,394 posts) and produces:
  * the coping-orientation distribution overall and by collection (Table 3),
  * the breakdown of the Other/Unclear posts (Table 4, from the manual coding),
  * the reaction-composition figure (Figure 4) and the engagement numbers.

    python rq2_engagement.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

import common as c


def load_annotated():
    df = pd.read_csv(c.data_path(c.ANNOTATED_FILE))
    df["collection"] = df["source_dataset"].map(c.SOURCE_TO_COLLECTION)
    for col in ["Total Interactions", "Comments", "Shares"] + c.REACTIONS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["reaction_total"] = df[c.REACTIONS].sum(axis=1)
    return df


def distribution(df):
    n = len(df)
    label_sets = df["llm_label"].apply(c.parse_labels)
    print(f"\nN = {n:,} | mean labels/post = {label_sets.apply(len).mean():.3f}")

    print("Label document frequency (%, Table 3 'All' column)")
    freq = Counter(l for s in label_sets for l in s)
    for l in c.ALL_LABELS:
        print(f"  {l:28} {freq.get(l, 0):5d}  ({100 * freq.get(l, 0) / n:.1f}%)")

    print("\nPrevalence by source collection (%, Table 3)")
    for l in c.ALL_LABELS:
        mask = c.has_label(df["llm_label"], l)
        by_col = "  ".join(
            f"{100 * mask[df['collection'] == col].mean():5.1f}" for col in c.COLLECTIONS)
        print(f"  {l:28} {by_col}")

    print("\nTop label co-occurrences")
    pairs = Counter()
    for s in label_sets:
        for a, b in combinations(sorted(s), 2):
            pairs[(a, b)] += 1
    for (a, b), count in pairs.most_common(5):
        print(f"  {count:4d}  {a} + {b}")


def other_unclear_breakdown():
    """Read the manual categorisation of the Other/Unclear posts (Table 4)."""
    print("\nOther/Unclear breakdown (Table 4)")
    summary = pd.read_excel(c.data_path(c.OTHER_UNCLEAR_FILE), sheet_name="Summary")
    summary = summary.dropna(subset=["Group", "Posts"])
    summary = summary[summary["Group"].astype(str).str.match(r"^[A-C]\.")]
    total = summary["Posts"].sum()
    for group, sub in summary.groupby("Group"):
        print(f"  {group}  ({100 * sub['Posts'].sum() / total:.0f}%)")
        for _, row in sub.iterrows():
            print(f"     {row['Category']:42} {100 * row['Posts'] / total:4.1f}%")


def engagement(df):
    react = ["Love", "Sad", "Angry", "Haha"]
    for r in react:
        df[f"{r}_share"] = np.where(df["reaction_total"] > 0,
                                    100 * df[r] / df["reaction_total"], np.nan)
    df["comment_share"] = np.where(df["Total Interactions"] > 0,
                                   100 * df["Comments"] / df["Total Interactions"], np.nan)
    df["share_share"] = np.where(df["Total Interactions"] > 0,
                                 100 * df["Shares"] / df["Total Interactions"], np.nan)

    print("\nMedian engagement per label")
    print(f"  {'label':26} {'n':>5} {'TotInt':>7} {'Comments':>9} {'Shares':>7}")
    for l in c.COPING_LABELS:
        sub = df[c.has_label(df["llm_label"], l)]
        print(f"  {l:26} {len(sub):5d} {sub['Total Interactions'].median():7.0f} "
              f"{sub['Comments'].median():9.0f} {sub['Shares'].median():7.0f}")

    print("\nReaction composition (mean % of reactions) + comment/share %")
    print(f"  {'label':26} {'Love':>6} {'Sad':>6} {'Angry':>6} {'Haha':>6} "
          f"{'Comm%':>6} {'Shr%':>6}")
    composition = {}
    for l in c.COPING_LABELS:
        sub = df[c.has_label(df["llm_label"], l)]
        composition[l] = {r: sub[f"{r}_share"].mean() for r in react}
        print(f"  {l:26} " +
              " ".join(f"{sub[f'{r}_share'].mean():6.1f}" for r in react) +
              f" {sub['comment_share'].mean():6.1f} {sub['share_share'].mean():6.1f}")
    print(f"  {'corpus average':26} " +
          " ".join(f"{df[f'{r}_share'].mean():6.1f}" for r in react) +
          f" {df['comment_share'].mean():6.1f} {df['share_share'].mean():6.1f}")
    return composition


def reaction_figure(composition):
    c.ensure_fig_dir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 300})
    order = [c.SHORT_NAME[l] for l in c.COPING_LABELS]
    react, colors = ["Love", "Sad", "Angry", "Haha"], ["#c0567a", "#3b7dad", "#c0392b", "#e0a030"]
    x = np.arange(len(order)); w = 0.2
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for i, (r, col) in enumerate(zip(react, colors)):
        ax.bar(x + (i - 1.5) * w, [composition[l][r] for l in c.COPING_LABELS],
               w, label=r, color=col)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Share of reactions (%)")
    ax.set_title("Reaction composition by coping orientation")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(f"{c.FIG_DIR}/fig_rq2_reactions.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    c.section("RQ2  Coping distribution and audience engagement")
    df = load_annotated()
    distribution(df)
    other_unclear_breakdown()
    composition = engagement(df)
    reaction_figure(composition)
    print(f"\nFigure written to ./{c.FIG_DIR}/fig_rq2_reactions.png")
