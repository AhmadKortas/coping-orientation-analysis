"""
RQ1 -- Corpus characterization.

Reads the three raw CrowdTangle collections and produces the volume/time-span
and account-type tables and the temporal, post-format, and geography figures.

    python rq1_corpus.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import Counter

import common as c


def load_collections():
    return {name: c.read_crowdtangle_csv(c.data_path(f))
            for name, f in c.RAW_FILES.items()}


def characterise(data):
    c.section("RQ1  Corpus characterization")
    stats = {}
    total = 0
    for name in c.COLLECTIONS:
        df = data[name]
        dates = pd.to_datetime(df["Post Created Date"], errors="coerce")
        pages = df["Page Name"].value_counts()
        stats[name] = {
            "n": len(df),
            "span": (str(dates.min().date()), str(dates.max().date())),
            "pages": int(df["Page Name"].dropna().nunique()),
            "top10": 100 * pages.head(10).sum() / len(df),
            "year": {int(y): int(n) for y, n in
                     dates.dt.year.value_counts().sort_index().items()},
            "category": df["Page Category"].apply(c.group_account_type)
                          .value_counts(normalize=True).mul(100).to_dict(),
            "format": df["Type"].apply(c.group_post_format)
                        .value_counts(normalize=True).mul(100).to_dict(),
            "country": df["Page Admin Top Country"].fillna("NA")
                         .value_counts(normalize=True).mul(100).to_dict(),
            "followers": pd.to_numeric(df["Followers at Posting"], errors="coerce"),
        }
        total += stats[name]["n"]

    print("\nVolume, span, distinct pages, top-10 concentration (Table 1)")
    for name in c.COLLECTIONS:
        s = stats[name]
        print(f"  {name:11} n={s['n']:>6,}  {s['span'][0]}..{s['span'][1]}  "
              f"pages={s['pages']:>6,}  top10={s['top10']:4.1f}%  "
              f"median_followers={s['followers'].median():,.0f}")
    print(f"  {'TOTAL':11} n={total:,}")

    print("\nShare of posts from 2020 onward")
    for name in c.COLLECTIONS:
        recent = sum(n for yr, n in stats[name]["year"].items() if yr >= 2020)
        print(f"  {name:11} {100 * recent / stats[name]['n']:.1f}%")

    print("\nAccount-type composition (%, Table 2)")
    cats = ["Media & News", "Entertainment & Creators", "Community & Orgs",
            "Personal (individuals)", "Business & Local", "Other/Unknown"]
    for cat in cats:
        print(f"  {cat:26} " +
              "  ".join(f"{stats[n]['category'].get(cat, 0):5.1f}" for n in c.COLLECTIONS))

    print("\nPost-format composition (%)")
    for fmt in ["Video", "Photo", "Status (text)", "Link", "Other"]:
        print(f"  {fmt:14} " +
              "  ".join(f"{stats[n]['format'].get(fmt, 0):5.1f}" for n in c.COLLECTIONS))

    return stats


def make_figures(stats):
    c.ensure_fig_dir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 300})
    years = list(range(2014, 2025))

    # Figure 1: posting volume by year
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name in c.COLLECTIONS:
        ax.plot(years, [stats[name]["year"].get(y, 0) for y in years],
                marker="o", ms=5, lw=2.2, color=c.COLLECTION_COLOR[name], label=name)
    ax.axvspan(2023.5, 2024.5, color="grey", alpha=0.10)
    ax.set_xlabel("Year of posting"); ax.set_ylabel("Number of posts")
    ax.set_title("Posting volume by year, per collection"); ax.set_xticks(years)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{c.FIG_DIR}/fig_temporal.png", bbox_inches="tight")
    plt.close(fig)

    # 100%-stacked horizontal bars, reused for account type and post format
    def stacked(field, categories, palette, title, fname):
        fig, ax = plt.subplots(figsize=(8.4, 3.3))
        ypos = np.arange(len(c.COLLECTIONS))[::-1]
        left = np.zeros(len(c.COLLECTIONS))
        for i, cat in enumerate(categories):
            vals = np.array([stats[n][field].get(cat, 0.0) for n in c.COLLECTIONS])
            ax.barh(ypos, vals, left=left, color=palette[i], label=cat,
                    height=0.6, edgecolor="white", lw=0.6)
            for j, (v, l) in enumerate(zip(vals, left)):
                if v >= 6:
                    ax.text(l + v / 2, ypos[j], f"{v:.0f}%", ha="center", va="center",
                            fontsize=8.5, color="white", fontweight="bold")
            left += vals
        ax.set_yticks(ypos); ax.set_yticklabels(c.COLLECTIONS)
        ax.set_xlim(0, 100); ax.set_xlabel("Share of posts (%)")
        ax.set_title(title); ax.grid(axis="y", alpha=0)
        ax.legend(frameon=False, ncol=3, fontsize=8.3,
                  loc="upper center", bbox_to_anchor=(0.5, -0.22))
        fig.tight_layout(); fig.savefig(f"{c.FIG_DIR}/{fname}", bbox_inches="tight")
        plt.close(fig)

    stacked("category",
            ["Media & News", "Entertainment & Creators", "Community & Orgs",
             "Personal (individuals)", "Business & Local", "Other/Unknown"],
            ["#c1543a", "#e0a030", "#4a9d6f", "#2f6f9f", "#8a6bbf", "#b8b8b8"],
            "Account types by collection", "fig_accounts.png")
    stacked("format",
            ["Video", "Photo", "Status (text)", "Link", "Other"],
            ["#2f6f9f", "#4a9d6f", "#e0a030", "#8a6bbf", "#b8b8b8"],
            "Post-format composition", "fig_posttype.png")

    # Figure 2: top page-admin countries
    avg = Counter()
    for n in c.COLLECTIONS:
        for country, p in stats[n]["country"].items():
            if country != "NA":
                avg[country] += p / 3
    top = [country for country, _ in avg.most_common(7)]
    names = {"EG": "Egypt", "IQ": "Iraq", "MA": "Morocco", "DZ": "Algeria",
             "AE": "UAE", "SY": "Syria", "TR": "Turkey", "LB": "Lebanon"}
    x = np.arange(len(top)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    for i, n in enumerate(c.COLLECTIONS):
        ax.bar(x + (i - 1) * w, [stats[n]["country"].get(country, 0) for country in top],
               w, label=n, color=c.COLLECTION_COLOR[n])
    ax.set_xticks(x); ax.set_xticklabels([names.get(country, country) for country in top])
    ax.set_ylabel("Share of posts (%)")
    ax.set_title("Top page-admin countries by collection")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{c.FIG_DIR}/fig_geography.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data = load_collections()
    stats = characterise(data)
    make_figures(stats)
    print(f"\nFigures written to ./{c.FIG_DIR}/")
