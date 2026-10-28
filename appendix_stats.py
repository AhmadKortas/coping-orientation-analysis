"""
Appendix -- inferential support for the engagement analysis.

Produces:
  * bootstrap 95% confidence intervals for each coping orientation's mean
    reaction shares, over the full sample and within the heartbreak collection
    alone (Table 6);
  * an OLS regression of the sad/love/angry reaction share on coping orientation,
    controlling for post format, follower count (log), country, and year
    (Table 7).

    python appendix_stats.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import common as c

RNG = np.random.default_rng(20260101)   # fixed seed -> reproducible bootstrap CIs
N_BOOT = 10000
REACTS = ["Love", "Sad", "Angry", "Haha"]


def load():
    df = pd.read_csv(c.data_path(c.ANNOTATED_FILE))
    df["collection"] = df["source_dataset"].map(c.SOURCE_TO_COLLECTION)
    for col in c.REACTIONS + ["Followers at Posting"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["reaction_total"] = df[c.REACTIONS].sum(axis=1)
    for r in REACTS:
        df[f"{r}_share"] = np.where(df["reaction_total"] > 0,
                                    100 * df[r] / df["reaction_total"], np.nan)
    df["year"] = pd.to_datetime(df["Post Created Date"], errors="coerce").dt.year
    df["log_followers"] = np.log10(df["Followers at Posting"].where(df["Followers at Posting"] > 0))
    df["country"] = df["Page Admin Top Country"].fillna("NA")
    df["post_format"] = df["Type"].apply(c.group_post_format)
    return df


def boot_mean_ci(values):
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(v), size=(N_BOOT, len(v)))
    boot_means = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return (v.mean(), lo, hi)


def reaction_cis(df, title):
    print(f"\n{title}")
    print(f"{'orientation':16}" + "".join(f"{r:>20}" for r in REACTS))
    for l in c.COPING_LABELS:
        sub = df[c.has_label(df["llm_label"], l)]
        cells = []
        for r in REACTS:
            m, lo, hi = boot_mean_ci(sub[f"{r}_share"].to_numpy())
            cells.append(f"{m:4.1f} [{lo:4.1f},{hi:4.1f}]")
        print(f"{c.SHORT_NAME[l]:16}" + "".join(f"{x:>20}" for x in cells))
    cells = []
    for r in REACTS:
        m, lo, hi = boot_mean_ci(df[f"{r}_share"].to_numpy())
        cells.append(f"{m:4.1f} [{lo:4.1f},{hi:4.1f}]")
    print(f"{'All':16}" + "".join(f"{x:>20}" for x in cells))


def regression(df):
    print("\nOLS: reaction share ~ coping orientation + controls (Table 7)")
    terms = {}
    for l in c.COPING_LABELS:
        name = c.SHORT_NAME[l].lower().replace("-", "_")
        terms[name] = c.has_label(df["llm_label"], l).astype(int)
    X = pd.DataFrame(terms)
    base = df[["Sad_share", "Love_share", "Angry_share", "log_followers",
               "country", "year", "post_format", "reaction_total"]].reset_index(drop=True)
    data = pd.concat([X.reset_index(drop=True), base], axis=1)
    data = data[data["reaction_total"] > 0].dropna(
        subset=["log_followers", "year", "Sad_share"])
    data["year"] = data["year"].astype(int).astype(str)

    coping = " + ".join(terms.keys())
    controls = "log_followers + C(country) + C(year) + C(post_format)"
    for dep in ["Sad_share", "Love_share", "Angry_share"]:
        model = smf.ols(f"{dep} ~ {coping} + {controls}", data=data).fit(cov_type="HC1")
        print(f"\n--- {dep}  (n={int(model.nobs)}, R^2={model.rsquared:.3f}) ---")
        print(f"{'orientation':16}{'coef':>8}{'95% CI':>22}")
        for name in terms:
            b = model.params[name]; lo, hi = model.conf_int().loc[name]
            print(f"{name:16}{b:8.2f}   [{lo:6.2f}, {hi:6.2f}]")


if __name__ == "__main__":
    c.section("Appendix  Engagement: bootstrap CIs and controlled regression")
    df = load()
    print(f"N = {len(df):,} | posts with >=1 reaction = {(df['reaction_total'] > 0).sum():,}")
    reaction_cis(df, "Full sample [mean, 95% bootstrap CI]")
    reaction_cis(df[df["collection"] == "Heartbreak"],
                 f"Heartbreak only (n={(df['collection'] == 'Heartbreak').sum():,})")
    regression(df)
