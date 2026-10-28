# Coping Orientations Analysis

Analysis code for the paper **"Beyond the Breakup: Annotating Coping Orientations
in Arabic Relationship-Dissolution Discourse."**

The paper studies how people cope with relationship loss in public Arabic
Facebook discourse. It builds a corpus of 67,700 posts across three collections
(betrayal/infidelity, breakup, and heartbreak), annotates a sample of them for
*coping orientation* with the Arabic-centric Fanar model, and validates those
annotations against a human expert. This repository contains the code that
produces every number, table, and figure in the paper.

## What each script does

| Script | Paper section | Produces |
|---|---|---|
| `rq1_corpus.py` | RQ1 – Corpus characterization | Volume/time-span and account-type tables; temporal, post-format, and geography figures |
| `rq2_engagement.py` | RQ2 – Coping distribution & engagement | Coping-orientation distribution (overall and by collection), the Other/Unclear breakdown, the reaction-composition figure, and the engagement numbers |
| `rq3_agreement.py` | RQ3 – Reliability | Per-label human–Fanar agreement (Cohen's κ), the Other/Unclear disagreement analysis, the codeable-subset agreement, and the confusion figure |
| `appendix_stats.py` | Appendix | Bootstrap 95% confidence intervals for the reaction shares and the controlled OLS regression |
| `common.py` | — | Shared configuration and helper functions used by all four scripts |

Each script prints the results it computes and writes any figures to `figures/`.
They can be run in any order and are independent of one another.

## Data

The analysis uses files exported from CrowdTangle plus the annotated samples.
They are **not** included in this repository (they contain third-party social
media content). Place them in a `data/` folder:

```
data/
  انفصال_OR_فسخ_العلاقة.csv              # breakup collection
  خيانة_OR_الخيانة_الزوجية.csv            # infidelity collection
  كسر_القلب_OR_حزن_الحب.csv               # heartbreak collection
  annotated_fanar_10kposts_breakup_natural_1.csv   # 12,394 Fanar-annotated posts
  manual_annotation_breakup_1k.xlsx                # 1,000 human-annotated posts
  other_unclear_categorized.xlsx                   # manual coding of Other/Unclear posts
```

The file names are set in `common.py`; change them there if yours differ. By
default the scripts read from `./data`; you can point elsewhere with the
`DATA_DIR` environment variable.

## Running

```bash
pip install -r requirements.txt

python rq1_corpus.py
python rq2_engagement.py
python rq3_agreement.py
python appendix_stats.py
```

To read the data from a different folder:

```bash
DATA_DIR=/path/to/data python rq2_engagement.py
```

## Notes

- The bootstrap confidence intervals use a fixed random seed, so they reproduce
  exactly on re-run. The intervals are stable regardless of the seed.
- The coping-orientation labels are read as assigned by the annotators; the code
  parses the comma-separated label strings into sets, since the scheme is
  multi-label.
- `Table 4` (the breakdown of the Other/Unclear posts) comes from a manual
  categorisation; `rq2_engagement.py` reads and aggregates that file rather than
  recomputing the categories.
