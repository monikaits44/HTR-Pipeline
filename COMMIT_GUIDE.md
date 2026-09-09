# COMMIT & PUBLICATION GUIDE  —  Iteration 2 (2026-09-09)

> Step-by-step Git workflow for the **minimal-surface** publication profile
> introduced in this iteration. This guide reflects the new
> `.gitignore` policy: only `models.py`, `configs/`, `utils/`,
> `synthetic_data_generation/`, one curated notebook, and a small set of
> report-outline figures/tables under `outputs/` are pushed to GitHub.
> Everything else — including `scripts/`, `experiments_execution/`,
> `evaluation_execution/logs/`, `documents/`, `saved_models/`, `data/`,
> `logs/`, `external/`, `asset/`, `archive/`, `letter2index.json`,
> `index2letter.json`, and `FINAL_EXPERIMENTS.md` — is preserved on disk
> but **not** tracked in Git.

Repository : `git@github.com:monikaits44/HTR-Pipeline.git`
Working branch : `hpc-vit-register-v2`

---

## 0 · Sanity checklist before you start

Run all of these from the repo root
(`cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline`).

```bash
# 0.1  Confirm branch & remote
git branch --show-current                        # → hpc-vit-register-v2
git remote -v                                    # → origin git@github.com:monikaits44/HTR-Pipeline.git

# 0.2  Confirm the ignore rules match iteration 2
grep -c "^archive/\|^scripts/\|^documents/\|^experiments_execution/\|^saved_models/" .gitignore
# → should print 5

# 0.3  Confirm the tracked working surface is what you expect
git ls-files --others --exclude-standard | sort > /tmp/new_files.txt
wc -l /tmp/new_files.txt                         # should be a small number
head -30 /tmp/new_files.txt

# 0.4  Nothing large should be about to be tracked
git ls-files --others --exclude-standard \
  | xargs -r du -b 2>/dev/null | sort -n | tail -20
# → nothing > ~200 KB except the booklet GIF (~154 KB) and the ctc_posterior
#   PDF/PNG figures (~3 MB combined)

# 0.5  Ignore-effectiveness spot-check
git check-ignore -v archive/ asset/ data/ documents/ evaluation_execution/logs/ \
                    experiments_execution/ external/ logs/ saved_models/ \
                    scripts/ letter2index.json index2letter.json \
                    FINAL_EXPERIMENTS.md
# → every path should match a rule in .gitignore

# 0.6  Confirm the booklet GIF actually exists and is fresh
ls -la outputs/beyond_memorization/curated/booklet_attention_preview.gif
```

If (0.6) fails, regenerate it:

```bash
python scripts/postprocessing/make_booklet_gif.py
```

The script is workspace-only (`scripts/` is ignored) but the produced GIF
under `outputs/beyond_memorization/curated/` **is** part of the
publication surface.

---

## 1 · Create a safety branch

```bash
git branch backup-before-publication-iter2-2026-09-09
git log -1 --oneline backup-before-publication-iter2-2026-09-09
```

---

## 2 · Stage & commit in logical chunks

The working tree contains, after this iteration:

* **modifications** — the tightened `.gitignore`, the rewritten
  `README.md`, and any small edits to `configs/`, `utils/`, `models.py`.
* **deletions** — files whose content moved off Git (all the exploratory
  notebooks, `docs/README.md`, `scripts/`, `experiments_execution/`,
  `documents/`, `letter2index.json`, `index2letter.json`,
  `FINAL_EXPERIMENTS.md`, etc.). These will show as `D` in
  `git status` because they were tracked before the ignore rules were
  tightened.
* **additions** — `LICENSE`, `CITATION.cff`, the new `README.md` (rewritten),
  the curated booklet GIF, the five report-outline figures under
  `outputs/pub_figures/`, and the tables under `outputs/tables/`.

### 2.1 · Publication-profile `.gitignore`

```bash
git add .gitignore

git commit -m "chore(gitignore): iteration-2 minimal-surface publication profile

Ignores workspace-only assets so the public repo ships only the model,
utils, configs, synth-data pipeline, one curated notebook, and the
report-outline figures/tables under outputs/:

- archive/  asset/  data/  documents/  external/  logs/
- evaluation_execution/logs/  experiments_execution/
- saved_models/  scripts/
- root files: letter2index.json  index2letter.json  FINAL_EXPERIMENTS.md

Everything remains on disk under archive/ or its original location — the
change is purely a Git-visibility policy, not a deletion."
```

### 2.2 · Remove now-ignored artefacts from Git

Anything that was previously tracked but now falls under the new ignore
rules will appear as `D` in `git status`. Stage all deletions in a single
commit so the tracked tree matches the on-disk-tracked expectation.

```bash
git add -u                                     # stages every 'D' automatically

# Sanity-check what the commit will contain — nothing but deletions:
git diff --cached --name-status | awk '{print $1}' | sort -u
# → should print only 'D'

git commit -m "refactor: drop workspace-only paths from Git tracking

All content is preserved on disk under archive/ or its original folder;
this commit only removes it from Git tracking so the public tree
matches the iteration-2 publication profile.

Removed from tracking (still present locally):
- notebook/*.ipynb except beyond_memorization_attention_maps.ipynb
- scripts/**, experiments_execution/**, evaluation_execution/logs/**
- documents/**, saved_models/**, data/**, logs/**, external/**, asset/**
- docs/README.md (index into now-ignored documents/)
- letter2index.json, index2letter.json, FINAL_EXPERIMENTS.md"
```

### 2.3 · Documentation

```bash
git add README.md LICENSE CITATION.cff COMMIT_GUIDE.md

git commit -m "docs: iteration-2 README, LICENSE, citation & commit guide

- README.md rewritten for the minimal surface: booklet GIF hero,
  4 quantitative results tables (register sweep, architecture comparison,
  ablations, cross-dataset+seed), 4 qualitative figures (booklet GIF,
  register self-attention, register quantitative, CTC posterior), setup
  and reproducibility sections, related work and citation
- LICENSE (MIT + attribution) and CITATION.cff already at v1.0.0
- COMMIT_GUIDE.md updated for the iteration-2 policy"
```

### 2.4 · Curated qualitative outputs

```bash
git add outputs/.gitkeep \
        outputs/beyond_memorization/curated/booklet_attention_preview.gif \
        outputs/pub_figures/booklet_attention_preview.gif 2>/dev/null || true

git add outputs/pub_figures/ctc_posterior.png \
        outputs/pub_figures/fig3_register_quantitative.png \
        outputs/pub_figures/fig3_register_quantitative.pdf \
        outputs/pub_figures/fig5_register_comparison_attn_talks.png \
        outputs/pub_figures/fig5_register_comparison_attn_talks.pdf

git commit -m "outputs(figures): report-outline qualitative artefacts

- booklet_attention_preview.gif: 7-frame per-character cross-attention
  rollout for the IAM word 'booklet' (BM ICDAR 2025 reproduction),
  regenerable via scripts/postprocessing/make_booklet_gif.py
- ctc_posterior.png: CTC posterior + peak-column overlay (Fig 4b)
- fig3_register_quantitative.{png,pdf}: register-effect quantitative
  attention-quality metrics (Fig 5)
- fig5_register_comparison_attn_talks.{png,pdf}: CTC self-attention
  register sweep on 'talks.' (Fig 4)"
```

### 2.5 · Curated quantitative outputs

```bash
git add outputs/tables/

git commit -m "outputs(tables): unified CSV + LaTeX results tables

Machine-readable versions of every table in the report:
- unified_results.{csv,tex}: all 63 runs with best-epoch metrics
- architecture_comparison.tex: CNN-RNN vs ViT-RGTS vs pretrained
- register_sweep.tex: R ∈ {0,2,4,8,16} sweep on IAM
- ablations.tex: 22 architecture/training-strategy ablations
- significance_register_sweep.{csv,tex}: paired t-tests for R sweep
- significance_vs_cnn_rnn.{csv,tex}: significance vs CNN-RNN baseline
- statistical_tests.{csv,tex}: full significance matrix
- final_results_report.csv: exact numbers embedded in the PDF report
- all_runs.csv: per-run metadata (config, epochs, params, wall-clock)"
```

### 2.6 · Curated notebook

```bash
git add notebook/beyond_memorization_attention_maps.ipynb

git commit -m "notebook: curated Beyond-Memorization attention-maps notebook

The single publication-surface Jupyter notebook, documenting the
attention-maps derivation for the BM ICDAR 2025 reproduction. All other
exploratory notebooks moved to archive/notebook_exploration/ (workspace
only)."
```

### 2.7 · Any remaining source code changes

```bash
git status --short
# → should be empty; if anything remains (e.g. utils/*.py or models.py
#   modifications), stage them:
git add -u utils/ models.py configs/
git commit -m "src: minor final adjustments to utils/, models.py, configs/"
```

---

## 3 · Local verification

```bash
# 3.1  Import smoke-test (must print two OK lines)
source .venv/bin/activate
python - <<'PY'
import sys; sys.path.insert(0, '.')
from models import HTRNet
from utils.htr_dataset import HTRDataset
from utils.metrics import CER, WER
from utils.attention_extractor import AttentionExtractor
from utils.attention_metrics import (attention_entropy,
                                      attention_sparsity_gini, peak_sharpness)
from utils.transforms import aug_transforms_cnn, aug_transforms_vit
from utils.finetuning import build_finetune_optimizer, GradualUnfreezer
print('OK - core imports')
PY

# 3.2  Repo size after ignore rules (should be a few MB)
git count-objects -vH | grep -E 'size|count'

# 3.3  Confirm no large binaries slipped in
git ls-files | xargs -I{} du -b "{}" 2>/dev/null | sort -n | tail -20
# → the largest file should be the booklet GIF (~154 KB) or a report figure

# 3.4  Confirm no secrets tracked
git ls-files | xargs grep -lE \
    'api[_-]?key|BEGIN [A-Z]+ PRIVATE KEY|AKIA[0-9A-Z]{16}' 2>/dev/null
# → prints nothing

# 3.5  README pre-flight: confirm every link resolves
grep -oE '\(outputs/[^)]+\)' README.md | tr -d '()' | while read p; do
  [ -e "$p" ] || echo "MISSING: $p"
done
# → prints nothing
```

---

## 4 · Push to origin

```bash
# 4.1  Working branch first
git push origin hpc-vit-register-v2

# 4.2  Open the branch URL in a browser and verify:
#   - README renders with the booklet GIF at the top (autoplay-loops)
#   - all 4 result tables render correctly
#   - 4 in-README images resolve (no broken thumbnails)
#   - CITATION.cff "Cite this repository" widget appears
#   - LICENSE shows as MIT in the sidebar
#   - archive/, data/, saved_models/, outputs subfolders other than the
#     curated ones are absent from the file tree
#   - only ONE notebook (beyond_memorization_attention_maps.ipynb) present
```

---

## 5 · Merge into `main` and tag

Only after the browser review in §4 passes.

```bash
git checkout main
git pull  --ff-only origin main
git merge --ff-only hpc-vit-register-v2
git push  origin main

git tag -a v1.1.0 -m "v1.1.0 — Iteration-2 minimal-surface publication (2026-09-09)

- Publication profile reduced to models.py, configs/, utils/,
  synthetic_data_generation/, one curated notebook, and outputs/
- Booklet GIF (BM ICDAR 2025 reproduction) as README hero image
- README rewritten with 4 quantitative + 4 qualitative results
- Report source (documents/) and full training harness
  (scripts/, experiments_execution/) moved off-Git per new .gitignore"
git push origin v1.1.0
```

---

## 6 · Flip repository to public

Only after §5 succeeds:

```
GitHub → repository → Settings → General → Danger Zone
       → Change repository visibility → Public
       → type "monikaits44/HTR-Pipeline" to confirm
```

Then, on the same Settings page:

* **About** (right sidebar of repo home): short description
  ("Improving HTR Attention Maps with Register Tokens — MSc project,
  FAU Erlangen"), topics
  `handwriting-recognition transformer ctc attention register-tokens
  iam pytorch reproducibility`.
* **Releases**: draft a release from tag `v1.1.0`. **Attach**
  `documents/final/pr_htr_report/main.pdf` from the workspace (the PDF
  is not tracked in Git, so it must be uploaded through the Releases UI).

---

## 7 · Post-publication housekeeping

```bash
git fetch --all --prune
git checkout main && git pull --ff-only
```

Delete the working branch on GitHub only if you no longer need it:

```bash
git push origin --delete hpc-vit-register-v2
```

---

## Appendix A · What is TRACKED after iteration 2

```
.
├── .gitignore
├── CITATION.cff
├── COMMIT_GUIDE.md
├── LICENSE
├── README.md
├── requirements.txt
├── models.py
│
├── configs/                              (10 YAML files)
├── utils/                                (8 Python modules)
├── synthetic_data_generation/            (7-step pipeline + LMDB loader)
│
├── notebook/
│   ├── beyond_memorization_attention_maps.ipynb
│   └── sample_images/                    (small sample PNGs for demos)
│
└── outputs/
    ├── .gitkeep
    ├── beyond_memorization/curated/
    │   └── booklet_attention_preview.gif
    ├── pub_figures/
    │   ├── ctc_posterior.png
    │   ├── fig3_register_quantitative.{png,pdf}
    │   └── fig5_register_comparison_attn_talks.{png,pdf}
    └── tables/                           (10 CSV + LaTeX table files)
```

## Appendix B · What is preserved LOCALLY but NOT tracked

| Path | Contents |
|------|----------|
| `archive/` | All historical exploratory code, notebooks, per-run analysis notes, IAM raw data mirror, papers, dashboard, legacy postprocessing v1, plus this iteration's `notebook_exploration/`, `outputs_full_pre_curation/`, and `beyond_memorization_reproduction/`. Nothing has been deleted. |
| `scripts/` | Full training / postprocessing / writer-identification codebase (`trainer.py`, `pub_fig*.py`, `postprocessing/**`, `visualization/**`, `writer_identification/**`) |
| `experiments_execution/` | 63 SLURM launchers + `run_experiment.sh` + submission scripts |
| `evaluation_execution/logs/` | Batch evaluation logs |
| `documents/` | Report source (`pr_htr_report/main.{tex,pdf}`), analysis MDs, figure TikZ, backup versions of the report |
| `saved_models/` | All 63+ per-run checkpoints (443 GB) |
| `data/` | IAM + READ2016 + synthetic (11 GB) |
| `external/` | Beyond-Memorization + WordStylist + SD-VAE clone (3.1 GB) |
| `asset/pretrained_models/` | Downloaded weights |
| `logs/`, `evaluation_execution/logs/` | SLURM stdout/stderr |
| `letter2index.json`, `index2letter.json` | Character maps (workspace-only per iteration-2 policy) |
| `FINAL_EXPERIMENTS.md` | Canonical experiment matrix reference |

## Appendix C · Potential issues to double-check

1. **README image links use relative paths that reference `outputs/…`.**
   All four images the README embeds (booklet GIF + 3 pub_figures PNGs)
   are inside the tracked publication surface, so they will render on
   GitHub. If any file is renamed or moved, remember to update the
   README references — the pre-flight check in §3.5 catches this.
2. **Report PDF distribution.** `documents/` is git-ignored, so the
   compiled report will not appear in the file tree. Distribute it via
   a **GitHub Release attachment** (§6) or a Zenodo DOI.
3. **`notebook/beyond_memorization_attention_maps.ipynb` outputs.** The
   notebook contains rendered cell outputs (figures). If the outputs
   cause diffs on re-runs, install
   [`nbstripout`](https://github.com/kynan/nbstripout) and enable the
   pre-commit filter — but this is optional for the initial release.
4. **BM archive size on disk.** `archive/beyond_memorization_reproduction/`
   is ~99 MB. Correctly ignored, but keep in mind if you ever push to
   a low-storage backup target.
5. **Booklet GIF regeneration relies on the workspace-only
   `scripts/` folder.** If a fresh downstream user needs to regenerate
   the GIF, extract just `make_booklet_gif.py` from
   `archive/beyond_memorization_reproduction/gif_source/` and copy it
   somewhere on `PYTHONPATH`.

---

_This guide reflects the iteration-2 policy applied on 2026-09-09. Retain
in the repo (it is intentionally lightweight and part of the tracked
surface) or delete after use — it is not required for runtime._
