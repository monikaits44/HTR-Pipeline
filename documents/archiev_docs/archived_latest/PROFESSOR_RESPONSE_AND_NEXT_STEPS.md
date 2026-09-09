# Professor Response & Next Steps — Register Token Attention Analysis

> **Date**: 2026-03-04  
> **Context**: Professor reviewed alignment plots from `visualizations/character_attention_mapping/alignment/` for images `a06-110-08` across runs 50–54 (0, 2, 4, 8, 16 register tokens).  
> **Professor's feedback**: "This looks already quite interesting. I guess that you don't need many of these registers, in the original paper, they effectively also use 1-2, it seems that, from the plots, the register-2 variant also looks best, but it is indeed hard to interpret, the 4 variant also looks quite good."

---

## 1. Proposed Response to Professor

> Dear Professor,
>
> Thank you very much for the feedback — it's very helpful in guiding the next steps.
>
> I agree with your observation: the Reg-2 variant (run_51) visually produces the sharpest and most spatially coherent character-level attention, and Reg-4 (run_52) is a close second. This aligns well with the findings in the original "Vision Transformers Need Registers" paper (Darcet et al., 2024), where 1–2 registers were found sufficient to eliminate attention artifacts.
>
> To quantify what we see in the plots, I measured the **Spearman rank correlation** (diagonality score) between character index and peak attention position across images. For short words, Reg-2 achieves the highest diagonality (ρ ≈ 0.55 vs. 0.28 for Reg-0), confirming that a small number of registers noticeably improves the left-to-right reading alignment of the attention heads.
>
> Interestingly, the full test-set metrics paint a slightly nuanced picture:
>
> | Run | Registers | Test CER | Test WER |
> |-----|-----------|----------|----------|
> | 50  | 0         | 6.26%    | 20.52%   |
> | 51  | **2**     | 6.12%    | 19.87%   |
> | 52  | **4**     | 6.07%    | 19.77%   |
> | 53  | 8         | 6.17%    | 20.27%   |
> | 54  | 16        | 5.97%    | 19.40%   |
>
> All register-augmented models improve over the baseline (Reg-0), with the sweet spot appearing to be at **2–4 registers** for attention interpretability and at **2–4 registers** for the best balance of accuracy and explainability. The Reg-16 model achieves a marginally better test CER (5.97%), though the improvement over Reg-2/4 is small (< 0.2% absolute) and it is harder to interpret visually, likely because the additional registers absorb more attention signal.
>
> The biggest insight so far is a **regime shift by text length**: all models achieve 0% CER on short words (≤ 11 chars) with clean diagonal attention patterns, but degrade to 10–13% CER on long lines (56+ chars) where attention becomes diffuse. This suggests that the benefit of registers is primarily in **attention quality / interpretability rather than raw recognition accuracy**, and that long-line performance is bounded by the BiLSTM CTC head's contextual capacity, not by attention routing.
>
> To make these findings more robust and easier to interpret, I plan the following next steps:
>
> 1. **Stratified analysis by text length**: Run the character attention mapping on a **curated set of images across short / medium / long text** from each of the train, validation, and test splits — this will let us systematically verify whether the Reg-2's attention advantage holds across varying complexities.
>
> 2. **Focused comparison on Reg-0 vs. Reg-2 vs. Reg-4**: Since these three are the most promising (and in line with the original paper), I'll produce detailed side-by-side visualizations specifically for these models on the curated set.
>
> 3. **Per-sample error analysis**: Identify images where the models disagree (one correct, another wrong) and show how attention differences explain the outcome.
>
> 4. **Quantitative attention quality metrics**: Compute diagonality, entropy, and peak sharpness across a larger sample to provide statistics rather than relying on visual inspection of individual images.
>
> I'll prepare these results and share them as soon as they're ready.
>
> Best regards

---

## 2. Experiment Plan — Systematic Evaluation

### 2.1 Curated Image Set (Stratified by Text Length & Split)

The goal is to move beyond the current 15 sample images and build a **representative evaluation set** that covers:

- **Text length buckets**: Short (2–10 chars), Medium (11–30 chars), Long (31–60 chars), Very Long (61+ chars)
- **Data splits**: Train, Validation, and Test — to check generalization vs. memorization
- **Difficulty levels**: Images where all models agree (easy) and where models disagree (hard/interesting)

#### Recommended Curated Set

| Category | Length | Split | Example Images | Ground Truth | Purpose |
|----------|--------|-------|----------------|--------------|---------|
| **Short** | 2–5 chars | Train | `a06-110-08` | "on." | Known easy — verify Reg-2 diagonal advantage |
| **Short** | 2–5 chars | Test | `p02-144-11` | "her." | Test generalization on unseen short words |
| **Short** | 6–10 chars | Train | `a01-038-12` | "talks." | 6 chars — already perfectly handled by all models |
| **Short** | 6–10 chars | Test | Select ~3 images | Various | Bridge between trivial and medium difficulty |
| **Medium** | 11–20 chars | Val | Select ~3 images | Various | Text where regime shift begins |
| **Medium** | 11–20 chars | Test | Select ~3 images | Various | Test set equivalent |
| **Medium** | 21–30 chars | Train | Select ~3 images | Various | Longer medium — check when diagonal breaks |
| **Medium** | 21–30 chars | Test | Select ~3 images | Various | Unseen medium-length texts |
| **Long** | 31–60 chars | Val | Select ~3 images | Various | Deep into the diffuse attention regime |
| **Long** | 31–60 chars | Test | Select ~3 images | Various | Long line test cases |
| **V.Long** | 61+ chars | Test | `c04-110-00` | 75+ chars | "Become a success..." — extreme case |
| **V.Long** | 61+ chars | Test | `c04-116-02` | 88 chars | Longest in test set |
| **V.Long** | 61+ chars | Train | Select ~2 images | Various | Compare train vs test attention on long lines |

**Target**: ~30–40 images total, ~8–10 per length category, mixed across splits.

### 2.2 Models to Focus On

Based on the professor's feedback, focus the detailed comparison on **3 models**:

| Model | Registers | Why |
|-------|-----------|-----|
| **run_50** | 0 | Baseline — no registers |
| **run_51** | 2 | Best attention quality (professor confirmed) |
| **run_52** | 4 | Second-best; close to original paper recommendation |

Include run_53 (8) and run_54 (16) as supplementary — show they offer diminishing returns.

### 2.3 What to Measure

| Metric | What It Tells Us | How to Compute |
|--------|-----------------|----------------|
| **Diagonality (Spearman ρ)** | Does attention follow left-to-right reading order? | Spearman correlation between char index and peak patch position |
| **Peak Sharpness** | How focused is the attention on each character? | `max(attn) / mean(attn)` for each character's attention vector |
| **Entropy** | How spread out is the attention distribution? | `-Σ p·log(p)` over patch positions |
| **CER per image** | Recognition accuracy | Edit distance / GT length |
| **Disagreement cases** | Where do register models differ from baseline? | Images where Reg-0 CER ≠ Reg-2 CER |
| **Attention–error correlation** | Does poor attention predict poor recognition? | Scatter plot of diagonality vs. CER |

### 2.4 Visualization Outputs to Produce

| Output | Models | Images | Purpose |
|--------|--------|--------|---------|
| **Side-by-side alignment** (Reg-0 vs Reg-2 vs Reg-4) | 3 | Curated 30–40 | Core comparison — professor can directly see the difference |
| **Attention carpet** per model | 3 | Curated 30–40 | Per-character spatial focus |
| **Diagonality vs. Text Length** scatter plot | All 5 | Curated 30–40 | Quantify the regime shift |
| **CER vs. Registers** bar chart by length bucket | All 5 | Full test set (2,915) | Statistical backing |
| **Error case studies** | 3 (focused) | 5–10 disagreement cases | Specific attention-error linkage |

---

## 3. Hypotheses to Test

Based on the professor's feedback and current findings:

### H1: Reg-2 provides the optimal attention quality for character-level interpretability
- **Measure**: Diagonality ρ, peak sharpness, entropy — stratified by text length  
- **Expectation**: Reg-2 > Reg-4 > Reg-0 for short/medium texts; all similar for long texts

### H2: The attention quality advantage of registers diminishes beyond 2–4 registers
- **Measure**: Diagonality ρ as a function of register count  
- **Expectation**: Diminishing (or even negative) returns beyond 4 registers, consistent with the original paper

### H3: The regime shift from spatial to contextual attention occurs at ~15 characters
- **Measure**: Diagonality ρ vs. text length  
- **Expectation**: Sharp drop in ρ between 10–20 characters for all models

### H4: Registers primarily act as attention sinks, not accuracy boosters
- **Measure**: Compare register token attention mass vs. patch token attention mass  
- **Expectation**: Registers absorb a disproportionate share of attention (especially Reg-8/16), leaving less signal for character-level routing

### H5: CER improvements from registers (if any) come from better spatial localization at medium text lengths
- **Measure**: Per-length-bucket CER for Reg-0 vs Reg-2 vs Reg-4  
- **Expectation**: Short = all 0% CER; Long = all ~10% CER; Medium = Reg-2/4 slightly better

---

## 4. Execution Script — Curated Image Selection

To select the curated images, run:

```bash
# Step 1: Extract text lengths from each split
for split in train val test; do
  echo "=== $split ==="
  awk '{
    key=$1; $1=""; text=substr($0,2);
    len=length(text);
    if (len <= 10) bucket="short";
    else if (len <= 30) bucket="medium";
    else if (len <= 60) bucket="long";
    else bucket="vlong";
    print bucket "\t" len "\t" key "\t" text
  }' data/IAM/processed_lines/$split/gt.txt | sort -t$'\t' -k2 -n
done
```

```bash
# Step 2: Sample ~3 images per bucket per split
for split in train val test; do
  for bucket in short medium long vlong; do
    echo "=== $split / $bucket ==="
    awk -v b="$bucket" '{
      key=$1; $1=""; text=substr($0,2); len=length(text);
      if (len <= 10) bk="short";
      else if (len <= 30) bk="medium";
      else if (len <= 60) bk="long";
      else bk="vlong";
      if (bk == b) print len "\t" key "\t" text
    }' data/IAM/processed_lines/$split/gt.txt | shuf | head -3
  done
done
```

```bash
# Step 3: Copy selected images to a curated directory
mkdir -p notebook/curated_images
# (copy selected images and create gt.txt)
```

```bash
# Step 4: Run character attention mapping on curated set
python scripts/postprocessing/character_attention_mapping.py \
    configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts_v2.yaml \
    --model-path saved_models/experiments/run_50/model.pt \
    --model-path saved_models/experiments/run_51/model.pt \
    --model-path saved_models/experiments/run_52/model.pt \
    --layer last --alpha 0.45 --dpi 200 --gamma 3.0 \
    --save-dir visualizations/curated_attention_analysis \
    -- notebook/curated_images/
```

---

## 5. Summary of Key Findings So Far

| Finding | Evidence | Significance |
|---------|----------|-------------|
| Reg-2 produces sharpest attention peaks | Peak sharpness = 12.2 (highest); Diag ρ = 0.55 for short words | Confirms professor's visual assessment quantitatively |
| Reg-4 is a strong runner-up | Diag ρ = 0.53 for short words; Test CER 6.07% (close to best) | Good balance of attention quality and accuracy |
| 0 registers = worst attention coherence | Diag ρ = 0.28 for short words (vs. 0.55 for Reg-2) | Clear evidence that registers improve attention routing |
| All models perfect on short words | 0% CER for all models on words ≤ 11 chars | The task difficulty is entirely in long sequences |
| Long-line attention is diffuse for all | Diag ρ < 0.20 for all models on 56+ char lines | Registers don't help with long-line spatial attention |
| Register count vs accuracy: diminishing returns | CER: {0: 6.26, 2: 6.12, 4: 6.07, 8: 6.17, 16: 5.97} | Non-monotonic; Reg-8 worse than Reg-4; Reg-16 only marginally better |
| Regime shift at ~15 chars | Scatter plot shows sharp diagonality drop at 10–15 chars | Fundamental architecture limitation, not register-fixable |

---

## 6. Timeline

| Step | Task | Est. Time |
|------|------|-----------|
| 1 | Curate the stratified image set (30–40 images) | 1 hour |
| 2 | Run attention mapping on Reg-0, Reg-2, Reg-4 | 2–3 hours (GPU) |
| 3 | Compute quantitative metrics (diagonality, entropy, sharpness) | 1–2 hours |
| 4 | Generate scatter plots and bar charts | 1 hour |
| 5 | Identify and analyze 5–10 disagreement cases | 1–2 hours |
| 6 | Write up findings document with figures | 2–3 hours |
| **Total** | | **~8–12 hours** |
