#!/usr/bin/env python3
"""
Curate and organize publication-ready figures for the report.

Copies the best figures into a single outputs/report_figures/ directory
with standardized naming for easy inclusion in LaTeX.

Usage:
    python scripts/postprocessing/curate_figures.py
"""

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUTPUT_DIR = "outputs/report_figures"

# Mapping: target name -> source path (prefer PDF for vector quality)
FIGURE_MAP = {
    # Architecture comparison & quantitative results
    "fig_register_quantitative.pdf": "outputs/pub_figures/fig3_register_quantitative.pdf",
    "fig_register_quantitative.png": "outputs/pub_figures/fig3_register_quantitative.png",

    # Attention map visualizations
    "fig_char_attention_grid.pdf": "outputs/pub_figures/fig5_char_attention_run_84.pdf",
    "fig_char_attention_grid.png": "outputs/pub_figures/fig5_char_attention_run_84.png",
    "fig_ctc_self_attention.pdf": "outputs/pub_figures/fig5_ctc_self_attention_4reg.pdf",
    "fig_ctc_self_attention.png": "outputs/pub_figures/fig5_ctc_self_attention_4reg.png",
    "fig_register_comparison.pdf": "outputs/pub_figures/fig5_register_comparison_attn_talks.pdf",
    "fig_register_comparison.png": "outputs/pub_figures/fig5_register_comparison_attn_talks.png",

    # GradCAM
    "fig_gradcam_attention.pdf": "outputs/pub_figures/fig2_grad_attention_a01-038-12.pdf",
    "fig_gradcam_attention.png": "outputs/pub_figures/fig2_grad_attention_a01-038-12.png",
    "fig_gradient_saliency.pdf": "outputs/pub_figures/fig5_gradient_saliency_4reg.pdf",
    "fig_gradient_saliency.png": "outputs/pub_figures/fig5_gradient_saliency_4reg.png",

    # Attention rollout + GradCAM multi-sample
    "fig_gradcam_summary.pdf": "outputs/viz_gradcam/char_gradcam_summary.pdf",
    "fig_attn_rollout_sample1.pdf": "outputs/viz_gradcam/attn_rollout_a01-038-12.pdf",
    "fig_gradcam_sample1.pdf": "outputs/viz_gradcam/char_gradcam_a01-038-12.pdf",

    # Register attention patterns
    "fig_register_attention.pdf": "outputs/attention_maps/fig4_register_attention.pdf",
    "fig_per_head_attention.pdf": "outputs/attention_maps/fig2_per_head_attention.pdf",
    "fig_token_norms.pdf": "outputs/attention_maps/fig3_token_norms.pdf",
    "fig_layer_comparison.pdf": "outputs/attention_maps/fig5_layer_comparison.pdf",

    # CTC peak attention
    "fig_attn_at_ctc_peaks.png": "outputs/pub_figures/attn_at_ctc_peaks.png",
    "fig_ctc_posterior.png": "outputs/pub_figures/ctc_posterior.png",
    "fig_best_head_comparison.png": "outputs/pub_figures/best_head_comparison.png",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    copied = 0
    missing = 0

    for target_name, source_path in sorted(FIGURE_MAP.items()):
        target_path = os.path.join(OUTPUT_DIR, target_name)
        if os.path.isfile(source_path):
            shutil.copy2(source_path, target_path)
            size_kb = os.path.getsize(target_path) / 1024
            print(f"  ✓ {target_name:<45} ({size_kb:.0f} KB)")
            copied += 1
        else:
            print(f"  ✗ {target_name:<45} — source missing: {source_path}")
            missing += 1

    # Also copy a few BM-style attention samples
    bm_dir = "outputs/htrvt_bm_attention"
    if os.path.isdir(bm_dir):
        bm_files = sorted([f for f in os.listdir(bm_dir) if f.endswith("_R4_chars.png")])[:3]
        for bf in bm_files:
            src = os.path.join(bm_dir, bf)
            tgt = os.path.join(OUTPUT_DIR, f"fig_bm_htrvt_{bf}")
            shutil.copy2(src, tgt)
            print(f"  ✓ fig_bm_htrvt_{bf}")
            copied += 1

    # Copy BM-style per-char samples from beyond_memorization
    bm_comp = "outputs/beyond_memorization"
    for subdir in ["noChange", "charIndex_0", "charIndex_2"]:
        src_dir = os.path.join(bm_comp, subdir)
        if os.path.isdir(src_dir):
            pngs = sorted([f for f in os.listdir(src_dir) if f.endswith(".png")])[:2]
            for pf in pngs:
                src = os.path.join(src_dir, pf)
                tgt = os.path.join(OUTPUT_DIR, f"fig_bm_{subdir}_{pf}")
                shutil.copy2(src, tgt)
                print(f"  ✓ fig_bm_{subdir}_{pf}")
                copied += 1

    print(f"\n{copied} figures copied, {missing} missing")
    print(f"Output: {OUTPUT_DIR}/")

    # Write index file
    index_path = os.path.join(OUTPUT_DIR, "FIGURE_INDEX.txt")
    with open(index_path, "w") as f:
        f.write("Report Figures — Curated Set\n")
        f.write("=" * 60 + "\n\n")
        f.write("Usage in LaTeX:\n")
        f.write("  \\includegraphics[width=\\linewidth]{figures/fig_NAME.pdf}\n\n")
        f.write("Figures:\n")
        for name in sorted(os.listdir(OUTPUT_DIR)):
            if name == "FIGURE_INDEX.txt":
                continue
            size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, name)) / 1024
            f.write(f"  {name:<50} {size_kb:>8.0f} KB\n")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
