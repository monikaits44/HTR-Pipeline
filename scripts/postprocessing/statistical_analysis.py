#!/usr/bin/env python3
"""
Statistical significance testing for HTR experiment comparisons.

Performs paired bootstrap tests on per-sample CER between experiment runs.
Uses evaluation_details.csv from each run directory.

Produces:
  - outputs/tables/statistical_tests.csv  (pairwise p-values)
  - outputs/tables/statistical_tests.tex  (LaTeX table)
  - stdout: formatted comparison table

Usage:
    python scripts/postprocessing/statistical_analysis.py
    python scripts/postprocessing/statistical_analysis.py --runs 107 115 123 116 122 135
    python scripts/postprocessing/statistical_analysis.py --reference 107 --comparisons 116 105 123 122 135
"""

import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args():
    p = argparse.ArgumentParser(description="Statistical significance testing")
    p.add_argument("--runs-dir", default="saved_models/experiments")
    p.add_argument("--output-dir", default="outputs/tables")
    p.add_argument("--runs", nargs="+", type=int, default=None,
                   help="Run IDs to compare (all pairs). Default: key runs.")
    p.add_argument("--reference", type=int, default=None,
                   help="Reference run to compare all others against.")
    p.add_argument("--comparisons", nargs="+", type=int, default=None,
                   help="Runs to compare against reference.")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_per_sample_cer(run_dir, dataset="test"):
    """Load per-sample CER from evaluation_details.csv at the last epoch."""
    csv_path = os.path.join(run_dir, "evaluation_details.csv")
    if not os.path.isfile(csv_path):
        return None

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["dataset"] == dataset]

    if not rows:
        return None

    max_epoch = max(int(r["epoch"]) for r in rows)
    last_epoch_rows = [r for r in rows if int(r["epoch"]) == max_epoch]

    # Deduplicate by sample_idx (keep last = SWA model if available)
    by_idx = {}
    for r in last_epoch_rows:
        by_idx[int(r["sample_idx"])] = float(r["sample_cer"])

    n = max(by_idx.keys()) + 1
    cer_array = np.zeros(n)
    for idx, cer in by_idx.items():
        cer_array[idx] = cer

    return cer_array


def paired_bootstrap_test(cer_a, cer_b, n_bootstrap=10000, rng=None):
    """
    Paired bootstrap test: is the mean CER of A significantly different from B?
    Returns (delta_mean, p_value, ci_lower, ci_upper).
    delta = mean(A) - mean(B); positive means A is worse.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(cer_a)
    assert len(cer_b) == n

    diffs = cer_a - cer_b
    observed_delta = diffs.mean()

    # Bootstrap CI for the difference
    boot_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = diffs[idx].mean()

    ci_lower = np.percentile(boot_deltas, 2.5)
    ci_upper = np.percentile(boot_deltas, 97.5)

    # P-value: bootstrap under H0 (shift diffs to have mean 0)
    shifted = diffs - diffs.mean()
    null_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        null_deltas[i] = shifted[idx].mean()
    p_value = np.mean(np.abs(null_deltas) >= abs(observed_delta))

    return observed_delta, p_value, ci_lower, ci_upper


def get_run_label(run_dir):
    """Get a short label for a run from its config."""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        return os.path.basename(run_dir)
    with open(config_path) as f:
        c = json.load(f)
    arch = c.get("arch", {}).get("type", "?")
    regs = c.get("arch", {}).get("num_registers", "")
    swa = c.get("swa", {}).get("enabled", False)
    epochs = c.get("train", {}).get("num_epochs", "?")

    names = {"cnn_rnn": "CNN-RNN", "vit_rgts": "ViT-RGTS", "torchvision_vit": "ViT-B/16",
             "trocr": "TrOCR", "htrvt": "HTR-VT"}
    name = names.get(arch, arch)
    parts = [name]
    if regs != "" and regs != "N/A":
        parts.append(f"R{regs}")
    if swa:
        parts.append("SWA")
    if epochs != "?" and int(epochs) != 80:
        parts.append(f"{epochs}ep")
    return " ".join(parts)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Determine which runs to compare
    if args.runs is not None:
        run_ids = args.runs
    elif args.reference is not None and args.comparisons is not None:
        run_ids = [args.reference] + args.comparisons
    else:
        # Default: key representative runs
        run_ids = [107, 115, 116, 105, 118, 123, 122, 135, 121, 142]

    # Load per-sample CER for each run
    run_data = {}
    for rid in run_ids:
        run_dir = os.path.join(args.runs_dir, f"run_{rid}")
        cer = load_per_sample_cer(run_dir)
        if cer is not None:
            label = get_run_label(run_dir)
            run_data[rid] = {"cer": cer, "label": label, "mean": cer.mean()}
            print(f"  run_{rid} ({label}): mean CER = {cer.mean()*100:.2f}%, n={len(cer)}")
        else:
            print(f"  run_{rid}: no evaluation_details.csv — skipping")

    if len(run_data) < 2:
        print("Need at least 2 runs with per-sample data.")
        return

    # Determine reference run
    if args.reference is not None and args.reference in run_data:
        ref_id = args.reference
    else:
        ref_id = min(run_data.keys(), key=lambda k: run_data[k]["mean"])
    print(f"\nReference: run_{ref_id} ({run_data[ref_id]['label']})\n")

    # Pairwise comparisons
    comparisons = []
    other_ids = sorted(k for k in run_data if k != ref_id)

    print(f"{'Run':<8} {'Label':<25} {'CER%':>6} {'Δ CER%':>8} {'95% CI':>18} {'p-value':>10} {'Sig.':>5}")
    print("-" * 85)

    for oid in other_ids:
        cer_ref = run_data[ref_id]["cer"]
        cer_other = run_data[oid]["cer"]

        # Ensure same length (trim to min)
        n = min(len(cer_ref), len(cer_other))
        delta, p_val, ci_lo, ci_hi = paired_bootstrap_test(
            cer_other[:n], cer_ref[:n], args.n_bootstrap, rng
        )
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        comparisons.append({
            "reference": f"run_{ref_id}",
            "comparison": f"run_{oid}",
            "ref_label": run_data[ref_id]["label"],
            "comp_label": run_data[oid]["label"],
            "ref_cer": run_data[ref_id]["mean"],
            "comp_cer": run_data[oid]["mean"],
            "delta": delta,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_value": p_val,
            "significance": sig,
        })

        print(f"run_{oid:<4} {run_data[oid]['label']:<25} {run_data[oid]['mean']*100:>5.2f} "
              f"{delta*100:>+7.2f} [{ci_lo*100:>+6.2f}, {ci_hi*100:>+6.2f}] "
              f"{p_val:>9.4f} {sig:>5}")

    # Write CSV
    csv_path = os.path.join(args.output_dir, "statistical_tests.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "reference", "comparison", "ref_label", "comp_label",
            "ref_cer", "comp_cer", "delta", "ci_lower", "ci_upper",
            "p_value", "significance"
        ])
        writer.writeheader()
        writer.writerows(comparisons)
    print(f"\nCSV: {csv_path}")

    # Write LaTeX
    tex_path = os.path.join(args.output_dir, "statistical_tests.tex")
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Paired bootstrap significance test (10K resamples) against "
                 f"run\\_{{\\,{ref_id}}} ({run_data[ref_id]['label']}). "
                 r"$\Delta$ CER = comparison $-$ reference.}")
    lines.append(r"\label{tab:significance}")
    lines.append(r"\begin{tabular}{lccccl}")
    lines.append(r"\toprule")
    lines.append(r"Model & CER (\%) & $\Delta$ CER & 95\% CI & $p$-value & Sig. \\")
    lines.append(r"\midrule")
    ref_line = (f"{run_data[ref_id]['label']} (ref.) & "
                f"{run_data[ref_id]['mean']*100:.2f} & -- & -- & -- & -- \\\\")
    lines.append(ref_line)
    lines.append(r"\midrule")
    for c in comparisons:
        sig_str = {
            "***": "$^{***}$", "**": "$^{**}$", "*": "$^{*}$", "n.s.": "n.s."
        }.get(c["significance"], c["significance"])
        lines.append(
            f"{c['comp_label']} & {c['comp_cer']*100:.2f} & "
            f"{c['delta']*100:+.2f} & "
            f"[{c['ci_lower']*100:+.2f}, {c['ci_upper']*100:+.2f}] & "
            f"{c['p_value']:.4f} & {sig_str} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{6}{l}{\footnotesize $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$} \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
