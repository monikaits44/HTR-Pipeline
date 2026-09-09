#!/usr/bin/env python3
"""
Aggregate experiment results across all runs into a unified comparison table.

Produces:
  - outputs/tables/unified_results.csv   (machine-readable)
  - outputs/tables/unified_results.tex   (LaTeX tabular for report)
  - stdout: formatted table

Usage:
    python scripts/postprocessing/aggregate_results.py
    python scripts/postprocessing/aggregate_results.py --runs-dir saved_models/experiments
"""

import argparse, csv, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate experiment results")
    p.add_argument("--runs-dir", default="saved_models/experiments",
                   help="Directory containing run_* folders")
    p.add_argument("--output-dir", default="outputs/tables",
                   help="Where to write output files")
    return p.parse_args()


def parse_run(run_dir):
    """Parse a single run directory. Returns dict or None if invalid."""
    config_path = os.path.join(run_dir, "config.json")
    results_path = os.path.join(run_dir, "results.csv")

    if not os.path.exists(config_path) or not os.path.exists(results_path):
        return None

    # Parse config
    with open(config_path) as f:
        config = json.load(f)

    arch_cfg = config.get("arch", {})
    arch_type = arch_cfg.get("type", "unknown")
    num_registers = arch_cfg.get("num_registers", "N/A")
    use_cnn_stem = arch_cfg.get("use_cnn_stem", False)
    dim = arch_cfg.get("dim", arch_cfg.get("vit_embed_dim", "?"))
    depth = arch_cfg.get("depth", arch_cfg.get("vit_depth", "?"))
    head_type = arch_cfg.get("head_type", "?")
    rnn_layers = arch_cfg.get("rnn_layers", "?")
    rnn_hidden = arch_cfg.get("rnn_hidden_size", "?")
    rnn_type = arch_cfg.get("rnn_type", "lstm")

    swa_cfg = config.get("swa", {})
    swa_enabled = swa_cfg.get("enabled", False)
    data_mode = config.get("data", {}).get("mode", "?")
    augmentation = config.get("train", {}).get("augmentation", "?")

    # Parse results CSV — get last epoch (best trained) and best val CER epoch
    rows = []
    with open(results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    last_row = rows[-1]
    num_epochs = len(rows)

    # Find best validation CER epoch
    best_val_row = min(rows, key=lambda r: float(r.get("val/cer", 999)))

    # Extract params from any row
    params = int(float(last_row.get("model/params", 0)))

    return {
        "run": os.path.basename(run_dir),
        "arch_type": arch_type,
        "num_registers": num_registers,
        "use_cnn_stem": use_cnn_stem,
        "dim": dim,
        "depth": depth,
        "head_type": head_type,
        "rnn_layers": rnn_layers,
        "rnn_hidden": rnn_hidden,
        "rnn_type": rnn_type,
        "swa": swa_enabled,
        "data_mode": data_mode,
        "augmentation": augmentation,
        "params": params,
        "epochs": num_epochs,
        # Last epoch metrics
        "last_train_loss": float(last_row.get("train/ctc_loss", 0)),
        "last_val_cer": float(last_row.get("val/cer", 999)),
        "last_val_wer": float(last_row.get("val/wer", 999)),
        "last_test_cer": float(last_row.get("test/cer", 999)),
        "last_test_wer": float(last_row.get("test/wer", 999)),
        # Best validation CER epoch metrics
        "best_epoch": int(best_val_row.get("epoch", 0)),
        "best_val_cer": float(best_val_row.get("val/cer", 999)),
        "best_val_wer": float(best_val_row.get("val/wer", 999)),
        "best_test_cer": float(best_val_row.get("test/cer", 999)),
        "best_test_wer": float(best_val_row.get("test/wer", 999)),
    }


def format_pct(val):
    """Format as percentage string."""
    return f"{val * 100:.2f}"


def format_params(params):
    """Format parameter count as human-readable."""
    if params >= 1e6:
        return f"{params / 1e6:.1f}M"
    elif params >= 1e3:
        return f"{params / 1e3:.0f}K"
    return str(params)


def pick_canonical_runs(results):
    """
    Pick the best run for each architecture+register configuration.
    For duplicate configs, keep the one with best validation CER.
    """
    buckets = {}
    for r in results:
        key = (r["arch_type"], str(r["num_registers"]))
        if key not in buckets or r["best_val_cer"] < buckets[key]["best_val_cer"]:
            buckets[key] = r
    return list(buckets.values())


def sort_results(results):
    """Sort: CNN-RNN first, then vit_rgts by register count, then others."""
    type_order = {"cnn_rnn": 0, "vit_rgts": 1, "torchvision_vit": 2, "trocr": 3}

    def sort_key(r):
        t = type_order.get(r["arch_type"], 99)
        reg = r["num_registers"] if isinstance(r["num_registers"], int) else 999
        return (t, reg)

    return sorted(results, key=sort_key)


def write_csv(results, output_path):
    """Write unified results CSV."""
    fields = [
        "run", "arch_type", "num_registers", "use_cnn_stem", "head_type",
        "swa", "data_mode", "dim", "depth", "params",
        "epochs", "best_epoch",
        "best_val_cer", "best_val_wer", "best_test_cer", "best_test_wer",
        "last_val_cer", "last_val_wer", "last_test_cer", "last_test_wer",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  CSV: {output_path}")


def write_latex(results, output_path):
    """Write LaTeX tabular for report."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Unified comparison of all trained models on IAM (Aachen split). "
                 r"CER and WER reported at the best validation epoch.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Architecture & Registers & Params & Epoch & Val CER (\%) & Test CER (\%) & Test WER (\%) \\")
    lines.append(r"\midrule")

    prev_arch = None
    for r in results:
        arch_name = {
            "cnn_rnn": "CNN-RNN",
            "vit_rgts": "ViT-RGTS v2",
            "torchvision_vit": "TorchVision ViT",
            "trocr": "TrOCR",
        }.get(r["arch_type"], r["arch_type"])

        # Add separator between architecture groups
        if prev_arch is not None and r["arch_type"] != prev_arch:
            lines.append(r"\midrule")
        prev_arch = r["arch_type"]

        reg_str = str(r["num_registers"]) if r["num_registers"] != "N/A" else "--"
        params_str = format_params(r["params"])

        line = (f"{arch_name} & {reg_str} & {params_str} & "
                f"{r['best_epoch']} & "
                f"{format_pct(r['best_val_cer'])} & "
                f"{format_pct(r['best_test_cer'])} & "
                f"{format_pct(r['best_test_wer'])} \\\\")
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX: {output_path}")


def print_table(results):
    """Print formatted table to stdout."""
    header = f"{'Run':<10} {'Architecture':<18} {'Reg':>4} {'Params':>8} {'Ep':>4} " \
             f"{'Val CER':>8} {'Test CER':>9} {'Test WER':>9}"
    print(header)
    print("-" * len(header))

    prev_arch = None
    for r in results:
        if prev_arch is not None and r["arch_type"] != prev_arch:
            print("-" * len(header))
        prev_arch = r["arch_type"]

        reg_str = str(r["num_registers"]) if r["num_registers"] != "N/A" else "--"
        print(f"{r['run']:<10} {r['arch_type']:<18} {reg_str:>4} "
              f"{format_params(r['params']):>8} {r['best_epoch']:>4} "
              f"{format_pct(r['best_val_cer']):>7}% "
              f"{format_pct(r['best_test_cer']):>8}% "
              f"{format_pct(r['best_test_wer']):>8}%")


def write_register_sweep_latex(results, output_path):
    """LaTeX table: ViT-RGTS v2 register sweep (0/2/4/8/16) ± SWA."""
    vit_runs = [r for r in results
                if r["arch_type"] == "vit_rgts" and r["use_cnn_stem"]
                and r["epochs"] >= 70 and r["best_test_cer"] < 0.5]
    # Group by (registers, swa) — keep best per group
    buckets = {}
    for r in vit_runs:
        key = (r["num_registers"], r["swa"])
        if key not in buckets or r["best_val_cer"] < buckets[key]["best_val_cer"]:
            buckets[key] = r
    rows = sorted(buckets.values(), key=lambda r: (r["num_registers"], r["swa"]))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{ViT-RGTS v2 register sweep on IAM (Aachen split). "
                 r"CER/WER at best validation epoch.}")
    lines.append(r"\label{tab:register_sweep}")
    lines.append(r"\begin{tabular}{rrcccc}")
    lines.append(r"\toprule")
    lines.append(r"Registers & SWA & Best Ep. & Val CER (\%) & Test CER (\%) & Test WER (\%) \\")
    lines.append(r"\midrule")
    for r in rows:
        swa_str = r"\checkmark" if r["swa"] else "--"
        lines.append(f"{r['num_registers']} & {swa_str} & {r['best_epoch']} & "
                     f"{format_pct(r['best_val_cer'])} & "
                     f"{format_pct(r['best_test_cer'])} & "
                     f"{format_pct(r['best_test_wer'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX (register sweep): {output_path}")


def write_ablation_latex(results, output_path):
    """LaTeX table: ablation studies (4-reg ViT-RGTS v2 variants)."""
    # Reference: standard 4-reg, 80ep, CNN stem, both head, dim=256, depth=6
    ref_params = 9.5e6
    abl_runs = [r for r in results
                if r["arch_type"] == "vit_rgts"
                and r.get("num_registers") == 4
                and not r["swa"]
                and r["epochs"] >= 30]

    def ablation_label(r):
        if not r["use_cnn_stem"]:
            return "No CNN stem"
        if r["head_type"] == "rnn":
            return "RNN head only"
        if r["head_type"] == "cnn":
            return "CNN head only"
        rnn_type = r.get("rnn_type", "lstm")
        if rnn_type != "?" and rnn_type == "gru":
            return "GRU head"
        if r.get("rnn_layers") == 1:
            return "1-layer BiLSTM"
        d = r.get("depth", 6)
        if d != "?" and int(d) == 4:
            return "Depth 4"
        if d != "?" and int(d) == 8:
            return "Depth 8"
        dm = r.get("dim", 256)
        if dm != "?" and int(dm) == 128:
            return "Dim 128"
        if dm != "?" and int(dm) == 512:
            return "Dim 512"
        p = r["params"]
        if p < 2e6:
            return "v1 (no stem, small)"
        if r["epochs"] > 100:
            return f"{r['epochs']} epochs"
        aug = r.get("augmentation", "cnn")
        if aug == "vit":
            return "ViT augmentation"
        if abs(p - ref_params) < 1e6 and r["epochs"] <= 80:
            return "Baseline (4-reg)"
        return f"dim={dm}, depth={d}"

    labeled = [(ablation_label(r), r) for r in abl_runs]
    # Deduplicate — keep best per label
    best_per = {}
    for label, r in labeled:
        if label not in best_per or r["best_val_cer"] < best_per[label]["best_val_cer"]:
            best_per[label] = r
            best_per[label]["_label"] = label
    rows = sorted(best_per.values(), key=lambda r: r["best_test_cer"])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study for ViT-RGTS v2 with 4 registers on IAM.}")
    lines.append(r"\label{tab:ablations}")
    lines.append(r"\begin{tabular}{lrcccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & Params & Epochs & Val CER (\%) & Test CER (\%) & Test WER (\%) \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(f"{r['_label']} & {format_params(r['params'])} & {r['epochs']} & "
                     f"{format_pct(r['best_val_cer'])} & "
                     f"{format_pct(r['best_test_cer'])} & "
                     f"{format_pct(r['best_test_wer'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX (ablations): {output_path}")


def write_architecture_comparison_latex(results, output_path):
    """LaTeX table: best result per architecture."""
    # Pick best run per architecture
    best_per_arch = {}
    for r in results:
        arch = r["arch_type"]
        if arch not in best_per_arch or r["best_test_cer"] < best_per_arch[arch]["best_test_cer"]:
            best_per_arch[arch] = r
    arch_order = ["cnn_rnn", "vit_rgts", "htrvt", "torchvision_vit", "trocr"]
    arch_names = {
        "cnn_rnn": "CNN-RNN (baseline)",
        "vit_rgts": "ViT-RGTS v2",
        "htrvt": "HTR-VT (READ2016)",
        "torchvision_vit": "TorchVision ViT-B/16",
        "trocr": "TrOCR-base",
    }
    rows = [best_per_arch[a] for a in arch_order if a in best_per_arch]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Best result per architecture on IAM (Aachen split).}")
    lines.append(r"\label{tab:arch_comparison}")
    lines.append(r"\begin{tabular}{lrrcccc}")
    lines.append(r"\toprule")
    lines.append(r"Architecture & Params & Reg. & SWA & Best Ep. & Test CER (\%) & Test WER (\%) \\")
    lines.append(r"\midrule")
    for r in rows:
        name = arch_names.get(r["arch_type"], r["arch_type"])
        reg_str = str(r["num_registers"]) if r["num_registers"] != "N/A" else "--"
        swa_str = r"\checkmark" if r["swa"] else "--"
        lines.append(f"{name} & {format_params(r['params'])} & {reg_str} & {swa_str} & "
                     f"{r['best_epoch']} & "
                     f"{format_pct(r['best_test_cer'])} & "
                     f"{format_pct(r['best_test_wer'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX (architecture comparison): {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all runs
    runs_dir = args.runs_dir
    all_results = []

    for entry in sorted(os.listdir(runs_dir)):
        if not entry.startswith("run_"):
            continue
        run_path = os.path.join(runs_dir, entry)
        if not os.path.isdir(run_path):
            continue

        result = parse_run(run_path)
        if result:
            all_results.append(result)

    print(f"Found {len(all_results)} valid runs\n")

    # Pick canonical (best per config) and sort
    canonical = pick_canonical_runs(all_results)
    canonical = sort_results(canonical)

    print("=" * 90)
    print("CANONICAL RUNS (best per architecture + register count)")
    print("=" * 90)
    print_table(canonical)

    # Write outputs
    print(f"\nOutputs:")
    write_csv(canonical, os.path.join(args.output_dir, "unified_results.csv"))
    write_latex(canonical, os.path.join(args.output_dir, "unified_results.tex"))

    # Grouped LaTeX tables for report
    write_register_sweep_latex(all_results, os.path.join(args.output_dir, "register_sweep.tex"))
    write_ablation_latex(all_results, os.path.join(args.output_dir, "ablations.tex"))
    write_architecture_comparison_latex(all_results, os.path.join(args.output_dir, "architecture_comparison.tex"))

    # Also write the FULL results (all runs, not just canonical)
    all_sorted = sort_results(all_results)
    write_csv(all_sorted, os.path.join(args.output_dir, "all_runs.csv"))

    print(f"\n{'=' * 90}")
    print("ALL RUNS (sorted by architecture + register count)")
    print(f"{'=' * 90}")
    print_table(all_sorted)


if __name__ == "__main__":
    main()
