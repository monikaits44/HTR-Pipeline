#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║          HTR Pipeline — Experiment Tracking Dashboard                    ║
║          Handwritten Text Recognition on IAM Dataset                     ║
║                                                                          ║
║  Architectures: CNN-RNN · ViT-RGTS · TorchVision ViT · TrOCR            ║
║  Metrics: CER · WER · CTC Loss · Attention Visualizations               ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    cd /home/hpc/iwi5/iwi5369h/HTR-Pipeline
    streamlit run dashboard/app.py --server.port 8501

Features:
    1. Overview         — High-level comparison of all experiments
    2. Training Curves  — Epoch-by-epoch CTC Loss, CER, WER, LR
    3. Architecture Lab — Compare ViT register variants & all architectures
    4. Run Explorer     — Deep-dive into any single run's config & per-sample analysis
    5. Attention Maps   — Character attention, register attention, Grad-CAM, t-SNE
    6. Dataset EDA      — IAM dataset statistics & character distributions
    7. Live Monitoring  — Watch running SLURM jobs in real time
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import glob
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HTR Pipeline Dashboard",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "saved_models" / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "output"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
LOGS_DIR = PROJECT_ROOT / "experiments_execution" / "logs"

ARCH_COLORS = {
    "cnn_rnn": "#2E86C1",
    "vit_rgts": "#E74C3C",
    "torchvision_vit": "#27AE60",
    "trocr": "#8E44AD",
}

ARCH_LABELS = {
    "cnn_rnn": "CNN-RNN",
    "vit_rgts": "ViT-RGTS",
    "torchvision_vit": "TorchVision ViT",
    "trocr": "TrOCR",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def discover_runs():
    """Scan saved_models/experiments/ for all run_* directories."""
    runs = []
    if not EXPERIMENTS_DIR.exists():
        return runs
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("run_"):
            run_num = int(d.name.split("_")[1])
            config_path = d / "config.json"
            results_path = d / "results.csv"
            if config_path.exists():
                runs.append({
                    "run_name": d.name,
                    "run_num": run_num,
                    "path": d,
                    "has_results": results_path.exists(),
                })
    runs.sort(key=lambda x: x["run_num"])
    return runs


@st.cache_data(ttl=60)
def load_config(run_path_str: str) -> dict:
    """Load config.json for a run."""
    config_path = Path(run_path_str) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=60)
def load_results(run_path_str: str) -> pd.DataFrame:
    """Load results.csv for a run, handling malformed CSVs gracefully."""
    results_path = Path(run_path_str) / "results.csv"
    if not results_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(results_path, on_bad_lines="skip")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_evaluation_details(run_path_str: str) -> pd.DataFrame:
    """Load evaluation_details.csv for a run."""
    eval_path = Path(run_path_str) / "evaluation_details.csv"
    if not eval_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(eval_path, on_bad_lines="skip")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_eda_summary() -> dict:
    """Load EDA summary JSON."""
    eda_path = OUTPUT_DIR / "data_analysis" / "eda_summary.json"
    if eda_path.exists():
        with open(eda_path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=60)
def build_overview_table():
    """Build a summary DataFrame across all runs."""
    runs = discover_runs()
    rows = []
    for run_info in runs:
        cfg = load_config(str(run_info["path"]))
        arch_cfg = cfg.get("arch", {})
        arch_type = arch_cfg.get("type", "unknown")

        row = {
            "Run": run_info["run_name"],
            "Run #": run_info["run_num"],
            "Architecture": ARCH_LABELS.get(arch_type, arch_type),
            "arch_type": arch_type,
            "Registers": arch_cfg.get("num_registers", "—"),
            "Epochs Configured": cfg.get("train", {}).get("num_epochs", "?"),
            "Batch Size": cfg.get("train", {}).get("batch_size", "?"),
            "LR": cfg.get("train", {}).get("lr", "?"),
            "Params": None,
            "Best Test CER": None,
            "Best Test WER": None,
            "Final CTC Loss": None,
            "Epochs Completed": 0,
        }

        # Architecture details
        if arch_type == "vit_rgts":
            row["Model Detail"] = f"dim={arch_cfg.get('dim')}, depth={arch_cfg.get('depth')}, heads={arch_cfg.get('heads')}"
        elif arch_type == "torchvision_vit":
            row["Model Detail"] = arch_cfg.get("model_name", "vit_b_16")
        elif arch_type == "trocr":
            row["Model Detail"] = arch_cfg.get("model_name", "trocr-base")
        elif arch_type == "cnn_rnn":
            row["Model Detail"] = f"head={arch_cfg.get('head_type','?')}, rnn={arch_cfg.get('rnn_layers','?')}L"
        else:
            row["Model Detail"] = ""

        if run_info["has_results"]:
            df = load_results(str(run_info["path"]))
            if not df.empty:
                row["Epochs Completed"] = len(df)
                if "test/cer" in df.columns:
                    row["Best Test CER"] = df["test/cer"].min()
                if "test/wer" in df.columns:
                    row["Best Test WER"] = df["test/wer"].min()
                if "train/ctc_loss" in df.columns:
                    row["Final CTC Loss"] = df["train/ctc_loss"].iloc[-1]
                if "model/params" in df.columns:
                    row["Params"] = int(df["model/params"].iloc[0])

        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt_cer(val):
    if val is None or pd.isna(val):
        return "—"
    return f"{val*100:.2f}%"

def fmt_wer(val):
    if val is None or pd.isna(val):
        return "—"
    return f"{val*100:.2f}%"

def fmt_params(val):
    if val is None or pd.isna(val):
        return "—"
    v = int(val)
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    elif v >= 1_000:
        return f"{v/1_000:.1f}K"
    return str(v)

def fmt_loss(val):
    if val is None or pd.isna(val):
        return "—"
    return f"{val:.2f}"

def safe_image(path):
    """Load and display an image if it exists."""
    p = Path(path)
    if p.exists():
        return Image.open(p)
    return None

def get_run_label(cfg, run_name):
    """Create a human-readable label for a run."""
    arch = cfg.get("arch", {})
    arch_type = arch.get("type", "unknown")
    label = ARCH_LABELS.get(arch_type, arch_type)
    if arch_type == "vit_rgts":
        nreg = arch.get("num_registers", 0)
        label += f" ({nreg}reg)"
    elif arch_type == "torchvision_vit":
        label += f" (pretrained)"
    elif arch_type == "trocr":
        frozen = "frozen" if arch.get("freeze_encoder", False) else "unfrozen"
        label += f" ({frozen})"
    return f"{run_name}: {label}"


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* Dashboard title */
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 0;
    }
    .dashboard-subtitle {
        font-size: 1.0rem;
        color: #6c757d;
        text-align: center;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; }
    .metric-card .label { font-size: 0.85rem; opacity: 0.9; }
    /* Architecture badges */
    .arch-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin-right: 0.3rem;
    }
    .arch-cnn_rnn { background: #2E86C1; }
    .arch-vit_rgts { background: #E74C3C; }
    .arch-torchvision_vit { background: #27AE60; }
    .arch-trocr { background: #8E44AD; }
    /* Info box */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #2E86C1;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
def tab_overview():
    overview_df = build_overview_table()

    if overview_df.empty:
        st.warning("No experiment runs found. Check `saved_models/experiments/`.")
        return

    # ── Top-level KPIs ────────────────────────────────────────────────
    best_row = overview_df.dropna(subset=["Best Test CER"])
    if not best_row.empty:
        best_row = best_row.loc[best_row["Best Test CER"].idxmin()]
    else:
        best_row = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Experiments", len(overview_df))
    c2.metric("Architectures", overview_df["Architecture"].nunique())
    if best_row is not None:
        c3.metric("Best CER", fmt_cer(best_row["Best Test CER"]), delta=best_row["Run"])
        c4.metric("Best WER", fmt_wer(best_row["Best Test WER"]), delta=best_row["Run"])
    else:
        c3.metric("Best CER", "—")
        c4.metric("Best WER", "—")

    st.markdown("---")

    # ── Per-Architecture Best ─────────────────────────────────────────
    st.subheader("🏆 Best Result per Architecture")

    arch_groups = overview_df.dropna(subset=["Best Test CER"]).groupby("Architecture")
    cols = st.columns(min(4, len(arch_groups)))

    for idx, (arch_name, grp) in enumerate(arch_groups):
        best = grp.loc[grp["Best Test CER"].idxmin()]
        arch_key = best["arch_type"]
        with cols[idx % len(cols)]:
            st.markdown(f"""
            <div class="metric-card" style="background: {ARCH_COLORS.get(arch_key, '#555')}">
                <div class="label">{arch_name}</div>
                <div class="value">{fmt_cer(best['Best Test CER'])}</div>
                <div class="label">CER · {best['Run']}</div>
                <div class="label">{fmt_params(best['Params'])} params</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Bar chart: CER by run ─────────────────────────────────────────
    st.subheader("📊 Test CER Across All Runs")
    chart_df = overview_df.dropna(subset=["Best Test CER"]).copy()
    chart_df["CER (%)"] = chart_df["Best Test CER"] * 100
    chart_df["color"] = chart_df["arch_type"].map(ARCH_COLORS)

    fig = px.bar(
        chart_df.sort_values("Run #"),
        x="Run",
        y="CER (%)",
        color="Architecture",
        color_discrete_map={v: ARCH_COLORS[k] for k, v in ARCH_LABELS.items()},
        hover_data=["Model Detail", "Params", "Epochs Completed"],
        title="Best Test CER per Experiment Run",
    )
    fig.update_layout(
        yaxis_title="Character Error Rate (%)",
        xaxis_title="Experiment Run",
        height=450,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Full table ────────────────────────────────────────────────────
    st.subheader("📋 All Experiments")

    display_df = overview_df[[
        "Run", "Architecture", "Model Detail", "Registers",
        "Epochs Completed", "Params", "Best Test CER", "Best Test WER",
        "Final CTC Loss", "LR", "Batch Size",
    ]].copy()

    display_df["Best Test CER"] = display_df["Best Test CER"].apply(fmt_cer)
    display_df["Best Test WER"] = display_df["Best Test WER"].apply(fmt_wer)
    display_df["Final CTC Loss"] = display_df["Final CTC Loss"].apply(fmt_loss)
    display_df["Params"] = display_df["Params"].apply(fmt_params)

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════════════
def tab_training_curves():
    runs = discover_runs()
    if not runs:
        st.warning("No experiment runs found.")
        return

    # ── Run selector ──────────────────────────────────────────────────
    run_options = {}
    for r in runs:
        cfg = load_config(str(r["path"]))
        label = get_run_label(cfg, r["run_name"])
        run_options[label] = r

    # Default: auto-select the 3 most recent runs (fully dynamic, no hardcoded run names)
    recent_labels = list(run_options.keys())[-3:]  # last 3 discovered (highest run numbers)
    selected = st.multiselect(
        "Select runs to compare",
        list(run_options.keys()),
        default=recent_labels,
    )

    if not selected:
        st.info("Select at least one run to visualize.")
        return

    # ── Metric selector ───────────────────────────────────────────────
    metric_options = {
        "CTC Loss": "train/ctc_loss",
        "Validation CER": "val/cer",
        "Test CER": "test/cer",
        "Validation WER": "val/wer",
        "Test WER": "test/wer",
        "Learning Rate": "lr",
    }

    c1, c2 = st.columns([2, 1])
    with c1:
        chosen_metrics = st.multiselect(
            "Metrics to plot",
            list(metric_options.keys()),
            default=["CTC Loss", "Test CER"],
        )
    with c2:
        log_y = st.checkbox("Log scale Y-axis", value=False)

    if not chosen_metrics:
        st.info("Select at least one metric.")
        return

    # ── Build plots ───────────────────────────────────────────────────
    for metric_name in chosen_metrics:
        col_name = metric_options[metric_name]
        fig = go.Figure()

        for label in selected:
            run_info = run_options[label]
            df = load_results(str(run_info["path"]))
            if df.empty or col_name not in df.columns:
                continue
            cfg = load_config(str(run_info["path"]))
            arch_type = cfg.get("arch", {}).get("type", "unknown")
            color = ARCH_COLORS.get(arch_type, "#888")

            fig.add_trace(go.Scatter(
                x=df["epoch"],
                y=df[col_name],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate=f"<b>{label}</b><br>Epoch: %{{x}}<br>{metric_name}: %{{y:.4f}}<extra></extra>",
            ))

        yaxis_type = "log" if log_y else "linear"
        fig.update_layout(
            title=f"📈 {metric_name} over Epochs",
            xaxis_title="Epoch",
            yaxis_title=metric_name,
            yaxis_type=yaxis_type,
            height=450,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Pre-generated training plots ──────────────────────────────────
    plots_dir = OUTPUT_DIR / "training_plots"
    if plots_dir.exists():
        pngs = sorted(plots_dir.glob("*.png"))
        if pngs:
            st.markdown("---")
            st.subheader("📊 Pre-Generated Training Plots")
            cols = st.columns(min(3, len(pngs)))
            for i, png in enumerate(pngs):
                with cols[i % 3]:
                    img = safe_image(png)
                    if img:
                        st.image(img, caption=png.stem.replace("_", " ").title(), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: ARCHITECTURE LAB
# ═══════════════════════════════════════════════════════════════════════════
def tab_architecture_lab():
    overview_df = build_overview_table()
    if overview_df.empty:
        st.warning("No experiments found.")
        return

    st.subheader("🔬 ViT-RGTS Register Comparison")

    vit_df = overview_df[overview_df["arch_type"] == "vit_rgts"].dropna(subset=["Best Test CER"]).copy()

    if vit_df.empty:
        st.info("No ViT-RGTS runs with results found.")
    else:
        vit_df["Registers"] = vit_df["Registers"].astype(str)
        vit_df["CER (%)"] = vit_df["Best Test CER"] * 100

        # Group by register count and show best per group
        st.markdown("**Best CER per register count:**")
        reg_best = vit_df.groupby("Registers").agg(
            Best_CER=("Best Test CER", "min"),
            Best_WER=("Best Test WER", "min"),
            Best_Run=("Run", "first"),
        ).reset_index()
        reg_best["Best CER (%)"] = reg_best["Best_CER"].apply(lambda x: f"{x*100:.2f}%")
        reg_best["Best WER (%)"] = reg_best["Best_WER"].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(reg_best[["Registers", "Best CER (%)", "Best WER (%)", "Best_Run"]],
                      use_container_width=True, hide_index=True)

        # Bar chart
        fig = px.bar(
            vit_df.sort_values("Registers"),
            x="Run", y="CER (%)",
            color="Registers",
            title="ViT-RGTS: CER by Register Count",
            hover_data=["Best Test WER", "Epochs Completed"],
        )
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Training curve overlay for register variants
        st.markdown("**Training convergence comparison:**")
        fig_conv = go.Figure()
        for _, row in vit_df.iterrows():
            run_path = EXPERIMENTS_DIR / row["Run"]
            df = load_results(str(run_path))
            if not df.empty and "test/cer" in df.columns:
                fig_conv.add_trace(go.Scatter(
                    x=df["epoch"], y=df["test/cer"] * 100,
                    mode="lines", name=f"{row['Run']} ({row['Registers']}reg)",
                    line=dict(width=2),
                ))
        fig_conv.update_layout(
            title="ViT-RGTS: Test CER Convergence by Register Count",
            xaxis_title="Epoch", yaxis_title="CER (%)",
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    st.markdown("---")

    # ── Cross-Architecture Comparison ─────────────────────────────────
    st.subheader("⚔️ Cross-Architecture Comparison")

    valid_df = overview_df.dropna(subset=["Best Test CER"]).copy()
    if valid_df.empty:
        st.info("No results to compare.")
        return

    # Scatter: CER vs Parameters
    valid_df["CER (%)"] = valid_df["Best Test CER"] * 100
    valid_df["Params (M)"] = valid_df["Params"].apply(
        lambda x: x / 1e6 if pd.notna(x) else 0
    )

    fig_scatter = px.scatter(
        valid_df,
        x="Params (M)", y="CER (%)",
        color="Architecture",
        size="Epochs Completed",
        hover_data=["Run", "Model Detail", "Best Test WER"],
        color_discrete_map={v: ARCH_COLORS[k] for k, v in ARCH_LABELS.items()},
        title="CER vs Model Size",
    )
    fig_scatter.update_layout(height=450, template="plotly_white")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Radar chart for best run per architecture
    st.subheader("🎯 Best Run per Architecture — Radar")
    best_per_arch = valid_df.groupby("Architecture").apply(
        lambda g: g.loc[g["Best Test CER"].idxmin()]
    ).reset_index(drop=True)

    if len(best_per_arch) > 1:
        categories = ["CER", "WER", "CTC Loss"]
        fig_radar = go.Figure()

        for _, row in best_per_arch.iterrows():
            cer_norm = min(row["Best Test CER"] * 100, 100) if pd.notna(row["Best Test CER"]) else 100
            wer_norm = min(row["Best Test WER"] * 100, 100) if pd.notna(row["Best Test WER"]) else 100
            loss_norm = min(row["Final CTC Loss"] / 1.5 if pd.notna(row["Final CTC Loss"]) else 100, 100)
            values = [cer_norm, wer_norm, loss_norm]
            arch_key = row["arch_type"]

            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=f"{row['Architecture']} ({row['Run']})",
                line=dict(color=ARCH_COLORS.get(arch_key, "#888")),
                opacity=0.6,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Lower is better (normalized to 0-100 scale)",
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: RUN EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
def tab_run_explorer():
    runs = discover_runs()
    if not runs:
        st.warning("No experiments found.")
        return

    # ── Run selector ──────────────────────────────────────────────────
    run_names = [r["run_name"] for r in runs]
    selected_run = st.selectbox("Select experiment run", run_names, index=len(run_names)-1)
    run_info = next(r for r in runs if r["run_name"] == selected_run)
    run_path = run_info["path"]

    cfg = load_config(str(run_path))
    results_df = load_results(str(run_path))

    # ── Config Display ────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Configuration")
        arch = cfg.get("arch", {})
        arch_type = arch.get("type", "unknown")

        st.markdown(f"**Architecture:** `{ARCH_LABELS.get(arch_type, arch_type)}`")
        st.markdown(f"**Epochs:** {cfg.get('train', {}).get('num_epochs', '?')}")
        st.markdown(f"**Batch Size:** {cfg.get('train', {}).get('batch_size', '?')}")
        st.markdown(f"**Learning Rate:** {cfg.get('train', {}).get('lr', '?')}")
        st.markdown(f"**Image Size:** {cfg.get('preproc', {}).get('image_height', '?')} × {cfg.get('preproc', {}).get('image_width', '?')}")

        with st.expander("Full Architecture Config"):
            st.json(arch)
        with st.expander("Full Config JSON"):
            st.json(cfg)

    with col2:
        st.subheader("📊 Training Summary")
        if results_df.empty:
            st.warning("No results.csv found for this run.")
        else:
            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            best_cer = results_df["test/cer"].min() if "test/cer" in results_df.columns else None
            best_wer = results_df["test/wer"].min() if "test/wer" in results_df.columns else None
            final_loss = results_df["train/ctc_loss"].iloc[-1] if "train/ctc_loss" in results_df.columns else None
            n_params = int(results_df["model/params"].iloc[0]) if "model/params" in results_df.columns else None

            m1.metric("Best Test CER", fmt_cer(best_cer))
            m2.metric("Best Test WER", fmt_wer(best_wer))
            m3.metric("Final CTC Loss", fmt_loss(final_loss))
            m4.metric("Parameters", fmt_params(n_params))

            # Epoch table
            display_cols = [c for c in ["epoch", "lr", "train/ctc_loss", "val/cer", "test/cer",
                                        "val/wer", "test/wer", "time/epoch(s)"] if c in results_df.columns]
            st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True, height=300)

    st.markdown("---")

    # ── Per-Sample Error Analysis ─────────────────────────────────────
    st.subheader("🔎 Per-Sample Error Analysis")
    eval_df = load_evaluation_details(str(run_path))

    if eval_df.empty:
        st.info("No evaluation_details.csv for this run.")
    else:
        # Only look at the last epoch's test data
        if "epoch" in eval_df.columns and "dataset" in eval_df.columns:
            last_epoch = eval_df["epoch"].max()
            test_df = eval_df[(eval_df["epoch"] == last_epoch) & (eval_df["dataset"] == "test")].copy()
        else:
            test_df = eval_df.copy()

        if test_df.empty:
            st.info("No test-set evaluation details found.")
        else:
            if "sample_cer" in test_df.columns:
                c1, c2 = st.columns(2)

                with c1:
                    # CER distribution
                    fig_hist = px.histogram(
                        test_df, x="sample_cer", nbins=50,
                        title="Per-Sample CER Distribution",
                        labels={"sample_cer": "Character Error Rate"},
                        color_discrete_sequence=[ARCH_COLORS.get(arch_type, "#2E86C1")],
                    )
                    fig_hist.update_layout(height=350, template="plotly_white")
                    st.plotly_chart(fig_hist, use_container_width=True)

                with c2:
                    # CER vs text length
                    if "gt_length" in test_df.columns:
                        fig_scatter = px.scatter(
                            test_df, x="gt_length", y="sample_cer",
                            title="CER vs Ground Truth Length",
                            labels={"gt_length": "GT Length (chars)", "sample_cer": "CER"},
                            opacity=0.4,
                            color_discrete_sequence=[ARCH_COLORS.get(arch_type, "#2E86C1")],
                        )
                        fig_scatter.update_layout(height=350, template="plotly_white")
                        st.plotly_chart(fig_scatter, use_container_width=True)

                # Worst & best samples
                col_w, col_b = st.columns(2)
                with col_w:
                    st.markdown("**❌ Worst Predictions (highest CER):**")
                    worst = test_df.nlargest(5, "sample_cer")[["ground_truth", "prediction", "sample_cer"]].copy()
                    worst["sample_cer"] = worst["sample_cer"].apply(lambda x: f"{x:.4f}")
                    st.dataframe(worst, use_container_width=True, hide_index=True)

                with col_b:
                    st.markdown("**✅ Best Predictions (lowest CER):**")
                    best = test_df.nsmallest(5, "sample_cer")[["ground_truth", "prediction", "sample_cer"]].copy()
                    best["sample_cer"] = best["sample_cer"].apply(lambda x: f"{x:.4f}")
                    st.dataframe(best, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Run Artifacts ─────────────────────────────────────────────────
    st.subheader("📁 Run Artifacts")
    artifacts = list(run_path.iterdir())
    artifact_names = sorted([a.name for a in artifacts])
    st.markdown(f"**Files in `{selected_run}/`:** {', '.join(artifact_names)}")

    # Show any PNG images in the run directory
    pngs = [a for a in artifacts if a.suffix == ".png"]
    if pngs:
        st.markdown("**Visualizations:**")
        cols = st.columns(min(3, len(pngs)))
        for i, png in enumerate(pngs):
            with cols[i % 3]:
                img = safe_image(png)
                if img:
                    st.image(img, caption=png.stem.replace("_", " ").title(), use_container_width=True)

    # Training log preview
    log_path = run_path / "training.log"
    if log_path.exists():
        with st.expander("📝 Training Log (last 50 lines)"):
            with open(log_path) as f:
                lines = f.readlines()
                st.code("".join(lines[-50:]), language="text")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: ATTENTION MAPS & EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════
def tab_attention():
    st.subheader("🧠 Attention Visualizations & Explainability")

    subtab = st.radio(
        "Visualization Category",
        ["Character Attention", "Register Analysis", "Grad-CAM", "Run-Specific Attention"],
        horizontal=True,
    )

    if subtab == "Character Attention":
        st.markdown("**Character-level attention from ViT models** — shows which image patches attend to each predicted character.")
        char_dir = VISUALIZATIONS_DIR / "character_attention"
        batch_dir = VISUALIZATIONS_DIR / "character_attention_batch"

        pngs = sorted(char_dir.glob("*.png")) if char_dir.exists() else []
        batch_pngs = sorted(batch_dir.glob("*.png")) if batch_dir.exists() else []
        all_pngs = pngs + batch_pngs

        if not all_pngs:
            st.info("No character attention visualizations found. Run `scripts/postprocessing/visualize_character_vit.py` to generate them.")
        else:
            cols = st.columns(min(2, len(all_pngs)))
            for i, png in enumerate(all_pngs):
                with cols[i % 2]:
                    img = safe_image(png)
                    if img:
                        st.image(img, caption=png.stem.replace("_", " ").title(), use_container_width=True)

    elif subtab == "Register Analysis":
        st.markdown("**Register token analysis** — attention patterns of learned register tokens across different configurations.")

        reg_dir = VISUALIZATIONS_DIR / "register_analysis"
        if not reg_dir.exists():
            st.info("No register analysis visualizations found.")
            return

        # Comparison
        comp_dir = reg_dir / "register_comparison"
        if comp_dir.exists():
            pngs = sorted(comp_dir.glob("*.png"))
            if pngs:
                st.markdown("### Register Comparison")
                for png in pngs:
                    img = safe_image(png)
                    if img:
                        st.image(img, caption=png.stem.replace("_", " ").title(), use_container_width=True)

        # Attention heads
        heads_dir = reg_dir / "attention_heads"
        if heads_dir.exists():
            pngs = sorted(heads_dir.glob("*.png"))
            if pngs:
                st.markdown("### Attention Heads per Register Count")
                cols = st.columns(min(3, len(pngs)))
                for i, png in enumerate(pngs):
                    with cols[i % 3]:
                        img = safe_image(png)
                        if img:
                            # Extract register count from filename
                            match = re.search(r"(\d+)regs", png.stem)
                            cap = f"{match.group(1)} Registers" if match else png.stem
                            st.image(img, caption=cap, use_container_width=True)

    elif subtab == "Grad-CAM":
        st.markdown("**Gradient-weighted Class Activation Maps** — highlights regions the model relies on for predictions.")
        gc_dir = VISUALIZATIONS_DIR / "character_gradcam"
        pngs = sorted(gc_dir.glob("*.png")) if gc_dir.exists() else []

        if not pngs:
            st.info("No Grad-CAM visualizations found. Run `scripts/postprocessing/gradcam_vit_rgts.py` to generate them.")
        else:
            cols = st.columns(min(2, len(pngs)))
            for i, png in enumerate(pngs):
                with cols[i % 2]:
                    img = safe_image(png)
                    if img:
                        st.image(img, caption=png.stem, use_container_width=True)

    elif subtab == "Run-Specific Attention":
        st.markdown("**Attention maps saved within specific experiment runs.**")

        runs = discover_runs()
        vit_runs = []
        for r in runs:
            cfg = load_config(str(r["path"]))
            if cfg.get("arch", {}).get("type") in ["vit_rgts", "torchvision_vit", "trocr"]:
                vit_runs.append(r)

        if not vit_runs:
            st.info("No ViT runs found.")
            return

        selected = st.selectbox(
            "Select ViT run",
            [r["run_name"] for r in vit_runs],
        )
        run_path = next(r["path"] for r in vit_runs if r["run_name"] == selected)

        # Find all PNGs in this run
        pngs = sorted(run_path.glob("*.png"))
        if not pngs:
            st.info("No visualizations found for this run.")
        else:
            cols = st.columns(min(3, len(pngs)))
            for i, png in enumerate(pngs):
                with cols[i % 3]:
                    img = safe_image(png)
                    if img:
                        st.image(img, caption=png.stem.replace("_", " ").title(), use_container_width=True)

        # Explainability demo
        explain_dir = run_path / "explainability_demo"
        if explain_dir.exists():
            st.markdown("### Explainability Demo")
            demo_pngs = sorted(explain_dir.glob("*.png"))
            if demo_pngs:
                cols = st.columns(min(3, len(demo_pngs)))
                for i, png in enumerate(demo_pngs):
                    with cols[i % 3]:
                        img = safe_image(png)
                        if img:
                            st.image(img, caption=png.stem.replace("_", " ").title(), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: DATASET EDA
# ═══════════════════════════════════════════════════════════════════════════
def tab_dataset():
    st.subheader("📚 IAM Handwriting Dataset — Exploratory Data Analysis")

    eda = load_eda_summary()

    if eda:
        # ── Dataset overview cards ────────────────────────────────────
        overview = eda.get("dataset_overview", {})
        splits = overview.get("splits", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples", overview.get("total_samples", "?"))

        train_info = splits.get("train", {})
        val_info = splits.get("val", {})
        test_info = splits.get("test", {})

        c2.metric("Train", train_info.get("num_samples", "?"))
        c3.metric("Validation", val_info.get("num_samples", "?"))
        c4.metric("Test", test_info.get("num_samples", "?"))

        # Split distribution pie chart
        fig_pie = px.pie(
            values=[train_info.get("num_samples", 0), val_info.get("num_samples", 0), test_info.get("num_samples", 0)],
            names=["Train", "Validation", "Test"],
            title="Dataset Split Distribution",
            color_discrete_sequence=["#2E86C1", "#27AE60", "#E74C3C"],
            hole=0.4,
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Text statistics
        text_stats = eda.get("text_statistics", {})
        writer_stats = eda.get("writer_statistics", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Characters", f"{text_stats.get('total_characters', 0):,}")
        c2.metric("Unique Characters", text_stats.get("unique_characters", "?"))
        c3.metric("Unique Writers", writer_stats.get("unique_writers", "?"))

        # Top characters
        top_chars = eda.get("top_characters", [])
        if top_chars:
            st.markdown("**Character Frequency Distribution (Top 20):**")
            char_df = pd.DataFrame(top_chars[:20])
            char_df["char"] = char_df["char"].apply(lambda c: repr(c) if c in [" ", "\t", "\n"] else c)
            fig_bar = px.bar(
                char_df, x="char", y="count",
                title="Top 20 Most Frequent Characters",
                color_discrete_sequence=["#2E86C1"],
            )
            fig_bar.update_layout(height=350, template="plotly_white",
                                  xaxis_title="Character", yaxis_title="Count")
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Pre-generated EDA plots ───────────────────────────────────────
    eda_dir = OUTPUT_DIR / "data_analysis"
    if eda_dir.exists():
        pngs = sorted(eda_dir.glob("*.png"))
        if pngs:
            st.subheader("📊 Data Analysis Plots")
            cols = st.columns(min(3, len(pngs)))
            for i, png in enumerate(pngs):
                with cols[i % 3]:
                    img = safe_image(png)
                    if img:
                        st.image(img, caption=png.stem.replace("_", " ").title(), use_container_width=True)
    else:
        if not eda:
            st.info("No EDA data found. Run preprocessing/EDA scripts to generate dataset analysis.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7: LIVE MONITORING
# ═══════════════════════════════════════════════════════════════════════════
def tab_monitoring():
    st.subheader("📡 Live Experiment Monitoring")
    st.caption("Monitor running SLURM jobs and tail their logs.")

    # ── SLURM Queue ───────────────────────────────────────────────────
    st.markdown("### 🖥️ Current SLURM Queue")
    if st.button("🔄 Refresh Queue", key="refresh_queue"):
        st.cache_data.clear()

    try:
        import subprocess
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "--format=%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                st.code(result.stdout, language="text")
                st.success(f"Found {len(lines)-1} job(s) in queue.")
            else:
                st.info("No jobs currently in the SLURM queue.")
        else:
            st.info("No jobs found or `squeue` not available.")
    except FileNotFoundError:
        st.warning("`squeue` command not found. Not running on a SLURM cluster?")
    except Exception as e:
        st.warning(f"Could not query SLURM: {e}")

    st.markdown("---")

    # ── Log viewer ────────────────────────────────────────────────────
    st.markdown("### 📋 Experiment Logs")

    log_dirs = {
        "Baseline (CNN-RNN)": LOGS_DIR / "baseline",
        "ViT-RGTS Registers": LOGS_DIR / "vit_rgts_registers",
        "Pretrained Models": LOGS_DIR / "pretrained",
    }

    for dir_name, dir_path in log_dirs.items():
        if dir_path.exists():
            log_files = sorted(dir_path.glob("output_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            error_files = sorted(dir_path.glob("error_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

            if log_files or error_files:
                with st.expander(f"📂 {dir_name} ({len(log_files)} output, {len(error_files)} error logs)"):
                    # Show most recent output log
                    if log_files:
                        latest = log_files[0]
                        st.markdown(f"**Latest:** `{latest.name}` ({latest.stat().st_size / 1024:.1f} KB)")
                        try:
                            with open(latest) as f:
                                content = f.read()
                            lines = content.strip().split("\n")
                            n_tail = st.slider(
                                f"Lines to show ({latest.name})",
                                10, min(200, len(lines)), 30,
                                key=f"slider_{dir_name}",
                            )
                            st.code("\n".join(lines[-n_tail:]), language="text")
                        except Exception as e:
                            st.error(f"Error reading log: {e}")

                    # Show error logs if non-empty
                    if error_files:
                        latest_err = error_files[0]
                        if latest_err.stat().st_size > 0:
                            st.markdown(f"**⚠️ Error log:** `{latest_err.name}`")
                            try:
                                with open(latest_err) as f:
                                    err_content = f.read()
                                if err_content.strip():
                                    st.code(err_content[-2000:], language="text")
                            except Exception:
                                pass

    # ── Auto-detect new runs ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🆕 Recently Modified Runs")
    runs = discover_runs()
    recent = sorted(runs, key=lambda r: r["path"].stat().st_mtime, reverse=True)[:5]

    for r in recent:
        mtime = datetime.fromtimestamp(r["path"].stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        cfg = load_config(str(r["path"]))
        arch = cfg.get("arch", {}).get("type", "unknown")
        results = load_results(str(r["path"]))
        n_epochs = len(results) if not results.empty else 0
        best_cer = results["test/cer"].min() if not results.empty and "test/cer" in results.columns else None
        st.markdown(
            f"- **{r['run_name']}** · {ARCH_LABELS.get(arch, arch)} · "
            f"{n_epochs} epochs · CER: {fmt_cer(best_cer)} · Modified: {mtime}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## ✍️ HTR Pipeline")
        st.caption("Handwritten Text Recognition\nIAM Dataset · CTC Decoding")
        st.markdown("---")

        # Quick stats
        runs = discover_runs()
        overview = build_overview_table()
        valid = overview.dropna(subset=["Best Test CER"])

        st.markdown(f"**Runs discovered:** {len(runs)}")
        if not valid.empty:
            best = valid.loc[valid["Best Test CER"].idxmin()]
            st.markdown(f"**Overall best CER:** {fmt_cer(best['Best Test CER'])}")
            st.markdown(f"**Best run:** {best['Run']}")
            st.markdown(f"**Architecture:** {best['Architecture']}")

        st.markdown("---")

        # Architecture legend
        st.markdown("### 🎨 Architecture Legend")
        for key, label in ARCH_LABELS.items():
            color = ARCH_COLORS[key]
            st.markdown(f'<span class="arch-badge arch-{key}">{label}</span>', unsafe_allow_html=True)

        st.markdown("---")

        # Data refresh
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption(f"Project: `HTR-Pipeline`\nPath: `{PROJECT_ROOT}`")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    inject_css()
    render_sidebar()

    # Title
    st.markdown('<div class="dashboard-title">✍️ HTR Pipeline Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">'
        'Handwritten Text Recognition · IAM Dataset · CNN-RNN · ViT-RGTS · TorchVision ViT · TrOCR'
        '</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "📈 Training Curves",
        "🔬 Architecture Lab",
        "🔍 Run Explorer",
        "🧠 Attention Maps",
        "📚 Dataset EDA",
        "📡 Monitoring",
    ])

    with tab1:
        tab_overview()
    with tab2:
        tab_training_curves()
    with tab3:
        tab_architecture_lab()
    with tab4:
        tab_run_explorer()
    with tab5:
        tab_attention()
    with tab6:
        tab_dataset()
    with tab7:
        tab_monitoring()


if __name__ == "__main__":
    main()
