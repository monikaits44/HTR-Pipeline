#!/usr/bin/env python3
"""
Figure 3: GradCAM + Attention Combined Analysis
================================================

Three-panel publication figure:
  (a) Per-character GradCAM maps across register configs (heatmap grid)
  (b) Attention entropy comparison: bar chart per character × register count
  (c) Spatial ordering quality: Spearman ρ + aligned head count per register

Combines gradient-based (GradCAM) and attention-based localization to provide
complementary evidence for the register token effect on character localization.

Usage:
    python scripts/pub_fig3_gradcam_quantitative.py
    python scripts/pub_fig3_gradcam_quantitative.py --image path/to/img.png --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models import HTRNet
from utils.preprocessing import load_image, preprocess

EXPERIMENTS = ROOT / "saved_models" / "experiments"
CLASSES_PATH = EXPERIMENTS / "classes.npy"
IAM_TEST = ROOT / "data" / "IAM" / "processed_lines" / "test"
IMAGE_H, IMAGE_W = 128, 1024
SEQ_LEN = 5

REGISTER_RUNS = {0: "run_144", 4: "run_146", 8: "run_151", 16: "run_152"}
REG_COLORS = {0: "#1976D2", 4: "#388E3C", 8: "#F57C00", 16: "#D32F2F"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def load_charset():
    return list(np.load(str(CLASSES_PATH), allow_pickle=True))


def load_model(run_id, device="cpu"):
    run_dir = EXPERIMENTS / run_id
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    arch_ns = SimpleNamespace(**cfg["arch"])
    charset = load_charset()
    model = HTRNet(arch_ns, len(charset) + 1)
    state = torch.load(str(run_dir / "model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def prepare_image(img_path, device="cpu", requires_grad=False):
    img_np = load_image(str(img_path))
    img_proc = preprocess(img_np, (IMAGE_H, IMAGE_W))
    tensor = torch.from_numpy(img_proc).unsqueeze(0).unsqueeze(0).float().to(device)
    if requires_grad:
        tensor.requires_grad_(True)
    col_var = np.var(img_proc, axis=0)
    thr = np.median(col_var) * 0.1 + 1e-6
    active = np.where(col_var > thr)[0]
    xs = max(0, active[0] - 4) if len(active) > 0 else 0
    xe = min(IMAGE_W, active[-1] + 4) if len(active) > 0 else IMAGE_W
    return tensor, img_proc, (xs, xe)


def ctc_decode(logits, charset):
    seq = torch.softmax(logits.detach(), dim=-1)[:, 0, :].argmax(dim=-1).cpu().numpy()
    chars, peaks = [], []
    prev = -1
    for t, idx in enumerate(seq):
        if idx != 0 and idx != prev:
            ci = idx - 1
            if 0 <= ci < len(charset):
                chars.append(charset[ci])
                peaks.append(t)
        prev = idx
    return "".join(chars), peaks


def shallow_rollout(attn_maps, n_layers=2):
    layers = attn_maps[-n_layers:]
    rollout = None
    for attn in layers:
        a = attn.mean(dim=1).numpy()[0]
        a = 0.5 * a + 0.5 * np.eye(a.shape[0])
        rollout = a if rollout is None else rollout @ a
    rollout /= rollout.sum(axis=-1, keepdims=True)
    return rollout


def extract_char_attention(attn_maps, R, Wp, peaks, seq_len=SEQ_LEN):
    rollout = shallow_rollout(attn_maps)
    char_attns = []
    for i in range(seq_len):
        if i < len(peaks):
            qi = R + peaks[i]
            row = rollout[qi, R:R + Wp].copy() if qi < rollout.shape[0] else np.zeros(Wp)
        else:
            row = np.zeros(Wp)
        char_attns.append(row)
    return char_attns


def compute_entropy(attn_1d):
    p = attn_1d / (attn_1d.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_spearman(char_attns, Wp):
    positions = []
    for ca in char_attns:
        if ca.sum() > 1e-10:
            positions.append(float(np.sum(np.arange(Wp) * ca) / ca.sum()))
        else:
            positions.append(0.0)
    valid = [i for i, ca in enumerate(char_attns) if ca.sum() > 1e-10]
    if len(valid) < 3:
        return 0.0
    rho, _ = spearmanr([i for i in valid], [positions[i] for i in valid])
    return float(rho) if not np.isnan(rho) else 0.0


def count_aligned_heads(attn_maps, R, Wp, peaks, rho_thresh=0.5):
    count = 0
    for attn in attn_maps:
        for h in range(attn.shape[1]):
            h_peaks = []
            for t_c in peaks[:SEQ_LEN]:
                qi = R + t_c
                if qi < attn.shape[2]:
                    row = attn[0, h, qi, R:R + Wp].numpy()
                    h_peaks.append(int(np.argmax(row)))
            if len(h_peaks) >= 3:
                rho, _ = spearmanr(range(len(h_peaks)), h_peaks)
                if not np.isnan(rho) and rho > rho_thresh:
                    count += 1
    return count


class GradCAMExtractor:
    """GradCAM for per-character CTC logits via CNN stem."""
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hook_handles = []
        target = self.model.backbone.cnn_stem[-1]  # last GELU
        self._hook_handles.append(
            target.register_forward_hook(self._fwd_hook))
        self._hook_handles.append(
            target.register_full_backward_hook(self._bwd_hook))

    def _fwd_hook(self, module, inp, out):
        self.activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def compute_per_char(self, image_tensor, peaks, charset):
        """Return list of GradCAM maps, one per peak timestep."""
        import cv2
        cams = []
        for t in peaks[:SEQ_LEN]:
            self.model.zero_grad()
            if image_tensor.grad is not None:
                image_tensor.grad.zero_()
            logits = self.model(image_tensor)
            target_class = logits[t, 0].argmax().item()
            score = logits[t, 0, target_class]
            # cuDNN's fused RNN kernels refuse to run backward() in eval mode;
            # disable cuDNN for this call so PyTorch falls back to its
            # (equally correct, just slower) native RNN backward path.
            with torch.backends.cudnn.flags(enabled=False):
                score.backward(retain_graph=True)

            gradients = self.gradients.detach()
            activations = self.activations.detach()
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)[0, 0].cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cv2.resize(cam, (IMAGE_W, IMAGE_H))
            cams.append(cam)

        # Pad to SEQ_LEN
        while len(cams) < SEQ_LEN:
            cams.append(np.zeros((IMAGE_H, IMAGE_W)))
        return cams

    def cleanup(self):
        for h in self._hook_handles:
            h.remove()


def find_good_image():
    gt_path = IAM_TEST / "gt.txt"
    if not gt_path.exists():
        return None, None
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                img_id, text = parts[0], parts[1].strip()
                if 4 <= len(text) <= 6 and (IAM_TEST / f"{img_id}.png").exists():
                    return IAM_TEST / f"{img_id}.png", text
    return None, None


def generate_figure(models, charset, img_path, gt_text, device, output_path):
    """Generate 3-panel figure: GradCAM grid + entropy + spatial ordering."""

    reg_counts = sorted(models.keys())

    # --- Collect data ---
    all_data = {}
    for nr in reg_counts:
        model = models[nr]

        # Attention-based
        tensor, img_proc, text_bbox = prepare_image(img_path, device)
        with torch.no_grad():
            logits, reg_tokens, attn_maps, token_norms, grid = model.forward_explain(tensor)
        decoded, peaks = ctc_decode(logits, charset)
        R = 0 if reg_tokens is None else reg_tokens.shape[1]
        Wp = grid[1]
        char_attns = extract_char_attention(attn_maps, R, Wp, peaks, SEQ_LEN)
        rho = compute_spearman(char_attns, Wp)
        aligned = count_aligned_heads(attn_maps, R, Wp, peaks)

        # GradCAM-based
        tensor_g, _, _ = prepare_image(img_path, device, requires_grad=True)
        extractor = GradCAMExtractor(model)
        gradcams = extractor.compute_per_char(tensor_g, peaks, charset)
        extractor.cleanup()

        all_data[nr] = {
            "decoded": decoded, "peaks": peaks,
            "char_attns": char_attns, "gradcams": gradcams,
            "Wp": Wp, "R": R, "rho": rho, "aligned_heads": aligned,
            "attn_maps": attn_maps,
        }
        print(f"  {nr:>2} reg: \"{decoded[:SEQ_LEN]}\" ρ={rho:.2f} aligned={aligned}")

    # Use first model's decoded text for labels
    ref = all_data[reg_counts[0]]
    decoded = ref["decoded"]
    chars = list(decoded[:SEQ_LEN])
    while len(chars) < SEQ_LEN:
        chars.append("—")

    xs, xe = prepare_image(img_path, device)[2]
    img_crop = prepare_image(img_path, device)[1][:, xs:xe]

    # --- Layout: top = GradCAM grid, bottom-left = entropy, bottom-right = ρ + heads ---
    fig = plt.figure(figsize=(14, 9))

    # Top: GradCAM grid (4 rows × 6 cols)
    gs_top = gridspec.GridSpec(
        len(reg_counts), SEQ_LEN + 1, figure=fig,
        left=0.06, right=0.94, top=0.92, bottom=0.48,
        wspace=0.03, hspace=0.15,
    )

    cmap = plt.cm.jet

    for ri, nr in enumerate(reg_counts):
        d = all_data[nr]
        # Input
        ax = fig.add_subplot(gs_top[ri, 0])
        ax.imshow(img_crop, cmap="gray", aspect="auto")
        ax.set_ylabel(f"R={nr}", fontsize=10, fontweight="bold",
                       rotation=0, labelpad=25, va="center")
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title("Input", fontsize=10, fontweight="bold", pad=4)

        for ci in range(SEQ_LEN):
            ax = fig.add_subplot(gs_top[ri, ci + 1])
            cam = d["gradcams"][ci]
            is_pad = ci >= len(d["peaks"]) or cam.max() < 1e-8

            if is_pad:
                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.15)
                ax.text(0.5, 0.5, "pad", transform=ax.transAxes,
                        fontsize=7, color="gray", ha="center", va="center", style="italic")
            else:
                cam_crop = cam[:, xs:xe]
                ax.imshow(img_crop, cmap="gray", aspect="auto", alpha=0.4)
                ax.imshow(cam_crop, cmap="jet", aspect="auto", alpha=0.6)

            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                label = chars[ci] if chars[ci] != " " else "⎵"
                ax.set_title(f"'{label}'", fontsize=10, fontweight="bold", pad=4)

    fig.text(0.5, 0.94, "(a) Per-Character GradCAM Across Register Configurations",
             fontsize=11, fontweight="bold", ha="center")

    # Bottom panels
    gs_bot = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.08, right=0.95, top=0.42, bottom=0.06,
        wspace=0.35,
    )

    # --- (b) Entropy ---
    ax_ent = fig.add_subplot(gs_bot[0, 0])
    x = np.arange(SEQ_LEN)
    bar_w = 0.8 / len(reg_counts)
    for i, nr in enumerate(reg_counts):
        d = all_data[nr]
        ents = [compute_entropy(ca) if ca.sum() > 1e-10 else 0 for ca in d["char_attns"]]
        ax_ent.bar(x + i * bar_w, ents, bar_w, color=REG_COLORS[nr],
                   label=f"R={nr}", alpha=0.85, edgecolor="white", linewidth=0.3)
    char_labels = [c if c != " " else "⎵" for c in chars]
    ax_ent.set_xticks(x + bar_w * (len(reg_counts) - 1) / 2)
    ax_ent.set_xticklabels(char_labels, fontsize=10)
    ax_ent.set_ylabel("Entropy (bits)")
    ax_ent.set_title("(b) Character Attention Entropy\n(lower = more focused)", fontsize=10)
    ax_ent.legend(fontsize=7, ncol=2)
    ax_ent.grid(axis="y", alpha=0.15)

    # --- (c) Spatial ordering ρ + aligned heads ---
    ax_rho = fig.add_subplot(gs_bot[0, 1])
    rhos = [all_data[nr]["rho"] for nr in reg_counts]
    heads = [all_data[nr]["aligned_heads"] for nr in reg_counts]
    total_heads = len(all_data[reg_counts[0]]["attn_maps"]) * \
                  all_data[reg_counts[0]]["attn_maps"][0].shape[1]

    x2 = np.arange(len(reg_counts))
    width = 0.35
    bars1 = ax_rho.bar(x2 - width / 2, rhos, width, color=[REG_COLORS[nr] for nr in reg_counts],
                       alpha=0.85, edgecolor="white", label="Spearman ρ")
    ax_rho.set_ylabel("Spearman ρ")
    ax_rho.set_ylim(-0.2, 1.15)
    ax_rho.axhline(y=1.0, color="gray", ls=":", alpha=0.3)

    ax2 = ax_rho.twinx()
    bars2 = ax2.bar(x2 + width / 2, heads, width, color=[REG_COLORS[nr] for nr in reg_counts],
                    alpha=0.4, edgecolor="black", linewidth=0.5, hatch="//", label="Aligned heads")
    ax2.set_ylabel(f"Aligned heads (/{total_heads})")
    ax2.set_ylim(0, max(heads) * 1.5 + 1)

    ax_rho.set_xticks(x2)
    ax_rho.set_xticklabels([f"R={nr}" for nr in reg_counts])
    ax_rho.set_title("(c) Spatial Ordering & Head Alignment\n(higher = better)", fontsize=10)

    # Value labels
    for bar, val in zip(bars1, rhos):
        ax_rho.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    for bar, val in zip(bars2, heads):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 str(val), ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax_rho.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_rho.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax_rho.grid(axis="y", alpha=0.15)

    for ext in [".png", ".pdf"]:
        out = output_path.with_suffix(ext)
        fig.savefig(str(out), facecolor="white")
        print(f"  Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fig 3: GradCAM + quantitative analysis")
    parser.add_argument("--image", help="Specific image path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "pub_figures"))
    parser.add_argument("--runs", nargs=4, help="Run IDs for 0,4,8,16 registers")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.runs:
        for i, nr in enumerate([0, 4, 8, 16]):
            REGISTER_RUNS[nr] = args.runs[i]

    charset = load_charset()
    models = {}
    for nr, run_id in REGISTER_RUNS.items():
        models[nr] = load_model(run_id, args.device)
        print(f"Loaded {run_id} ({nr} registers)")

    if args.image:
        img_path, gt_text = Path(args.image), "?"
    else:
        img_path, gt_text = find_good_image()
        if img_path is None:
            print("No suitable image found. Use --image.")
            return
    print(f"Image: {img_path.name}, GT: \"{gt_text}\"")

    generate_figure(
        models, charset, img_path, gt_text,
        args.device, output_dir / "fig3_gradcam_quantitative",
    )
    print("Done.")


if __name__ == "__main__":
    main()
