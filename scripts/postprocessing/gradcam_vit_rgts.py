#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
import torch

# Make sure project root is on sys.path
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import HTRNet
from utils.htr_dataset import HTRDataset


def parse_config_and_overrides(argv):
    """
    Match trainer-style config merging:

    python gradcam_vit_rgts.py config.yaml baseline.yaml baseline_vit_rgts.yaml \
        resume=... device=cuda
    """
    if len(argv) < 2:
        raise ValueError(
            "Usage: python gradcam_vit_rgts.py <config.yaml> [extra.yaml ...] key=value ..."
        )

    # 1) base YAML
    conf = OmegaConf.load(argv[1])

    # 2) additional YAMLs
    for arg in argv[2:]:
        if arg.endswith(".yaml"):
            extra = OmegaConf.load(arg)
            conf = OmegaConf.merge(conf, extra)

    # 3) key=value overrides
    for arg in argv[2:]:
        if "=" in arg and not arg.endswith(".yaml"):
            k, v_str = arg.split("=", 1)
            try:
                v = eval(v_str, {"__builtins__": {}})
            except Exception:
                v = v_str
            conf = OmegaConf.merge(conf, OmegaConf.create({k: v}))

    OmegaConf.set_struct(conf, True)
    return conf


def load_model_and_dataset(config):
    device = config.device

    # Dataset (test split)
    fixed_size = (config.preproc.image_height, config.preproc.image_width)
    dataset = HTRDataset(config.data.path, "test", fixed_size=fixed_size, transforms=None)
    print("# testing lines:", len(dataset))

    # Classes + mapping
    classes = np.load(os.path.join(config.data.path, "classes.npy"))
    c2i = {c: (i + 1) for i, c in enumerate(classes)}
    i2c = {(i + 1): c for i, c in enumerate(classes)}

    # Model
    print("Preparing Net - Architectural elements:")
    print(config.arch)

    net = HTRNet(config.arch, len(classes) + 1)

    if config.resume is None:
        raise ValueError("config.resume must point to a trained checkpoint (model.pt).")

    print(f"Loading checkpoint from: {config.resume}")
    ckpt = torch.load(config.resume, map_location="cpu")
    net.load_state_dict(ckpt, strict=True)
    net.to(device)
    # NOTE: we do NOT call net.eval() here because we want gradients
    # and cuDNN RNN backward requires training mode.

    # Sanity: ViT backbone
    arch_type = getattr(config.arch, "type", "cnn_rnn")
    if arch_type != "vit_rgts":
        raise ValueError(
            f"Expected arch.type == 'vit_rgts', got '{arch_type}'. "
            "Use baseline_vit_rgts.yaml in the config list."
        )

    return net, dataset, i2c


def ctc_decode(tdec, i2c, blank_id=0):
    """Collapse repeats + remove blanks."""
    tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
    dec = "".join([i2c[t] for t in tt if t != blank_id])
    return dec


def main():
    """
    Example usage (from project root):

    python scripts/postprocessing/gradcam_vit_rgts.py ^
        configs/config.yaml configs/baseline.yaml configs/baseline_vit_rgts.yaml ^
        resume=./saved_models/experiments/run_24/model.pt ^
        device='cuda' ^
        --img-idx 0
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="+",
        help="YAML config files (first is base, others merged)",
    )
    parser.add_argument(
        "--img-idx",
        type=int,
        default=0,
        help="Index of test sample to visualize",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save figure; if empty, show",
    )
    args, unknown = parser.parse_known_args()

    # Build a fake argv: [script, cfg1, cfg2, ..., overrides...]
    fake_argv = ["gradcam_vit_rgts.py"] + args.configs + unknown
    config = parse_config_and_overrides(fake_argv)

    # ------------------------------------------------------------------
    # Load model + test dataset
    # ------------------------------------------------------------------
    net, test_set, i2c = load_model_and_dataset(config)
    device = config.device

    img_idx = args.img_idx
    if img_idx < 0 or img_idx >= len(test_set):
        raise IndexError(f"img_idx {img_idx} out of range (0..{len(test_set) - 1})")

    # Put model in train mode so cuDNN RNN backward works
    net.train()

    # Get image + GT text from dataset
    img_path, gt_text = test_set.data[img_idx]
    print(f"[INFO] Visualizing Grad-CAM for test sample idx={img_idx}")
    print(f"[INFO] Image path: {img_path}")
    print(f"[INFO] GT text   : {gt_text}")

    img_tensor, _ = test_set[img_idx]  # [1, H, W]
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1,1,H,W]

    # ------------------------------------------------------------------
    # Forward with gradient wrt seq_tokens (patch tokens)
    # ------------------------------------------------------------------
    # We want gradients w.r.t. seq_tokens, so enable grad globally here
    img_tensor.requires_grad_(False)

    # backbone: seq_tokens [T,B,D], reg_tokens [B,R,D], grid=(Hp,Wp)
    seq_tokens, reg_tokens, (Hp, Wp) = net.backbone(img_tensor)  # [T,1,D], [1,R,D]
    seq_tokens.retain_grad()

    # Convert to 4D for CTC head: [B,D,1,T]
    seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)  # [1,D,1,T]
    logits = net.top(seq_4d)
    if isinstance(logits, tuple):
        logits = logits[0]

    # Decode prediction for reference
    tdec = logits.argmax(2).permute(1, 0).cpu().numpy().squeeze()  # [T]
    pred_text = ctc_decode(tdec, i2c).strip()
    print(f"[INFO] Pred text : {pred_text}")

    # Target scalar: maximum logit across time & classes
    target = logits.max()
    net.zero_grad()
    target.backward()

    # ------------------------------------------------------------------
    # Gradients wrt patch tokens: Grad-CAM-style importance
    # ------------------------------------------------------------------
    grads = seq_tokens.grad.detach().cpu().numpy()  # [T,1,D]
    grads = np.squeeze(grads, axis=1)               # [T,D]

    # Importance per token = mean |grad| over channels
    token_importance = np.mean(np.abs(grads), axis=-1)  # [T]
    # Normalize
    token_importance -= token_importance.min()
    token_importance /= (token_importance.max() + 1e-8)

    # Map T tokens back to [Hp, Wp] grid
    if token_importance.shape[0] != Hp * Wp:
        print(
            f"[WARN] T={token_importance.shape[0]} != Hp*Wp={Hp*Wp}, "
            "cannot reshape nicely; skipping visualization."
        )
        return

    heat_grid = token_importance.reshape(Hp, Wp)

    # ------------------------------------------------------------------
    # Prepare image + overlay
    # ------------------------------------------------------------------
    img_pil = Image.open(img_path).convert("L")
    img_np = np.array(img_pil)  # [H_img, W_img]

    # Upsample heat_grid to image size
    H_img, W_img = img_np.shape
    heat_img = Image.fromarray((heat_grid * 255).astype(np.uint8))
    heat_img = heat_img.resize((W_img, H_img), resample=Image.BILINEAR)
    heat_up = np.array(heat_img) / 255.0

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title(f"Original image\nGT: {gt_text}\nPred: {pred_text}")
    axes[0].axis("off")

    axes[1].imshow(img_np, cmap="gray")
    axes[1].imshow(heat_up, cmap="jet", alpha=0.4)
    axes[1].set_title("Grad-CAM-style patch importance")
    axes[1].axis("off")

    plt.tight_layout()

    if args.save:
        out_path = args.save
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved Grad-CAM figure → {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
