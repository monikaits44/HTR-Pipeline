#!/usr/bin/env python3
import argparse
import os
import sys
import csv  # <-- NEW

from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

# -------------------------------------------------------------------------
# 1) Make sure we can import from project root (models.py, utils/, etc.)
# -------------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now these imports should work exactly like in evaluate.py / trainer.py
from models import HTRNet
from utils.htr_dataset import HTRDataset
from utils.metrics import CER, WER


def parse_args():
    """
    Match the same config merging behavior as trainer.py:

    - First YAML: base config (configs/config.yaml)
    - Additional YAMLs: merged in order (configs/baseline.yaml, configs/baseline_vit_rgts.yaml)
    - key=value args: treated as overrides (e.g., resume=..., device=cuda)
    """
    # No arguments at all?
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python extract_vit_rgts_features.py "
            "<config.yaml> [extra.yaml ...] [key=value ...]"
        )

    # 1) Load first YAML
    conf = OmegaConf.load(sys.argv[1])

    # 2) Merge any additional YAMLs
    for arg in sys.argv[2:]:
        if arg.endswith(".yaml"):
            extra_conf = OmegaConf.load(arg)
            conf = OmegaConf.merge(conf, extra_conf)

    # 3) Apply key=value overrides (resume=..., device=..., etc.)
    for arg in sys.argv[2:]:
        if "=" in arg and not arg.endswith(".yaml"):
            key, value_str = arg.split("=", 1)
            # Try to interpret basic Python literals (int/float/bool)
            try:
                value = eval(value_str, {"__builtins__": {}})
            except Exception:
                value = value_str
            override = OmegaConf.create({key: value})
            conf = OmegaConf.merge(conf, override)

    OmegaConf.set_struct(conf, True)
    return conf


def build_dataloader(config):
    """
    Build test loader with batch_size=1 for clean per-sample dumping.
    Re-uses the same HTRDataset used in trainer/evaluate.
    """
    dataset_folder = config.data.path
    fixed_size = (config.preproc.image_height, config.preproc.image_width)

    test_set = HTRDataset(dataset_folder, "test", fixed_size=fixed_size, transforms=None)
    print("# testing lines", len(test_set))

    # force batch_size = 1 for easy per-sample handling
    loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=config.eval.num_workers,
    )

    # load classes from training set
    classes = np.load(os.path.join(dataset_folder, "classes.npy"))

    # char↔index dicts
    cdict = {c: (i + 1) for i, c in enumerate(classes)}
    icdict = {(i + 1): c for i, c in enumerate(classes)}

    classes_dict = {
        "classes": classes,
        "c2i": cdict,
        "i2c": icdict,
    }

    return test_set, loader, classes_dict


def build_model(config, classes):
    """
    Instantiate HTRNet exactly like trainer/evaluate, then load weights
    from config.resume.
    """
    device = config.device

    print("Preparing Net - Architectural elements:")
    print(config.arch)

    net = HTRNet(config.arch, len(classes) + 1)

    if config.resume is None:
        raise ValueError(
            "config.resume is None – you must point to a trained checkpoint "
            "(e.g. resume=./saved_models/experiments/run_24/model.pt)"
        )

    print(f"Loading checkpoint from: {config.resume}")
    checkpoint = torch.load(config.resume, map_location="cpu")
    load_status = net.load_state_dict(checkpoint, strict=True)
    print(load_status)

    net.to(device)
    net.eval()

    # print number of parameters
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of parameters:", n_params)

    # Sanity: ensure this is the ViT+Registers variant
    arch_type = getattr(config.arch, "type", "cnn_rnn")
    if arch_type != "vit_rgts":
        raise ValueError(
            f"Expected arch.type == 'vit_rgts', but got '{arch_type}'. "
            "Make sure you are using baseline_vit_rgts.yaml in your config."
        )

    # Also check model has a ViT backbone
    if not hasattr(net, "backbone"):
        raise AttributeError(
            "HTRNet has no attribute 'backbone'. "
            "Double-check that your models.py contains the ViTRGTSBackbone "
            "and that arch.type is wired correctly."
        )

    return net


def ctc_decode(tdec, tdict, blank_id=0):
    """
    Same decoding logic as in trainer/evaluate: remove duplicates + blanks.
    tdec: [T] numpy array of predicted class indices.
    """
    tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
    dec_transcr = "".join([tdict[t] for t in tt if t != blank_id])
    return dec_transcr


def main():
    config = parse_args()

    # ------------------------------------------------------------------
    # 1) Build dataloader + classes
    # ------------------------------------------------------------------
    test_set, test_loader, classes_dict = build_dataloader(config)
    classes = classes_dict["classes"]
    i2c = classes_dict["i2c"]

    # ------------------------------------------------------------------
    # 2) Build model (ViT+Registers)
    # ------------------------------------------------------------------
    device = config.device
    net = build_model(config, classes)

    # ------------------------------------------------------------------
    # 3) Prepare output directory (alongside model.pt in run_XX)
    # ------------------------------------------------------------------
    run_dir = os.path.dirname(os.path.abspath(config.resume))
    out_dir = os.path.join(run_dir, "vit_rgts_explain")
    os.makedirs(out_dir, exist_ok=True)

    # CSV with per-sample info — use csv.writer with quoting
    csv_path = os.path.join(out_dir, "vit_rgts_features.csv")
    csv_f = open(csv_path, "w", encoding="utf-8", newline="")  # newline="" for csv
    writer = csv.writer(csv_f)  # default quoting handles commas in fields

    # Header
    writer.writerow([
        "sample_idx",
        "img_path",
        "gt_text",
        "pred_text",
        "cer",
        "wer",
        "num_registers",
        "num_patches",
        "Hp",
        "Wp",
    ])
    csv_f.flush()

    # Global metric accumulators
    cer_meter = CER()
    wer_meter = WER(mode=config.eval.wer_mode)

    # ------------------------------------------------------------------
    # 4) Iterate over test set, extract ViT features
    # ------------------------------------------------------------------
    print("\nExtracting ViT+Registers features on TEST set...")
    sample_idx = 0

    for (imgs, transcrs) in tqdm.tqdm(test_loader):
        # Batch size = 1 by construction
        imgs = imgs.to(device)  # [1, 1, H, W]
        gt_text = transcrs[0].strip()
        img_path = test_set.data[sample_idx][0]  # original image path from dataset

        with torch.no_grad():
            # ----------------------------------------------------------
            # (a) Get transformer tokens from backbone
            #     seq_tokens: [T, B, D]
            #     reg_tokens: [B, R, D]
            #     grid_size : (Hp, Wp)
            # ----------------------------------------------------------
            seq_tokens, reg_tokens, (Hp, Wp) = net.backbone(imgs)

            # For CTC head, backbone time dimension [T, B, D] → [B, D, 1, T]
            seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)  # [B, D, 1, T]

            # Forward through CTC head
            logits = net.top(seq_4d)
            if isinstance(logits, tuple):
                # head_type='both' style – first is RNN logits
                logits = logits[0]

        # --------------------------------------------------------------
        # (b) CTC decoding to get predicted text
        # --------------------------------------------------------------
        tdec = logits.argmax(2).permute(1, 0).cpu().numpy().squeeze()  # [T]
        pred_text = ctc_decode(tdec, i2c).strip()

        # --------------------------------------------------------------
        # (c) Metrics per sample
        # --------------------------------------------------------------
        cer_sample = CER()
        wer_sample = WER(mode=config.eval.wer_mode)
        cer_sample.update(pred_text, gt_text)
        wer_sample.update(pred_text, gt_text)
        cer_val = cer_sample.score()
        wer_val = wer_sample.score()

        cer_meter.update(pred_text, gt_text)
        wer_meter.update(pred_text, gt_text)

        # --------------------------------------------------------------
        # (d) Token norms (registers + patches)
        # --------------------------------------------------------------
        # seq_tokens: [T, B, D] → [B, T, D]
        seq_bt = seq_tokens.permute(1, 0, 2)  # [B, T, D]
        # reg_tokens: [B, R, D]
        # Concatenate for analysis: [B, R+T, D]
        tokens_all = torch.cat([reg_tokens, seq_bt], dim=1)
        token_norms = tokens_all.norm(dim=-1).cpu().numpy()[0]  # [R+T]

        # Save per-sample numpy blobs
        np.save(
            os.path.join(out_dir, f"sample_{sample_idx:05d}_reg_tokens.npy"),
            reg_tokens.cpu().numpy(),
        )
        np.save(
            os.path.join(out_dir, f"sample_{sample_idx:05d}_seq_tokens.npy"),
            seq_tokens.cpu().numpy(),
        )
        np.save(
            os.path.join(out_dir, f"sample_{sample_idx:05d}_token_norms.npy"),
            token_norms,
        )
        np.save(
            os.path.join(out_dir, f"sample_{sample_idx:05d}_logits.npy"),
            logits.cpu().numpy(),
        )

        # Log CSV row — csv.writer will auto-quote gt_text/pred_text if needed
        num_registers = int(reg_tokens.shape[1])
        num_patches = int(seq_tokens.shape[0])

        writer.writerow([
            int(sample_idx),
            img_path,
            gt_text,
            pred_text,
            f"{cer_val:.6f}",
            f"{wer_val:.6f}",
            num_registers,
            num_patches,
            int(Hp),
            int(Wp),
        ])
        csv_f.flush()

        sample_idx += 1

    # ------------------------------------------------------------------
    # 5) Global summary
    # ------------------------------------------------------------------
    print("\nExtraction complete!")
    print(f"Saved per-sample features to: {out_dir}")
    print(f"CSV summary: {csv_path}")
    print(f"TEST CER: {cer_meter.score():.4f}")
    print(f"TEST WER: {wer_meter.score():.4f}")

    csv_f.close()


if __name__ == "__main__":
    main()
