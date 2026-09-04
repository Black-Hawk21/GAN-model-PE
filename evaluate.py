"""
evaluate.py
===========
Evaluator for GAN-SeCom semantic communication model.

Sweeps across a range of SNR values, runs the model internally for each SNR,
and computes the following metrics against the original input images:

  • BER vs SNR      – Bit Error Rate of the quantised latent channel symbols
  • PSNR vs SNR     – Peak Signal-to-Noise Ratio of reconstructed images
  • SSIM vs SNR     – Structural Similarity Index (MS-SSIM)
  • Spectral Eff.   – Shannon/achievable spectral efficiency in bits/s/Hz

All plots are saved to <results_dir>/eval_<timestamp>/  and a CSV/JSON summary
is written there as well.

Usage
-----
    python evaluate.py \\
        --data_dir   data/examples \\
        --results_dir results \\
        --ckpt        pretrained/CelebAMask-HQ-512x512.pt \\
        --snr_range   -5 25 \\
        --snr_step    5 \\
        --attack      none

Run `python evaluate.py --help` for a full list of arguments.
"""

import os
import sys
import math
import argparse
import time
import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt

from tqdm import tqdm

# ── project imports ────────────────────────────────────────────────────────────
from models import make_model
from channel_attacks import build_attack, ATTACK_REGISTRY
from criteria.lpips import lpips as lpips_module

# optional – graceful fallback
try:
    import piq
    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False
    print("[warn] piq not found – using manual PSNR computation.")

try:
    from pytorch_msssim import ms_ssim as _ms_ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False
    print("[warn] pytorch_msssim not found – SSIM will use a simple approximation.")


# ══════════════════════════════════════════════════════════════════════════════
# Channel modules  (self-contained, mirrors main.py)
# ══════════════════════════════════════════════════════════════════════════════

class PowerNormalize(nn.Module):
    """Normalise transmitted tensor to unit average power."""
    def __init__(self, t_pow: float = 1.0):
        super().__init__()
        self.t_pow = t_pow

    def forward(self, x: torch.Tensor, dim=(1, 2)) -> torch.Tensor:
        pwr = torch.mean(x ** 2, dim=dim, keepdim=True)
        return math.sqrt(self.t_pow) * x / torch.sqrt(pwr)


class AWGN_Channel(nn.Module):
    """Additive White Gaussian Noise channel parameterised by SNR in dB."""
    def __init__(self, snr_db: float):
        super().__init__()
        self.change_snr(snr_db)

    def change_snr(self, snr_db: float):
        self.snr_db = snr_db
        self.std = 10 ** (-0.05 * snr_db)   # σ = 10^(–SNR_dB / 20)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std
        return x + noise


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_psnr(ref: torch.Tensor, hyp: torch.Tensor) -> float:
    """
    PSNR in dB between two [0,1]-range float tensors (B, C, H, W).
    """
    if HAS_PIQ:
        return float(piq.psnr(ref, hyp, data_range=1.0).mean().item())
    mse = F.mse_loss(hyp, ref, reduction="none").mean(dim=[1, 2, 3])
    psnr_vals = 10.0 * torch.log10(1.0 / (mse + 1e-12))
    return float(psnr_vals.mean().item())


def compute_ssim(ref: torch.Tensor, hyp: torch.Tensor) -> float:
    """
    MS-SSIM (or simple SSIM fallback) in [0, 1] between two [0,1] tensors.
    """
    if HAS_MSSSIM:
        return float(_ms_ssim(ref, hyp, data_range=1.0, size_average=True).item())
    # Simple global-stats SSIM approximation
    mu_r = ref.mean();  mu_h = hyp.mean()
    sig_r = ref.std();  sig_h = hyp.std()
    cov   = ((ref - mu_r) * (hyp - mu_h)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return float(
        ((2*mu_r*mu_h + c1) * (2*cov + c2)) /
        ((mu_r**2 + mu_h**2 + c1) * (sig_r**2 + sig_h**2 + c2))
    )


def compute_ber(tx: torch.Tensor, rx: torch.Tensor, n_bits: int = 8) -> float:
    """
    Estimate Bit Error Rate from floating-point latent vectors.

    Steps:
      1. Quantise both to n_bits uniform levels (range anchored to tx).
      2. XOR integer indices → count bit differences via np.unpackbits.

    Parameters
    ----------
    tx, rx  : transmitted / received latent tensors (any shape)
    n_bits  : quantisation resolution (default 8 → 256 levels)
    """
    tx_flat = tx.detach().cpu().float().reshape(-1)
    rx_flat = rx.detach().cpu().float().reshape(-1)

    lo = float(tx_flat.min())
    hi = float(tx_flat.max()) + 1e-9
    levels = (2 ** n_bits) - 1

    tx_q = ((tx_flat - lo) / (hi - lo) * levels).round().long().clamp(0, levels)
    rx_q = ((rx_flat - lo) / (hi - lo) * levels).round().long().clamp(0, levels)

    diff     = (tx_q ^ rx_q).numpy().astype(np.uint64)
    # View as bytes and count set bits
    bit_errs = int(np.unpackbits(diff.view(np.uint8)).sum())
    total_bits = len(tx_flat) * n_bits
    return float(bit_errs) / float(total_bits)


def compute_spectral_efficiency(snr_db: float) -> float:
    """
    Shannon channel capacity (spectral efficiency) in bits/s/Hz.

        SE = log2(1 + SNR_linear)
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    return float(math.log2(1.0 + snr_lin))


def compute_effective_se(
    tx: torch.Tensor,
    rx: torch.Tensor,
    snr_db: float,
    latent_dim: int,
    image_pixels: int,
) -> float:
    """
    Effective spectral efficiency, accounting for compression ratio and BER:

        SE_eff = (1 – BER) × log2(1+SNR) × (latent_dim / image_pixels)

    The compression ratio maps raw Shannon capacity to the semantic
    operating point (fewer channel uses than source pixels).
    """
    ber  = compute_ber(tx, rx)
    se   = compute_spectral_efficiency(snr_db)
    cr   = latent_dim / max(image_pixels, 1)
    return float((1.0 - ber) * se * cr)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class EvalDataset(torch.utils.data.Dataset):
    """Load all .jpg / .png images from a directory, sorted alphabetically."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, data_dir: str, size: int = 512):
        self.paths = sorted(
            p for p in Path(data_dir).iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in '{data_dir}'.")
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), str(self.paths[idx])


# ══════════════════════════════════════════════════════════════════════════════
# Latent optimiser  (mirrors main.py's optimize_latent)
# ══════════════════════════════════════════════════════════════════════════════

def _lr_schedule(t: float, lr0: float, rampdown=0.25, rampup=0.05) -> float:
    r = min(1.0, (1.0 - t) / rampdown)
    r = 0.5 - 0.5 * math.cos(r * math.pi)
    r = r * min(1.0, t / rampup)
    return lr0 * r


@torch.no_grad()
def _sample_latent_mean(g_ema, device: str, n: int = 10_000) -> torch.Tensor:
    noise_sample = torch.randn(n, 512, device=device)
    return g_ema.style(noise_sample).mean(0)


def optimize_latent(
    args,
    g_ema,
    target: torch.Tensor,
    p_norm: PowerNormalize,
    channel: AWGN_Channel,
    attack: nn.Module,
    latent_mean: torch.Tensor,
    percept,
    device: str,
) -> torch.Tensor:
    """
    Run latent-space optimisation for a single image batch.

    Returns
    -------
    torch.Tensor – optimised W+ (or W) latent at the final step.
    """
    B = target.shape[0]
    noises = g_ema.render_net.get_noise(noise=None, randomize_noise=False)
    for n in noises:
        n.requires_grad = False

    if args.w_plus:
        latent_in = (
            latent_mean.detach().clone()
            .unsqueeze(0).repeat(B, 1)
            .unsqueeze(1).repeat(1, g_ema.n_latent, 1)
        )
    else:
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(B, 1)
    latent_in.requires_grad = True

    optimizer = optim.Adam([latent_in], lr=args.lr)

    for i in range(args.step):
        optimizer.zero_grad()
        optimizer.param_groups[0]["lr"] = _lr_schedule(i / args.step, args.lr)

        tx = channel(p_norm(latent_in, dim=(1, 2)))
        rx = attack(tx)
        img_gen, _ = g_ema([rx], input_is_latent=True,
                           randomize_noise=False, noise=None)

        # Perceptual loss
        p_gen = F.adaptive_avg_pool2d(img_gen, (256, 256))
        p_tgt = F.adaptive_avg_pool2d(target,  (256, 256))
        p_loss  = percept(p_gen, p_tgt).mean()
        l1_loss = F.mse_loss(img_gen, target)

        if args.w_plus:
            mean_loss = F.mse_loss(
                latent_in,
                latent_mean.unsqueeze(0).repeat(B, g_ema.n_latent, 1)
            )
        else:
            mean_loss = F.mse_loss(latent_in, latent_mean.repeat(B, 1))

        loss = (
            p_loss    * args.lambda_lpips +
            l1_loss   * args.lambda_l1 +
            mean_loss * args.lambda_mean
        )
        loss.backward()
        optimizer.step()

    return latent_in.detach()


# ══════════════════════════════════════════════════════════════════════════════
# Core evaluation loop for a single SNR point
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_snr_point(
    args,
    g_ema,
    p_norm: PowerNormalize,
    channel: AWGN_Channel,
    attack: nn.Module,
    data_loader,
    latent_mean: torch.Tensor,
    percept,
    device: str,
    snr_db: float,
    out_img_dir: Path,
) -> Dict[str, float]:
    """
    Evaluate all images at a given SNR level.

    Returns
    -------
    dict with keys: psnr, ssim, ber, spectral_eff, effective_se
    """
    channel.change_snr(snr_db)

    psnr_acc, ssim_acc, ber_acc, se_acc, eff_acc = [], [], [], [], []
    image_pixels = args.size * args.size * 3   # H × W × C

    for images, paths in tqdm(data_loader,
                              desc=f"  SNR={snr_db:+.0f} dB",
                              leave=False):
        images = images.to(device)

        # Optimise latent for this image
        latent_opt = optimize_latent(
            args, g_ema, images, p_norm, channel, attack,
            latent_mean, percept, device,
        )

        with torch.no_grad():
            # Forward pass through channel
            tx = p_norm(latent_opt, dim=(1, 2))
            rx = attack(channel(tx))

            # Decode
            img_gen, _ = g_ema([rx], input_is_latent=True,
                               randomize_noise=False, noise=None)

            # Scale to [0, 1]
            ref_01 = images.clamp(-1, 1) * 0.5 + 0.5
            hyp_01 = img_gen.clamp(-1, 1) * 0.5 + 0.5

            # Image quality metrics
            psnr_acc.append(compute_psnr(ref_01, hyp_01))
            ssim_acc.append(compute_ssim(ref_01, hyp_01))

            # Channel / communication metrics
            ber_acc.append(compute_ber(tx, rx))
            se_acc.append(compute_spectral_efficiency(snr_db))

            latent_dim = int(tx.reshape(tx.shape[0], -1).shape[1])
            eff_acc.append(compute_effective_se(tx, rx, snr_db, latent_dim, image_pixels))

            # Save reconstructed images
            imgs_np = (hyp_01.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            for i, arr in enumerate(imgs_np):
                fname = Path(paths[i]).name
                Image.fromarray(arr).save(out_img_dir / fname)

    return {
        "psnr":         float(np.mean(psnr_acc)),
        "ssim":         float(np.mean(ssim_acc)),
        "ber":          float(np.mean(ber_acc)),
        "spectral_eff": float(np.mean(se_acc)),
        "effective_se": float(np.mean(eff_acc)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

_C = ["#4C8BF5", "#E8544A", "#34A853", "#FBBC04", "#AA46BB"]   # colour palette
_S = dict(linewidth=2.2, marker="o", markersize=7)              # shared line style


def _savefig(fig: plt.Figure, path: Path):
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path.name}")


def plot_ber_vs_snr(snr_list, ber_list, out_dir, attack_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(snr_list, ber_list, color=_C[0], **_S, label="BER")
    ax.set(xlabel="SNR (dB)", ylabel="Bit Error Rate (log scale)",
           title=f"BER vs SNR  [attack: {attack_name}]")
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend(fontsize=11)
    fig.tight_layout()
    _savefig(fig, out_dir / "ber_vs_snr.png")


def plot_psnr_vs_snr(snr_list, psnr_list, out_dir, attack_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(snr_list, psnr_list, color=_C[1], **_S, label="PSNR")
    ax.set(xlabel="SNR (dB)", ylabel="PSNR (dB)",
           title=f"PSNR vs SNR  [attack: {attack_name}]")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(fontsize=11)
    fig.tight_layout()
    _savefig(fig, out_dir / "psnr_vs_snr.png")


def plot_ssim_vs_snr(snr_list, ssim_list, out_dir, attack_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(snr_list, ssim_list, color=_C[2], **_S, label="MS-SSIM")
    ax.set(xlabel="SNR (dB)", ylabel="MS-SSIM",
           title=f"SSIM vs SNR  [attack: {attack_name}]")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(fontsize=11)
    fig.tight_layout()
    _savefig(fig, out_dir / "ssim_vs_snr.png")


def plot_spectral_efficiency(snr_list, se_list, eff_list, out_dir, attack_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(snr_list, se_list,  color=_C[3], **_S, label="Shannon capacity  log\u2082(1+SNR)")
    ax.plot(snr_list, eff_list, color=_C[4], **_S, linestyle="--",
            label="Effective SE  \u00d7 CR \u00d7 (1\u2212BER)")
    ax.set(xlabel="SNR (dB)", ylabel="Spectral Efficiency (bits/s/Hz)",
           title=f"Spectral Efficiency vs SNR  [attack: {attack_name}]")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, out_dir / "spectral_efficiency_vs_snr.png")


def plot_combined(snr_list, results, out_dir, attack_name):
    """Four-panel summary figure."""
    ber_l  = [r["ber"]          for r in results]
    psnr_l = [r["psnr"]         for r in results]
    ssim_l = [r["ssim"]         for r in results]
    se_l   = [r["spectral_eff"] for r in results]
    eff_l  = [r["effective_se"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"GAN-SeCom Evaluation  |  Attack: {attack_name}",
        fontsize=15, fontweight="bold", y=1.01,
    )

    axes[0, 0].semilogy(snr_list, ber_l,  color=_C[0], **_S)
    axes[0, 0].set(xlabel="SNR (dB)", ylabel="BER (log)", title="BER vs SNR")
    axes[0, 0].grid(True, which="both", linestyle="--", alpha=0.4)

    axes[0, 1].plot(snr_list, psnr_l, color=_C[1], **_S)
    axes[0, 1].set(xlabel="SNR (dB)", ylabel="PSNR (dB)", title="PSNR vs SNR")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    axes[1, 0].plot(snr_list, ssim_l, color=_C[2], **_S)
    axes[1, 0].set(xlabel="SNR (dB)", ylabel="MS-SSIM", title="SSIM vs SNR")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    axes[1, 1].plot(snr_list, se_l,  color=_C[3], **_S, label="Shannon capacity")
    axes[1, 1].plot(snr_list, eff_l, color=_C[4], **_S, linestyle="--", label="Effective SE")
    axes[1, 1].set(xlabel="SNR (dB)", ylabel="bits/s/Hz",
                   title="Spectral Efficiency vs SNR")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)
    axes[1, 1].legend(fontsize=9)

    for ax in axes.flat:
        ax.tick_params(labelsize=10)
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontsize(11)

    fig.tight_layout()
    _savefig(fig, out_dir / "combined_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    def parse_bool(x):
        return x not in ("False", "false", "0")

    p = argparse.ArgumentParser(
        description="GAN-SeCom evaluator: BER, PSNR, SSIM & Spectral Efficiency vs SNR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",     type=str, default="data/examples",
                   help="Directory of input images (all .jpg/.png will be used).")
    p.add_argument("--results_dir",  type=str, default="results",
                   help="Root directory; a timestamped sub-dir is created here.")
    p.add_argument("--ckpt",         type=str,
                   default="pretrained/CelebAMask-HQ-512x512.pt",
                   help="Path to the model checkpoint (.pt).")
    # ── model ────────────────────────────────────────────────────────────────
    p.add_argument("--size",         type=int,        default=512)
    p.add_argument("--w_plus",       type=parse_bool, default=True,
                   help="Optimise in W+ space (True) or W space (False).")
    p.add_argument("--no_noises",    type=parse_bool, default=True)
    p.add_argument("--step",         type=int,        default=300,
                   help="Latent optimisation steps per image.")
    p.add_argument("--lr",           type=float,      default=0.1)
    p.add_argument("--lambda_l1",    type=float,      default=0.3)
    p.add_argument("--lambda_lpips", type=float,      default=1.0)
    p.add_argument("--lambda_ssim",  type=float,      default=0.0)
    p.add_argument("--lambda_mean",  type=float,      default=0.0)
    # ── SNR sweep ────────────────────────────────────────────────────────────
    p.add_argument("--snr_range", type=float, nargs=2, default=[-5, 25],
                   metavar=("SNR_MIN", "SNR_MAX"),
                   help="Inclusive SNR sweep range in dB.")
    p.add_argument("--snr_step",  type=float, default=5,
                   help="SNR step size in dB.")
    p.add_argument("--snr_list",  type=float, nargs="+", default=None,
                   help="Explicit SNR values to test (overrides --snr_range/--snr_step).")
    # ── channel attack ───────────────────────────────────────────────────────
    p.add_argument("--attack",          type=str,   default="none",
                   choices=list(ATTACK_REGISTRY.keys()),
                   help="Channel attack type.")
    p.add_argument("--attack_power",    type=float, default=10.0)
    p.add_argument("--flood_ratio",     type=float, default=0.3)
    p.add_argument("--flood_value",     type=float, default=5.0)
    p.add_argument("--replay_delay",    type=int,   default=1)
    p.add_argument("--spoof_gain",      type=float, default=2.0)
    p.add_argument("--fading_type",     type=str,   default="rayleigh",
                   choices=["rayleigh", "rician"])
    p.add_argument("--rician_k",        type=float, default=1.0)
    p.add_argument("--null_ratio",      type=float, default=0.3)
    p.add_argument("--interleave_seed", type=int,   default=0)
    # ── misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--device",     type=str, default="cuda",
                   help="Compute device: 'cuda' or 'cpu'.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed",       type=int, default=42)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"[info] '{args.device}' unavailable, falling back to '{device}'.")

    # ── output directory ──────────────────────────────────────────────────────
    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_dir) / f"eval_{ts}_{args.attack}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[eval] Results will be saved to: {out_dir}\n")

    # ── logging ───────────────────────────────────────────────────────────────
    log_path = out_dir / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)
    log.info(f"Arguments: {vars(args)}")

    # ── SNR sweep values ──────────────────────────────────────────────────────
    if args.snr_list is not None:
        snr_values = sorted(args.snr_list)
    else:
        lo, hi = args.snr_range
        snr_values = list(np.arange(lo, hi + 1e-9, args.snr_step))
    log.info(f"SNR sweep: {snr_values} dB")

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = EvalDataset(args.data_dir, size=args.size)
    log.info(f"Dataset: {len(dataset)} images from '{args.data_dir}'")
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )

    # ── model + channel setup ─────────────────────────────────────────────────
    log.info(f"Loading checkpoint: {args.ckpt}")
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    g_ema = make_model(ckpt["args"])
    g_ema.to(device).eval()
    g_ema.load_state_dict(ckpt["g_ema"])

    percept = lpips_module.LPIPS(net_type="vgg").to(device)
    p_norm  = PowerNormalize(t_pow=1.0).to(device)
    channel = AWGN_Channel(snr_db=snr_values[0]).to(device)
    attack  = build_attack(args).to(device)
    log.info(f"Channel attack: {attack}")

    # Pre-compute the W-space mean once (expensive but amortised)
    log.info("Sampling latent mean (W-space average) …")
    with torch.no_grad():
        latent_mean = _sample_latent_mean(g_ema, device)

    # ── main evaluation sweep ─────────────────────────────────────────────────
    all_results: List[Dict] = []

    for snr_db in snr_values:
        log.info(f"\n{'─'*60}")
        log.info(f"Evaluating  SNR = {snr_db:+.1f} dB")

        snr_img_dir = out_dir / f"recon_snr{snr_db:+.0f}dB"
        snr_img_dir.mkdir(parents=True, exist_ok=True)

        metrics = evaluate_snr_point(
            args, g_ema, p_norm, channel, attack,
            loader, latent_mean, percept, device,
            snr_db=snr_db,
            out_img_dir=snr_img_dir,
        )
        metrics["snr_db"] = snr_db
        all_results.append(metrics)

        log.info(
            f"  PSNR={metrics['psnr']:.2f} dB | "
            f"SSIM={metrics['ssim']:.4f} | "
            f"BER={metrics['ber']:.4e} | "
            f"SE={metrics['spectral_eff']:.3f} b/s/Hz | "
            f"Eff.SE={metrics['effective_se']:.4f}"
        )

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / "metrics.csv"
    headers  = ["snr_db", "psnr", "ssim", "ber", "spectral_eff", "effective_se"]
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in all_results:
            f.write(",".join(str(r[h]) for h in headers) + "\n")
    log.info(f"\n[saved] CSV  → {csv_path}")

    # ── save JSON ─────────────────────────────────────────────────────────────
    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"[saved] JSON → {json_path}")

    # ── generate plots ────────────────────────────────────────────────────────
    snr_l  = [r["snr_db"]       for r in all_results]
    ber_l  = [r["ber"]          for r in all_results]
    psnr_l = [r["psnr"]         for r in all_results]
    ssim_l = [r["ssim"]         for r in all_results]
    se_l   = [r["spectral_eff"] for r in all_results]
    eff_l  = [r["effective_se"] for r in all_results]

    log.info("\nGenerating plots …")
    plot_ber_vs_snr(snr_l, ber_l,          out_dir, args.attack)
    plot_psnr_vs_snr(snr_l, psnr_l,        out_dir, args.attack)
    plot_ssim_vs_snr(snr_l, ssim_l,        out_dir, args.attack)
    plot_spectral_efficiency(snr_l, se_l, eff_l, out_dir, args.attack)
    plot_combined(snr_l, all_results,       out_dir, args.attack)

    # ── terminal summary table ────────────────────────────────────────────────
    col_w = [9, 10, 9, 11, 15, 12]
    hdr   = ["SNR (dB)", "PSNR (dB)", "MS-SSIM", "BER", "SE (b/s/Hz)", "Eff. SE"]
    sep   = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt   = lambda vals: "|" + "|".join(
        f"{str(v):^{w}}" for v, w in zip(vals, col_w)
    ) + "|"

    print("\n" + sep)
    print(fmt(hdr))
    print(sep)
    for r in all_results:
        print(fmt([
            f"{r['snr_db']:+.0f}",
            f"{r['psnr']:.2f}",
            f"{r['ssim']:.4f}",
            f"{r['ber']:.2e}",
            f"{r['spectral_eff']:.3f}",
            f"{r['effective_se']:.4f}",
        ]))
    print(sep)

    log.info(f"\n[done] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
