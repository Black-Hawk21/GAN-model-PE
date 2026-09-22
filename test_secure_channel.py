"""
test_secure_channel.py -- CPU-only checks for the Cifrar secure link.

No GAN checkpoint or GPU needed.  Uses a small stand-in latent so the whole
suite runs in well under a minute.

    python test_secure_channel.py          # unit checks + attack table
    python test_secure_channel.py --full   # also the SNR sweep at real latent size (28x512)

Also works under pytest:  pytest test_secure_channel.py
"""

import argparse
import math
import sys
import time

import torch
import torch.nn as nn

from channel_attacks import ATTACK_REGISTRY, ReplayAttack, build_attack
from secure_channel import SecureLatentLink

SHAPE = (4, 64)          # small stand-in for (n_latent, 512)
FULL_SHAPE = (28, 512)   # CelebAMask-HQ-512 W+ latent


class AWGN(nn.Module):
    """Same maths as AWGN_Channel in main.py / evaluate.py."""
    def __init__(self, snr_db):
        super().__init__()
        self.std = 10 ** (-0.05 * snr_db)

    def forward(self, x):
        return x + torch.randn_like(x) * self.std


class Identity(nn.Module):
    def forward(self, x):
        return x


class FlipSymbols(nn.Module):
    """Flip the sign of chosen BPSK symbols (= flip those ciphertext bits)."""
    def __init__(self, idx):
        super().__init__()
        self.idx = list(idx)

    def forward(self, x):
        x = x.clone()
        x[..., self.idx] *= -1
        return x


def latent(batch=1, shape=SHAPE, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((batch, *shape), generator=g)
    return x / x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()   # PowerNormalize


def attack_args(name):
    return argparse.Namespace(
        attack=name, attack_power=10.0, flood_ratio=0.3, flood_value=5.0,
        replay_delay=1, spoof_gain=2.0, fading_type="rayleigh", rician_k=1.0,
        null_ratio=0.3, interleave_seed=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unit checks
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_roundtrip():
    link = SecureLatentLink(seed=7)
    x = latent(batch=3)
    y, reps = link(x, Identity(), Identity())
    assert [r.status for r in reps] == ["ok"] * 3
    assert y.shape == x.shape and y.dtype == x.dtype
    step = max(float(s.max() - s.min()) for s in x) / 255
    assert float((x - y).abs().max()) <= step / 2 + 1e-5
    assert link.stats.summary()["link_ber"] == 0.0


def test_16bit_is_finer():
    x = latent()
    y8, _ = SecureLatentLink(seed=7, quant_bits=8)(x, Identity(), Identity())
    y16, _ = SecureLatentLink(seed=7, quant_bits=16)(x, Identity(), Identity())
    assert float((x - y16).abs().max()) < float((x - y8).abs().max()) / 50


def test_wrong_seed_rejected():
    link = SecureLatentLink(seed=1)
    link.rx = SecureLatentLink(seed=2).rx          # receiver holds a different secret
    y, reps = link(latent(), Identity(), Identity())
    assert not reps[0].accepted and float(y.abs().max()) == 0.0


def test_noise_bit_flip_is_flagged_but_data_intact():
    # Cifrar chunk 0: 4-bit X, 32-bit checksum, then data bit at 36 and noise
    # bits at 37..36+X.  Flipping symbol 37 hits a noise bit: no data changes,
    # but the checksum fails.
    x = latent()
    clean, _ = SecureLatentLink(seed=7)(x, Identity(), Identity())

    y, reps = SecureLatentLink(seed=7, on_tamper="drop")(x, Identity(), FlipSymbols([37]))
    assert reps[0].status == "tampered" and not reps[0].accepted
    assert float(y.abs().max()) == 0.0               # erasure

    y, reps = SecureLatentLink(seed=7, on_tamper="keep")(x, Identity(), FlipSymbols([37]))
    assert reps[0].status == "tampered" and reps[0].accepted
    assert torch.equal(y, clean)                     # payload was never touched


def test_replay_strict_vs_monotonic_vs_off():
    x = latent(batch=4)
    expect = {
        "strict":    ["ok", "replayed", "replayed", "replayed"],
        "monotonic": ["ok", "replayed", "ok", "ok"],      # delayed stream slips through
        "off":       ["ok", "ok", "ok", "ok"],            # pure Cifrar: blind to replay
    }
    for mode, want in expect.items():
        link = SecureLatentLink(seed=7, replay_check=mode)
        y, reps = link(x, Identity(), ReplayAttack(replay_delay=1))
        got = [r.status for r in reps]
        assert got == want, f"{mode}: {got}"
    # With the check off, image 1 is silently reconstructed from image 0's packet.
    assert torch.allclose(y[1], y[0]) and not torch.allclose(y[1], x[1], atol=0.05)


def test_heavy_noise_rejected():
    link = SecureLatentLink(seed=7)
    y, reps = link(latent(batch=2), AWGN(0), Identity())
    assert all(not r.accepted for r in reps)
    assert link.stats.summary()["link_ber"] > 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Attack table
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED = {   # at 20 dB, 4 packets, strict replay check
    "none":           "accept",
    "pilot_spoofing": "accept",   # positive gain never flips a BPSK sign
    "fading":         "accept",   # |h| >= 0 applied after the noise: signs unchanged
    "jamming":        "reject",
    "bit_flooding":   "reject",
    "eavesdrop_null": "reject",
    "interleave":     "reject",
    "replay":         "first",    # first slot passes, stale packets rejected after
}


def test_attack_table(verbose=False):
    torch.manual_seed(0)
    rows = []
    for name in ATTACK_REGISTRY:
        link = SecureLatentLink(seed=7)
        _, reps = link(latent(batch=4), AWGN(20), build_attack(attack_args(name)))
        acc = [r.accepted for r in reps]
        want = EXPECTED[name]
        ok = (all(acc) if want == "accept" else
              not any(acc) if want == "reject" else
              acc == [True, False, False, False])
        rows.append((name, link.stats.summary(), [r.status for r in reps], ok))
        assert ok, f"{name}: {[r.status for r in reps]}"
    if verbose:
        print(f"\n  {'attack':<15}{'accepted':>9}{'link BER':>11}  statuses")
        for name, s, st, _ in rows:
            print(f"  {name:<15}{s['accepted']:>5}/{s['packets']:<3}{s['link_ber']:>11.2e}  {st}")


# ─────────────────────────────────────────────────────────────────────────────
# Full-size SNR sweep (slow: ~4 s per packet in pure Python)
# ─────────────────────────────────────────────────────────────────────────────

def snr_sweep(snrs=(10, 12, 13, 14, 15, 16, 20), packets=6):
    link = SecureLatentLink(seed=42)
    x = latent(batch=1, shape=FULL_SHAPE)
    print(f"\n  full-size latent {FULL_SHAPE}, {packets} packets per SNR")
    print(f"  {'SNR':>4} {'reject':>7} {'link BER':>10} {'theory BER':>11} "
          f"{'P(>=1 err) theory':>18}")
    for snr in snrs:
        link.reset_stats()
        t0 = time.time()
        for _ in range(packets):
            link(x, AWGN(snr), Identity())
        s = link.stats.summary()
        p = 0.5 * math.erfc(10 ** (snr / 20) / math.sqrt(2))       # Q(sqrt(SNR))
        p_fail = 1 - (1 - p) ** s["channel_uses"]
        print(f"  {snr:>4} {s['reject_rate']:>7.2f} {s['link_ber']:>10.2e} {p:>11.2e} "
              f"{p_fail:>18.3f}   ({(time.time() - t0) / packets:.1f}s/pkt, "
              f"{s['channel_uses']:.0f} symbols/img)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t0 = time.time()
        t(verbose=True) if t is test_attack_table else t()
        print(f"PASS  {t.__name__}  ({time.time() - t0:.1f}s)")
    if "--full" in sys.argv:
        snr_sweep()
    print("\nall checks passed")
