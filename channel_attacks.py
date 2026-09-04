"""
channel_attacks.py
==================
Wireless channel attack simulations for GAN-based semantic communication.

Each attack is an nn.Module that acts on the power-normalised latent tensor
that has already passed through the AWGN channel, i.e. the signal at the
*receiver* side after undergoing an adversarial intervention.

Attack classes
--------------
NoAttack           – passthrough, no distortion
JammingAttack      – constant-power additive Gaussian jammer
BitFloodingAttack  – saturates / clips a fraction of latent entries
ReplayAttack       – replaces signal with a previously stored (stale) copy
PilotSpoofingAttack– scales signal by a malicious channel-estimate gain
FadingAttack       – applies Rayleigh or Rician fast-fading
EavesdropNullAttack– zeros out a random subset of latent dimensions
InterleaveAttack   – randomly permutes latent dimensions

Usage
-----
    from channel_attacks import build_attack
    attack = build_attack(args)          # args comes from argparse
    attacked_latent = attack(latent)     # forward pass
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_ratio(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"'{name}' must be in [0, 1], got {value}")


# ---------------------------------------------------------------------------
# 1. No Attack (passthrough)
# ---------------------------------------------------------------------------

class NoAttack(nn.Module):
    """Identity – no attack applied. Useful as the baseline."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __repr__(self) -> str:
        return "NoAttack()"


# ---------------------------------------------------------------------------
# 2. Jamming Attack
# ---------------------------------------------------------------------------

class JammingAttack(nn.Module):
    """
    Constant-Power Gaussian Jammer.

    Models a persistent adversary that injects high-power white Gaussian
    noise into the channel independently of the legitimate signal.  The
    jammer power (noise std) is set by ``attack_power``.

    Parameters
    ----------
    attack_power : float
        Standard deviation of the injected Gaussian noise.  Higher values
        produce stronger jamming.  Typical range: 0.5 – 20.
    """

    def __init__(self, attack_power: float = 10.0):
        super().__init__()
        if attack_power <= 0:
            raise ValueError("attack_power must be positive.")
        self.attack_power = attack_power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        jam_noise = torch.randn_like(x) * self.attack_power
        return x + jam_noise

    def __repr__(self) -> str:
        return f"JammingAttack(attack_power={self.attack_power})"


# ---------------------------------------------------------------------------
# 3. Bit-Flooding Attack
# ---------------------------------------------------------------------------

class BitFloodingAttack(nn.Module):
    """
    Bit / Data Flooding Attack.

    An adversary floods a random fraction of the transmitted latent
    dimensions with a large fixed value, effectively saturating those
    entries at the receiver.  This models denial-of-service style attacks
    where the attacker overwhelms part of the channel with bogus data.

    Parameters
    ----------
    flood_ratio : float
        Fraction of latent entries to corrupt, in [0, 1].
    flood_value : float
        Absolute value used to overwrite selected entries.
        Selected entries are set to ±flood_value (sign chosen randomly).
    """

    def __init__(self, flood_ratio: float = 0.3, flood_value: float = 5.0):
        super().__init__()
        _check_ratio("flood_ratio", flood_ratio)
        if flood_value <= 0:
            raise ValueError("flood_value must be positive.")
        self.flood_ratio = flood_ratio
        self.flood_value = flood_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build a binary mask over the flat last dimension(s)
        mask = torch.rand_like(x) < self.flood_ratio          # True = flood
        # Random sign for the flood value to be more realistic
        sign = (torch.randint(0, 2, x.shape, device=x.device) * 2 - 1).float()
        flooded = torch.where(mask, sign * self.flood_value, x)
        return flooded

    def __repr__(self) -> str:
        return (f"BitFloodingAttack(flood_ratio={self.flood_ratio}, "
                f"flood_value={self.flood_value})")


# ---------------------------------------------------------------------------
# 4. Replay Attack
# ---------------------------------------------------------------------------

class ReplayAttack(nn.Module):
    """
    Replay Attack.

    The adversary intercepts and stores a previous transmission and replays
    it to the receiver in place of the current one.  This is modelled by
    maintaining a small internal buffer of past tensors.  When the buffer
    is not yet full (initial steps), the current signal is returned
    unchanged (attacker is still recording).

    Parameters
    ----------
    replay_delay : int
        Number of forward passes to buffer before starting to replay.
        For example, replay_delay=1 means the previous batch is replayed.
    """

    def __init__(self, replay_delay: int = 1):
        super().__init__()
        if replay_delay < 1:
            raise ValueError("replay_delay must be >= 1.")
        self.replay_delay = replay_delay
        # Use a list so PyTorch does not treat these as parameters
        self._buffer: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._buffer.append(x.detach().clone())
        if len(self._buffer) <= self.replay_delay:
            # Buffer not full yet – pass current signal (attacker recording)
            return x
        # Return the stale signal from `replay_delay` steps ago
        stale = self._buffer[-(self.replay_delay + 1)]
        # Trim buffer to avoid unbounded growth
        if len(self._buffer) > self.replay_delay + 2:
            self._buffer.pop(0)
        return stale

    def reset_buffer(self) -> None:
        """Clear the replay buffer (call between independent runs)."""
        self._buffer.clear()

    def __repr__(self) -> str:
        return f"ReplayAttack(replay_delay={self.replay_delay})"


# ---------------------------------------------------------------------------
# 5. Pilot Spoofing Attack
# ---------------------------------------------------------------------------

class PilotSpoofingAttack(nn.Module):
    """
    Pilot Spoofing / False Channel-Estimate Attack.

    In OFDM-based semantic comms the receiver uses pilot symbols to
    estimate the channel and equalise the received signal.  A pilot
    spoofer transmits malicious pilots that force the receiver to use a
    wrong channel estimate, effectively scaling the signal by an incorrect
    gain.  This is modelled as a deterministic multiplicative distortion.

    Parameters
    ----------
    spoof_gain : float
        Multiplicative factor applied to the received latent.
        Values != 1.0 represent a mis-equalised channel.
    """

    def __init__(self, spoof_gain: float = 2.0):
        super().__init__()
        self.spoof_gain = spoof_gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.spoof_gain

    def __repr__(self) -> str:
        return f"PilotSpoofingAttack(spoof_gain={self.spoof_gain})"


# ---------------------------------------------------------------------------
# 6. Fading Attack (Rayleigh / Rician)
# ---------------------------------------------------------------------------

class FadingAttack(nn.Module):
    """
    Fast-Fading Channel Attack.

    Simulates multipath fast fading on top of the existing AWGN, modelling
    an environment where the attacker exploits or induces additional fading
    (e.g. by blocking the LoS path or creating artificial multipath).

    Two fading models are supported:

    * **Rayleigh** – no line-of-sight component.  Each element of the
      latent is multiplied by the magnitude of a complex Gaussian:
          h ~ CN(0, 1)  →  |h| ~ Rayleigh(1/√2)

    * **Rician** – with a dominant LoS component of power K relative to
      the scattered power.  The LoS component is fixed at 1+0j.

    Parameters
    ----------
    fading_type : str
        ``'rayleigh'`` or ``'rician'``.
    rician_k : float
        Rician K-factor (linear, not dB).  Only used when
        fading_type='rician'.  Typical values: 1 – 10.
    """

    def __init__(self, fading_type: str = "rayleigh", rician_k: float = 1.0):
        super().__init__()
        fading_type = fading_type.lower()
        if fading_type not in ("rayleigh", "rician"):
            raise ValueError("fading_type must be 'rayleigh' or 'rician'.")
        if rician_k < 0:
            raise ValueError("rician_k must be non-negative.")
        self.fading_type = fading_type
        self.rician_k = rician_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fading_type == "rayleigh":
            # h = (h_r + j*h_i) / sqrt(2),  |h| applied element-wise
            h_r = torch.randn_like(x)
            h_i = torch.randn_like(x)
            magnitude = torch.sqrt((h_r ** 2 + h_i ** 2) / 2.0)
        else:  # rician
            k = self.rician_k
            # Scattered component variance = 1/(2*(K+1)) per real dimension
            sigma = 1.0 / np.sqrt(2.0 * (k + 1.0))
            # LoS component (real part only for simplicity)
            los = np.sqrt(k / (k + 1.0))
            h_r = los + torch.randn_like(x) * sigma
            h_i = torch.randn_like(x) * sigma
            magnitude = torch.sqrt(h_r ** 2 + h_i ** 2)

        return x * magnitude

    def __repr__(self) -> str:
        return (f"FadingAttack(fading_type='{self.fading_type}', "
                f"rician_k={self.rician_k})")


# ---------------------------------------------------------------------------
# 7. Eavesdrop-Null Attack
# ---------------------------------------------------------------------------

class EavesdropNullAttack(nn.Module):
    """
    Eavesdrop & Null Attack (selective dropping).

    An eavesdropper intercepts the transmission and nulls out a random
    fraction of the latent dimensions before forwarding it, degrading the
    receiver's reconstruction quality.  Each forward pass uses a freshly
    sampled random mask, so the nulled dimensions vary over time.

    Parameters
    ----------
    null_ratio : float
        Fraction of latent entries to zero out, in [0, 1].
    """

    def __init__(self, null_ratio: float = 0.3):
        super().__init__()
        _check_ratio("null_ratio", null_ratio)
        self.null_ratio = null_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand_like(x) >= self.null_ratio).float()
        return x * mask

    def __repr__(self) -> str:
        return f"EavesdropNullAttack(null_ratio={self.null_ratio})"


# ---------------------------------------------------------------------------
# 8. Interleave Attack
# ---------------------------------------------------------------------------

class InterleaveAttack(nn.Module):
    """
    Interleave / Symbol-Shuffling Attack.

    Models inter-symbol interference or deliberate symbol reordering by an
    adversary.  The latent vector's last dimension (or all non-batch
    dimensions flattened) is randomly permuted.  A fixed seed makes the
    permutation reproducible across calls, simulating a static mapping
    injected by the attacker.

    Parameters
    ----------
    interleave_seed : int
        Seed for the permutation RNG.  The same seed gives the same
        permutation every call (deterministic attacker).  Set to -1 to
        draw a fresh random permutation each forward pass.
    """

    def __init__(self, interleave_seed: int = 0):
        super().__init__()
        self.interleave_seed = interleave_seed
        self._perm: torch.Tensor | None = None   # cached permutation

    def _get_perm(self, n: int, device: torch.device) -> torch.Tensor:
        if self.interleave_seed < 0:
            # Fresh random permutation each time
            return torch.randperm(n, device=device)
        # Deterministic: cache and reuse
        if self._perm is None or self._perm.shape[0] != n:
            rng = torch.Generator()
            rng.manual_seed(self.interleave_seed)
            self._perm = torch.randperm(n, generator=rng, device=device)
        return self._perm.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        # Flatten everything except batch dim, permute, then restore
        flat = x.reshape(orig_shape[0], -1)          # (B, N)
        n = flat.shape[1]
        perm = self._get_perm(n, x.device)
        shuffled = flat[:, perm]
        return shuffled.reshape(orig_shape)

    def __repr__(self) -> str:
        return f"InterleaveAttack(interleave_seed={self.interleave_seed})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ATTACK_REGISTRY = {
    "none":           NoAttack,
    "jamming":        JammingAttack,
    "bit_flooding":   BitFloodingAttack,
    "replay":         ReplayAttack,
    "pilot_spoofing": PilotSpoofingAttack,
    "fading":         FadingAttack,
    "eavesdrop_null": EavesdropNullAttack,
    "interleave":     InterleaveAttack,
}


def build_attack(args) -> nn.Module:
    """
    Instantiate the requested channel attack from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Must have an ``attack`` attribute that matches one of the keys in
        ATTACK_REGISTRY.  Additional attack-specific attributes are read
        as needed (e.g. args.attack_power, args.flood_ratio, ...).

    Returns
    -------
    nn.Module
        The instantiated (and CPU-resident) attack module.
        Move to the correct device with ``.to(device)`` or ``.cuda()``.
    """
    attack_name = args.attack.lower()
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(
            f"Unknown attack '{attack_name}'. "
            f"Choose from: {list(ATTACK_REGISTRY.keys())}"
        )

    if attack_name == "none":
        return NoAttack()

    if attack_name == "jamming":
        return JammingAttack(attack_power=args.attack_power)

    if attack_name == "bit_flooding":
        return BitFloodingAttack(
            flood_ratio=args.flood_ratio,
            flood_value=args.flood_value,
        )

    if attack_name == "replay":
        return ReplayAttack(replay_delay=args.replay_delay)

    if attack_name == "pilot_spoofing":
        return PilotSpoofingAttack(spoof_gain=args.spoof_gain)

    if attack_name == "fading":
        return FadingAttack(
            fading_type=args.fading_type,
            rician_k=args.rician_k,
        )

    if attack_name == "eavesdrop_null":
        return EavesdropNullAttack(null_ratio=args.null_ratio)

    if attack_name == "interleave":
        return InterleaveAttack(interleave_seed=args.interleave_seed)

    # Should never reach here due to registry check above
    raise RuntimeError(f"Unhandled attack: {attack_name}")
