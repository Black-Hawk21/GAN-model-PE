"""
secure_channel.py
=================
Call-based bridge that routes the GAN-SeCom latent through the Cifrar (Enc3)
security layer.

All cipher logic stays in ``cifrar.py`` (unmodified).  This module only calls
``Cifrar.encrypt`` / ``Cifrar.decrypt`` and adds the plumbing a tensor pipeline
needs around them: quantisation, a small packet header, BPSK mapping onto the
project's existing channel/attack modules, and the receiver's accept/reject
decision.

Per-image signal path (one packet per image)
--------------------------------------------
    TX  latent -> quantise -> [header | payload] -> Cifrar.encrypt -> BPSK (+/-1)
    CH  -> AWGN_Channel -> channel attack        (the project's own nn.Modules)
    RX  -> hard decision -> Cifrar.decrypt -> integrity / header / freshness checks
        -> dequantise -> latent   (an all-zero erasure if the packet is rejected)

Usage -- this is all main.py / evaluate.py do
---------------------------------------------
    from secure_channel import SecureLatentLink

    link = SecureLatentLink(seed=42)
    rx_latent, reports = link(tx_latent, channel, attack)   # tx_latent: (B, ...)
    print(link.stats)

Protocol assumptions
--------------------
* The latent shape (28 x 512 for CelebAMask-HQ-512) is fixed by the generator,
  so both ends know it and it is not transmitted.  The payload size, and hence
  Cifrar's ``num_bits_orig``, is derived from it.
* The shared secret is the Cifrar seed.
* The link is slot-synchronous: every call sends exactly one packet per image
  and the receiver consumes exactly one packet per image.

Freshness check (``replay_check``) -- added here, not part of Cifrar
--------------------------------------------------------------------
Cifrar is deterministic (no nonce), so a replayed ciphertext decrypts cleanly
with ``tampered=False``.  The header therefore carries a sequence number:

* ``"strict"``    -- seq must equal the receiver's own slot counter.  Catches
                     duplicated *and* delayed packets.  Right for this simulator.
* ``"monotonic"`` -- seq must exceed the last accepted seq.  Tolerates packet
                     loss on a real link, but only catches the first duplicate
                     of a delayed stream.
* ``"off"``       -- pure Cifrar behaviour; replays are accepted.

Cifrar's checksum is not keyed, so a sender-aware attacker can still rewrite the
sequence number.  The check stops naive replay, not a targeted forgery.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from cifrar import Cifrar, CipherPacket

__all__ = [
    "LatentQuantizer",
    "SecureTransmitter",
    "SecureReceiver",
    "SecureLatentLink",
    "PacketReport",
    "LinkStats",
    "REPLAY_MODES",
    "TAMPER_POLICIES",
]

REPLAY_MODES = ("strict", "monotonic", "off")
TAMPER_POLICIES = ("drop", "keep")

# magic(2s) version(B) quant_bits(B) seq(I) lo(f) hi(f)  -> 16 bytes
_HEADER = struct.Struct("<2sBBIff")
_MAGIC = b"GS"
_VERSION = 1
_SEQ_MASK = 0xFFFFFFFF
# Sanity bound on (hi - lo).  A power-normalised latent sits far below this;
# a header whose floats were corrupted usually does not.
_MAX_RANGE = 1e4


# ═════════════════════════════════════════════════════════════════════════════
# Quantisation
# ═════════════════════════════════════════════════════════════════════════════

class LatentQuantizer:
    """Uniform min/max quantiser: float tensor <-> little-endian unsigned ints.

    ``lo``/``hi`` are stored as float32 in the header, and both ends do the
    scale arithmetic in float64 on those exact values, so TX and RX use
    bit-identical quantisation parameters.
    """

    _DTYPES = {8: np.dtype("u1"), 16: np.dtype("<u2")}

    def __init__(self, bits: int = 8):
        if bits not in self._DTYPES:
            raise ValueError(f"quant_bits must be 8 or 16, got {bits}")
        self.bits = bits
        self.dtype = self._DTYPES[bits]
        self.levels = (1 << bits) - 1

    def payload_nbytes(self, numel: int) -> int:
        return numel * self.dtype.itemsize

    def quantise(self, x: torch.Tensor) -> Tuple[bytes, float, float]:
        arr = x.detach().to("cpu", torch.float32).numpy().astype(np.float64)
        lo = float(np.float32(arr.min()))
        hi = float(np.float32(arr.max()))
        scale = (hi - lo) or 1.0
        q = np.rint((arr - lo) / scale * self.levels)
        q = np.clip(q, 0, self.levels).astype(self.dtype)
        return q.tobytes(), lo, hi

    def dequantise(self, raw: bytes, lo: float, hi: float,
                   shape: Sequence[int], device) -> torch.Tensor:
        scale = (hi - lo) or 1.0
        q = np.frombuffer(raw, dtype=self.dtype).astype(np.float64)
        arr = (q / self.levels * scale + lo).astype(np.float32).reshape(tuple(shape))
        return torch.from_numpy(arr).to(device)


# ═════════════════════════════════════════════════════════════════════════════
# BPSK mapping  (symbol k carries ciphertext bit k in Cifrar's LSB-first order)
# ═════════════════════════════════════════════════════════════════════════════

def _bytes_to_symbols(data: bytes, device) -> torch.Tensor:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    sym = (1.0 - 2.0 * bits.astype(np.float32)).astype(np.float32)  # 0->+1, 1->-1
    return torch.from_numpy(sym).to(device).unsqueeze(0)              # (1, N)


def _symbols_to_bits(sym: torch.Tensor) -> np.ndarray:
    return (sym.detach().reshape(-1) < 0).to("cpu", torch.uint8).numpy()


# ═════════════════════════════════════════════════════════════════════════════
# Reports / statistics
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PacketReport:
    """Outcome of one packet.

    status: ``ok`` | ``tampered`` | ``replayed`` | ``bad_header`` | ``bad_length``
    """
    status: str
    accepted: bool                 # latent was passed on to the generator
    tampered: bool                 # Cifrar integrity check failed
    seq: Optional[int] = None      # sequence number read from the header
    bit_errors: int = -1           # channel bit errors (simulation-only, -1 = n/a)
    channel_uses: int = 0          # BPSK symbols spent on this packet

    def __str__(self) -> str:
        errs = "n/a" if self.bit_errors < 0 else f"{self.bit_errors}/{self.channel_uses}"
        return (f"status={self.status} accepted={self.accepted} "
                f"tampered={self.tampered} seq={self.seq} bit_errors={errs}")


class LinkStats:
    """Running counters over the packets sent through a link."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.packets = 0
        self.accepted = 0
        self.tampered = 0
        self.replayed = 0
        self.bit_errors = 0
        self.bits_compared = 0
        self.channel_uses = 0

    def update(self, rep: PacketReport) -> None:
        self.packets += 1
        self.accepted += int(rep.accepted)
        self.tampered += int(rep.tampered)
        self.replayed += int(rep.status == "replayed")
        self.channel_uses += rep.channel_uses
        if rep.bit_errors >= 0:
            self.bit_errors += rep.bit_errors
            self.bits_compared += rep.channel_uses

    def summary(self) -> Dict[str, float]:
        n = max(self.packets, 1)
        return {
            "packets":        self.packets,
            "accepted":       self.accepted,
            "reject_rate":    (self.packets - self.accepted) / n,
            "tamper_rate":    self.tampered / n,
            "replay_rate":    self.replayed / n,
            "link_ber":       self.bit_errors / max(self.bits_compared, 1),
            "channel_uses":   self.channel_uses / n,   # per packet (= per image)
        }

    def __str__(self) -> str:
        s = self.summary()
        return (f"packets={s['packets']} accepted={s['accepted']} "
                f"reject_rate={s['reject_rate']:.3f} tamper_rate={s['tamper_rate']:.3f} "
                f"replay_rate={s['replay_rate']:.3f} link_BER={s['link_ber']:.3e} "
                f"channel_uses/image={s['channel_uses']:.0f}")


# ═════════════════════════════════════════════════════════════════════════════
# Transmitter / receiver
# ═════════════════════════════════════════════════════════════════════════════

class SecureTransmitter:
    """Quantise -> add header -> Cifrar.encrypt -> BPSK."""

    def __init__(self, seed: int, quant_bits: int = 8):
        self._codec = Cifrar(seed=seed)
        self.quantizer = LatentQuantizer(quant_bits)
        self.seq = 0

    def send(self, sample: torch.Tensor) -> torch.Tensor:
        """Encrypt one image's latent; return BPSK symbols of shape (1, N)."""
        payload, lo, hi = self.quantizer.quantise(sample)
        header = _HEADER.pack(_MAGIC, _VERSION, self.quantizer.bits, self.seq, lo, hi)
        self.seq = (self.seq + 1) & _SEQ_MASK
        packet = self._codec.encrypt(header + payload)
        return _bytes_to_symbols(packet.ciphertext, sample.device)


class SecureReceiver:
    """Hard decision -> Cifrar.decrypt -> checks -> dequantise."""

    def __init__(self, seed: int, quant_bits: int = 8,
                 on_tamper: str = "drop", replay_check: str = "strict"):
        if on_tamper not in TAMPER_POLICIES:
            raise ValueError(f"on_tamper must be one of {TAMPER_POLICIES}")
        if replay_check not in REPLAY_MODES:
            raise ValueError(f"replay_check must be one of {REPLAY_MODES}")
        self._codec = Cifrar(seed=seed)
        self.quantizer = LatentQuantizer(quant_bits)
        self.on_tamper = on_tamper
        self.replay_check = replay_check
        self.slot = 0                      # packets received so far (strict mode)
        self.last_seq: Optional[int] = None  # last accepted seq (monotonic mode)
        self._len_cache: Dict[int, int] = {}

    def _expected_bits(self, nbytes: int) -> int:
        # Ciphertext length depends only on (seed, payload size).  Learn it once
        # through the public API instead of re-deriving Cifrar's internals.
        if nbytes not in self._len_cache:
            self._len_cache[nbytes] = 8 * len(self._codec.encrypt(bytes(nbytes)).ciphertext)
        return self._len_cache[nbytes]

    def _parse_header(self, plaintext: bytes):
        magic, version, qbits, seq, lo, hi = _HEADER.unpack_from(plaintext)
        ok = (magic == _MAGIC and version == _VERSION
              and qbits == self.quantizer.bits
              and math.isfinite(lo) and math.isfinite(hi)
              and 0.0 <= hi - lo <= _MAX_RANGE)
        return ok, seq, lo, hi

    def _is_fresh(self, seq: int, slot: int) -> bool:
        if self.replay_check == "strict":
            return seq == slot
        if self.replay_check == "monotonic":
            return self.last_seq is None or seq > self.last_seq
        return True

    def receive(self, symbols: torch.Tensor, shape: Sequence[int],
                device) -> Tuple[torch.Tensor, PacketReport]:
        """Decode one packet.  Always returns a tensor of ``shape``."""
        slot = self.slot
        self.slot = (self.slot + 1) & _SEQ_MASK

        numel = math.prod(shape)
        nbytes = _HEADER.size + self.quantizer.payload_nbytes(numel)
        erasure = torch.zeros(tuple(shape), dtype=torch.float32, device=device)

        bits = _symbols_to_bits(symbols)
        uses = int(bits.size)
        if uses != self._expected_bits(nbytes):
            return erasure, PacketReport("bad_length", False, False, channel_uses=uses)

        ciphertext = np.packbits(bits, bitorder="little").tobytes()
        plaintext, tampered = self._codec.decrypt(CipherPacket(ciphertext, nbytes * 8))
        # A corrupted X field in a Cifrar chunk shifts the recovered bit count.
        plaintext = plaintext[:nbytes].ljust(nbytes, b"\0")

        if tampered and self.on_tamper == "drop":
            return erasure, PacketReport("tampered", False, True, channel_uses=uses)

        header_ok, seq, lo, hi = self._parse_header(plaintext)
        if not header_ok:
            return erasure, PacketReport("bad_header", False, tampered, channel_uses=uses)

        if tampered:  # on_tamper == "keep": decode the (possibly corrupted) data
            latent = self.quantizer.dequantise(plaintext[_HEADER.size:], lo, hi, shape, device)
            return latent, PacketReport("tampered", True, True, seq, channel_uses=uses)

        if not self._is_fresh(seq, slot):
            return erasure, PacketReport("replayed", False, False, seq, channel_uses=uses)

        self.last_seq = seq
        latent = self.quantizer.dequantise(plaintext[_HEADER.size:], lo, hi, shape, device)
        return latent, PacketReport("ok", True, False, seq, channel_uses=uses)


# ═════════════════════════════════════════════════════════════════════════════
# End-to-end link  (the only object the pipeline talks to)
# ═════════════════════════════════════════════════════════════════════════════

class SecureLatentLink:
    """TX -> channel -> attack -> RX, one packet per image in the batch.

    Parameters
    ----------
    seed : int
        Shared Cifrar secret.
    quant_bits : int
        8 or 16.  16 halves quantisation error but doubles ciphertext size.
    on_tamper : str
        ``"drop"`` replaces packets that fail the integrity check with an
        all-zero latent.  ``"keep"`` decodes them anyway, which is useful for
        studying how corrupted-but-used latents look.
    replay_check : str
        ``"strict"``, ``"monotonic"`` or ``"off"`` (see module docstring).
    """

    def __init__(self, seed: int = 42, quant_bits: int = 8,
                 on_tamper: str = "drop", replay_check: str = "strict"):
        self.tx = SecureTransmitter(seed, quant_bits)
        self.rx = SecureReceiver(seed, quant_bits, on_tamper, replay_check)
        self.stats = LinkStats()

    def reset_stats(self) -> None:
        self.stats.reset()

    @torch.no_grad()
    def __call__(self, latent: torch.Tensor, channel: nn.Module,
                 attack: nn.Module) -> Tuple[torch.Tensor, List[PacketReport]]:
        """Send every sample in ``latent`` (B, ...) through the secure link.

        Returns the received latent (same shape/dtype/device) and one
        :class:`PacketReport` per sample.
        """
        received, reports = [], []
        for sample in latent:
            tx_sym = self.tx.send(sample)
            rx_sym = attack(channel(tx_sym))
            rec, rep = self.rx.receive(rx_sym, sample.shape, sample.device)
            rep.channel_uses = tx_sym.numel()
            if rx_sym.numel() == tx_sym.numel():
                rep.bit_errors = int(((tx_sym < 0) != (rx_sym.reshape(tx_sym.shape) < 0)).sum())
            self.stats.update(rep)
            received.append(rec)
            reports.append(rep)
        return torch.stack(received).to(latent.dtype), reports

    def __repr__(self) -> str:  # the seed is a secret, so it is not printed
        return (f"SecureLatentLink(cipher=Cifrar, modulation=BPSK, "
                f"quant_bits={self.tx.quantizer.bits}, on_tamper='{self.rx.on_tamper}', "
                f"replay_check='{self.rx.replay_check}')")
