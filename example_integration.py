"""
example_integration.py -- how to bolt Cifrar onto the DeepJSCC pipeline.

The whole point (per the design brief): the security logic lives in ``cifrar.py``
and the main pipeline only *calls* it.  Nothing here reaches into the cipher
internals -- swapping Cifrar out later means changing these few lines only.

Run:  python example_integration.py
(Only needs numpy for the tensor demo; the cipher itself is pure stdlib.)
"""

import numpy as np

from cifrar import Cifrar


# ---------------------------------------------------------------------------
# 1. Bytes in, bytes out -- the minimal integration.
# ---------------------------------------------------------------------------
def demo_bytes():
    codec = Cifrar(seed=42)                      # shared secret; set once
    payload = b"any serialised message or header"

    packet = codec.encrypt(payload)              # -> send packet over the channel
    recovered, tampered = codec.decrypt(packet)  # <- at the receiver

    assert recovered == payload and not tampered
    print(f"[bytes]  {len(payload)} B -> {len(packet.ciphertext)} B ciphertext, "
          f"recovered OK, tampered={tampered}")


# ---------------------------------------------------------------------------
# 2. Latent-tensor bridge -- the DeepJSCC use case.
#    Quantise the latent to uint8, hand the raw bytes to Cifrar, reverse on RX.
#    This is exactly the "digital bridge" step described in the paper.
# ---------------------------------------------------------------------------
def quantise(latent: np.ndarray):
    """Map a real latent tensor to uint8 bytes; return bytes + (min,max,shape)."""
    lo, hi = float(latent.min()), float(latent.max())
    scale = (hi - lo) or 1.0
    q = np.round((latent - lo) / scale * 255.0).astype(np.uint8)
    return q.tobytes(), (lo, hi, latent.shape)


def dequantise(raw: bytes, meta):
    lo, hi, shape = meta
    scale = (hi - lo) or 1.0
    q = np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(np.float32)
    return q / 255.0 * scale + lo


def demo_latent():
    codec = Cifrar(seed=42)

    # Pretend this came out of the encoder+TX-transformer: (B, C, H, W).
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((1, 16, 4, 4)).astype(np.float32)

    # --- transmitter: quantise -> encrypt ---
    raw, meta = quantise(latent)
    packet = codec.encrypt(raw)

    # --- receiver: decrypt -> dequantise ---
    raw_rx, tampered = codec.decrypt(packet)
    latent_rx = dequantise(raw_rx, meta)

    err = np.abs(latent - latent_rx).max()
    print(f"[latent] shape={latent.shape}, tampered={tampered}, "
          f"max quantisation error={err:.4f}")

    # --- tamper demo: corrupt a few ciphertext bytes -> MAC catches it ---
    # (simple_hash is a lightweight 32-bit MAC, so corrupting a short run is
    #  the honest way to show detection; a single-bit flip is caught with
    #  overwhelming, but not cryptographic, probability.)
    from cifrar import CipherPacket
    bad = bytearray(packet.ciphertext)
    for i in range(40, 48):
        bad[i] ^= 0xFF
    _, tampered_bad = codec.decrypt(CipherPacket(bytes(bad), packet.num_bits_orig))
    print(f"[tamper] after corrupting 8 bytes -> tampered={tampered_bad} (expected True)")


if __name__ == "__main__":
    demo_bytes()
    demo_latent()
