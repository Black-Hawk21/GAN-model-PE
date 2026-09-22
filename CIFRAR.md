# Cifrar (Enc3) — Python port

A pure-Python, dependency-free, **plug-and-play** re-implementation of the
`enc3_lib.cpp` deterministic bit-level obfuscation and tamper-detection layer.

- **Separate, call-based module** — the crypto lives here, the main pipeline
  only *calls* it. No build step (no `g++`, no `pybind11`, no `.so`).
- **Bit-exact** with the original C++: the ciphertext produced by `cifrar.py`
  is byte-identical to `enc3_lib.cpp` (verified across 5 test vectors incl. a
  256-byte binary payload), so it interoperates with the existing `.so`.

## Files
| File | Purpose |
|------|---------|
| `cifrar.py` | The cipher. Import this; that's the whole dependency. |
| `example_integration.py` | Shows the ~3 lines the pipeline needs (bytes + latent-tensor bridge). |

## Usage (the only two calls the pipeline makes)

```python
from cifrar import Cifrar

codec  = Cifrar(seed=42)               # shared secret, set once
packet = codec.encrypt(payload_bytes)  # transmitter -> send over channel
data, tampered = codec.decrypt(packet) # receiver -> (bytes, bool)
```

For the DeepJSCC latent (quantise → encrypt → … → decrypt → dequantise), see
`example_integration.py`.

### Low-level API (identical signatures to the C++ module)
```python
from cifrar import encrypt_data, decrypt_data
enc = encrypt_data(b"...", seed=42)                       # -> EncryptResult
dec = decrypt_data(enc.ciphertext, 42, enc.num_bits_orig) # -> DecryptResult
```

## How it works (matches the paper)
1. **XOR** each 26-bit chunk with key-selected round constants.
2. **Circular bit-shift**, **swap halves**, **bit reverse**.
3. **1-to-X micro-interleaving**: after every data bit inject `X ∈ {8,9,10}`
   pseudo-random noise bits (so <11% of the wire carries signal).
4. **32-bit MAC** seals each chunk; any modification flips the `tampered` flag.

All randomness is a C++-standard `mt19937` seeded from the shared secret
(master pad `seed ^ 0xDEADBEEF`), so both endpoints reproduce the exact same
obfuscation without transmitting a key.

> Note: `simple_hash` is a lightweight 32-bit MAC (as in the original), not a
> cryptographic hash. It catches corruption with very high — but not
> cryptographic — probability. Swap it for HMAC-SHA256 if you need a
> cryptographic integrity guarantee.

## Test
```bash
python cifrar.py              # round-trip + tamper self-test
python example_integration.py # bytes + latent-tensor demo (needs numpy)
```
