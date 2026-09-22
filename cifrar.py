"""
cifrar.py -- Pure-Python, dependency-free port of the Enc3 / Cifrar
deterministic bit-level obfuscation and tamper-detection layer.

This is a faithful, *bit-exact* re-implementation of ``enc3_lib.cpp`` (the
C++/pybind11 module used in the Semantic-Communication-Pipeline).  It uses only
the Python standard library, so it drops into any project with no build step
(no g++, no pybind11, no .so to compile).

Design goals (plug-and-play):
    * Single self-contained file -- copy it next to your pipeline and import it.
    * Call-based API -- the main pipeline never contains any crypto logic; it
      just calls ``encrypt_data`` / ``decrypt_data`` (or the ``Cifrar`` class).
    * Bit-exact with the C++ implementation, so ciphertext produced here can be
      decrypted by the original ``enc3`` module and vice-versa (both sides use
      the same C++-standard mt19937 PRNG and the same 26-bit chunk pipeline).

Quick start
-----------
    from cifrar import Cifrar

    codec = Cifrar(seed=42)                 # shared secret seed
    packet = codec.encrypt(b"payload")      # -> CipherPacket
    data, tampered = codec.decrypt(packet)  # -> (b"payload", False)

Low-level API (mirrors the C++ signatures exactly)
--------------------------------------------------
    from cifrar import encrypt_data, decrypt_data

    enc = encrypt_data(b"payload", seed=42)
    dec = decrypt_data(enc.ciphertext, seed=42, num_bits_orig=enc.num_bits_orig)
    assert dec.plaintext == b"payload" and not dec.tampered
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

__all__ = [
    "MT19937",
    "EncryptResult",
    "DecryptResult",
    "CipherPacket",
    "Cifrar",
    "encrypt_data",
    "decrypt_data",
    "generate_key",
]

_U32 = 0xFFFFFFFF


# =====================================================================
#  MT19937 -- matches C++ std::mt19937(uint32_t seed) exactly
# =====================================================================
class MT19937:
    """C++-standard 32-bit Mersenne Twister.

    Seeding and tempering match ``std::mt19937`` seeded with a single
    ``uint32_t`` value, which is what ``enc3_lib.cpp`` relies on.
    """

    __slots__ = ("mt", "index")

    def __init__(self, seed: int):
        self.mt: List[int] = [0] * 624
        self.index = 624
        self.mt[0] = seed & _U32
        for i in range(1, 624):
            prev = self.mt[i - 1]
            self.mt[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & _U32

    def _generate(self) -> None:
        mt = self.mt
        for i in range(624):
            y = (mt[i] & 0x80000000) + (mt[(i + 1) % 624] & 0x7FFFFFFF)
            mt[i] = mt[(i + 397) % 624] ^ (y >> 1)
            if y & 1:
                mt[i] ^= 0x9908B0DF
        self.index = 0

    def next_u32(self) -> int:
        """Return the next 32-bit unsigned integer (like ``rng()`` in C++)."""
        if self.index >= 624:
            self._generate()
        y = self.mt[self.index]
        self.index += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & _U32


# =====================================================================
#  bytes <-> bits  (LSB-first per byte, matching the C++ helpers)
# =====================================================================
def _bytes_to_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    bits_extend = bits.append
    for c in data:
        for i in range(8):
            bits_extend((c >> i) & 1)
    return bits


def _bits_to_bytes(bits: List[int]) -> bytes:
    out = bytearray()
    n = len(bits)
    for i in range(0, n, 8):
        c = 0
        for j in range(8):
            if i + j < n:
                c |= (bits[i + j] << j)
        out.append(c & 0xFF)
    return bytes(out)


# =====================================================================
#  Key
# =====================================================================
@dataclass
class _Key:
    constants: List[int]
    mask: int


def generate_key(seed: int) -> _Key:
    """Deterministically derive the round constants and selection mask."""
    rng = MT19937(seed)
    constants = [rng.next_u32() for _ in range(6)]
    mask = rng.next_u32()
    return _Key(constants=constants, mask=mask)


# =====================================================================
#  Bit operations (mirror enc3_lib.cpp)
# =====================================================================
def _xor_bits(a: List[int], b: List[int]) -> List[int]:
    lb = len(b)
    return [a[i] ^ b[i % lb] for i in range(len(a))]


def _rotate_left(a: List[int], k: int) -> List[int]:
    n = len(a)
    return [a[((i - k) % n + n) % n] for i in range(n)]


def _rotate_right(a: List[int], k: int) -> List[int]:
    return _rotate_left(a, len(a) - k)


def _reverse_bits(a: List[int]) -> List[int]:
    return a[::-1]


def _swap_halves(a: List[int]) -> List[int]:
    mid = len(a) // 2
    return a[mid:] + a[:mid]


# =====================================================================
#  Hash (MAC) -- 32-bit, matches simple_hash() with uint32 wraparound
# =====================================================================
def _simple_hash(bits: List[int]) -> int:
    h = 0x9E3779B9
    for b in bits:
        h ^= (b + ((h << 6) & _U32) + (h >> 2)) & _U32
        h &= _U32
    return h


# =====================================================================
#  Padding -- deterministic (seeded from per-chunk pad seed)
#  1-to-X micro-interleaving: after EVERY data bit, inject X noise bits,
#  X in {8, 9, 10}.  36-bit metadata (4-bit X + 32-bit MAC) is prepended.
# =====================================================================
def _apply_padding(data: List[int], pad_seed: int) -> List[int]:
    rng = MT19937(pad_seed)
    x = rng.next_u32() % 3 + 8  # 8, 9, or 10

    result: List[int] = []
    for bit in data:
        result.append(bit)
        for _ in range(x):
            result.append(rng.next_u32() & 1)

    mac = _simple_hash(result)

    meta: List[int] = []
    for i in range(4):            # 4 bits store X (8..10)
        meta.append((x >> i) & 1)
    for i in range(32):           # 32-bit MAC
        meta.append((mac >> i) & 1)
    meta.extend(result)
    return meta


def _remove_padding(data: List[int]) -> Tuple[List[int], bool]:
    if len(data) < 36:
        return [], True

    x = 0
    for i in range(4):
        x |= (data[i] << i)

    expected_mac = 0
    for i in range(32):
        expected_mac |= (data[4 + i] << i)
    expected_mac &= _U32

    stream = data[36:]
    actual_mac = _simple_hash(stream)
    tampered = (actual_mac != expected_mac)

    result: List[int] = []
    i = 0
    n = len(stream)
    while i < n:
        result.append(stream[i])
        i += 1
        i += x          # skip the X padding bits after each data bit
    return result, tampered


# =====================================================================
#  Chunk-level encrypt / decrypt (26-bit chunks)
# =====================================================================
def _constant_bits(value: int, n: int = 26) -> List[int]:
    return [(value >> j) & 1 for j in range(n)]


def _encrypt_chunk(s: List[int], key: _Key, pad_seed: int) -> List[int]:
    for i in range(1, 7):                       # mask bits 1..6
        if key.mask & (1 << i):
            s = _xor_bits(s, _constant_bits(key.constants[i - 1]))
    s = _rotate_left(s, 3)
    s = _swap_halves(s)
    s = _reverse_bits(s)
    return _apply_padding(s, pad_seed)


def _decrypt_chunk(s: List[int], key: _Key) -> Tuple[List[int], bool]:
    s, tampered = _remove_padding(s)
    s = _reverse_bits(s)
    s = _swap_halves(s)
    s = _rotate_right(s, 3)
    for i in range(6, 0, -1):                   # inverse order, mask bits 6..1
        if key.mask & (1 << i):
            s = _xor_bits(s, _constant_bits(key.constants[i - 1]))
    return s, tampered


# =====================================================================
#  Public results (mirror the C++ structs)
# =====================================================================
@dataclass
class EncryptResult:
    ciphertext: bytes
    num_bits_orig: int
    ok: bool = True


@dataclass
class DecryptResult:
    plaintext: bytes
    tampered: bool


# =====================================================================
#  PUBLIC low-level API (same names/semantics as the C++ module)
# =====================================================================
def encrypt_data(data: bytes, seed: int) -> EncryptResult:
    """Encrypt ``data`` (raw bytes) with the shared ``seed``.

    Returns an :class:`EncryptResult` whose ``ciphertext`` and
    ``num_bits_orig`` are exactly what the C++ ``encrypt_data`` produces.
    """
    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    key = generate_key(seed)
    bits = _bytes_to_bits(data)
    orig_len = len(bits)

    pad_rng = MT19937(seed ^ 0xDEADBEEF)

    encrypted_all: List[int] = []
    for i in range(0, len(bits), 26):
        chunk = bits[i:i + 26]
        if len(chunk) < 26:
            chunk = chunk + [0] * (26 - len(chunk))
        ps = pad_rng.next_u32()
        encrypted_all.extend(_encrypt_chunk(chunk, key, ps))

    return EncryptResult(
        ciphertext=_bits_to_bytes(encrypted_all),
        num_bits_orig=orig_len,
        ok=True,
    )


def decrypt_data(ciphertext: bytes, seed: int, num_bits_orig: int) -> DecryptResult:
    """Decrypt ``ciphertext`` produced by :func:`encrypt_data`.

    ``num_bits_orig`` is the value returned alongside the ciphertext; it is
    needed to re-derive the per-chunk padding boundaries and to trim the
    recovered stream back to its original length.  ``tampered`` is ``True`` if
    any chunk fails its MAC check.
    """
    if isinstance(ciphertext, (bytearray, memoryview)):
        ciphertext = bytes(ciphertext)
    key = generate_key(seed)
    all_bits = _bytes_to_bits(ciphertext)

    pad_rng = MT19937(seed ^ 0xDEADBEEF)
    num_chunks = (num_bits_orig + 25) // 26

    tampered_any = False
    final_bits: List[int] = []
    offset = 0
    total = len(all_bits)

    for _ in range(num_chunks):
        # Re-derive X for this chunk from the same deterministic sequence.
        chunk_rng = MT19937(pad_rng.next_u32())
        x = chunk_rng.next_u32() % 3 + 8

        padded_stream_len = 26 * (1 + x)
        chunk_enc_bits = 36 + padded_stream_len
        if offset + chunk_enc_bits > total:
            chunk_enc_bits = total - offset

        chunk = all_bits[offset:offset + chunk_enc_bits]
        offset += chunk_enc_bits

        dec, chunk_tampered = _decrypt_chunk(chunk, key)
        if chunk_tampered:
            tampered_any = True
        final_bits.extend(dec)

    if len(final_bits) > num_bits_orig:
        final_bits = final_bits[:num_bits_orig]

    return DecryptResult(
        plaintext=_bits_to_bytes(final_bits),
        tampered=tampered_any,
    )


# =====================================================================
#  Plug-and-play wrapper
# =====================================================================
@dataclass
class CipherPacket:
    """Self-describing encrypted packet: carries everything decrypt needs."""
    ciphertext: bytes
    num_bits_orig: int


class Cifrar:
    """Stateful, call-based wrapper around the Cifrar cipher.

    The main pipeline holds one instance and never touches the internals::

        codec  = Cifrar(seed=42)
        packet = codec.encrypt(payload_bytes)     # over the "wire"
        data, tampered = codec.decrypt(packet)

    For interop with the raw C++-style API, use :meth:`encrypt_raw` /
    :meth:`decrypt_raw`, which return/accept ``(ciphertext, num_bits_orig)``.
    """

    def __init__(self, seed: int = 42):
        self.seed = int(seed) & _U32

    # -- packet-based (recommended, self-describing) --------------------
    def encrypt(self, data: bytes) -> CipherPacket:
        r = encrypt_data(data, self.seed)
        return CipherPacket(ciphertext=r.ciphertext, num_bits_orig=r.num_bits_orig)

    def decrypt(self, packet: CipherPacket) -> Tuple[bytes, bool]:
        d = decrypt_data(packet.ciphertext, self.seed, packet.num_bits_orig)
        return d.plaintext, d.tampered

    # -- raw (matches the C++ call signatures) --------------------------
    def encrypt_raw(self, data: bytes) -> Tuple[bytes, int]:
        r = encrypt_data(data, self.seed)
        return r.ciphertext, r.num_bits_orig

    def decrypt_raw(self, ciphertext: bytes, num_bits_orig: int) -> Tuple[bytes, bool]:
        d = decrypt_data(ciphertext, self.seed, num_bits_orig)
        return d.plaintext, d.tampered


# =====================================================================
#  Self-test (mirrors the C++ STANDALONE_TEST)
# =====================================================================
def _self_test() -> None:
    msg = b"Hello, Deep JSCC!"
    seed = 12345

    enc = encrypt_data(msg, seed)
    print(f"Encrypted {len(msg)} bytes -> {len(enc.ciphertext)} bytes")

    dec = decrypt_data(enc.ciphertext, seed, enc.num_bits_orig)
    print(f"Recovered: {dec.plaintext!r}")
    print(f"Tampered:  {'YES' if dec.tampered else 'no'}")
    assert dec.plaintext[:len(msg)] == msg, "round-trip mismatch"
    assert not dec.tampered, "false-positive tamper flag"
    print("round-trip OK")

    corrupted = bytearray(enc.ciphertext)
    corrupted[5] ^= 0x01
    dec2 = decrypt_data(bytes(corrupted), seed, enc.num_bits_orig)
    print(f"After corruption -- tampered: {'YES (detected)' if dec2.tampered else 'no (MISS)'}")
    assert dec2.tampered, "tamper not detected"

    # Class wrapper round-trip on binary payload.
    codec = Cifrar(seed=42)
    payload = bytes(range(256))
    data, tampered = codec.decrypt(codec.encrypt(payload))
    assert data == payload and not tampered, "class round-trip mismatch"
    print("Cifrar class round-trip OK (256-byte binary payload)")


if __name__ == "__main__":
    _self_test()
