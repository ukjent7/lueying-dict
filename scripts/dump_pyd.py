#!/usr/bin/env python3
"""读 main.pyd：词 -> [(key, freq)]，另可模拟 unigram 路径分比较整词 vs 单字拼合。"""
import math
import struct
import sys
from collections import defaultdict

MAGIC = b"PYD2\x00\x00\x00\x01"
WORD_PENALTY = 2.2  # 与 build.py / 引擎同
SCALE = 1_000_000_000


def load(path):
    with open(path, "rb") as f:
        blob = f.read()
    assert blob[:8] == MAGIC, blob[:8]
    n, dlen = struct.unpack_from("<II", blob, 8)
    recs = blob[16 : 16 + n * 14]
    data = blob[16 + n * 14 : 16 + n * 14 + dlen]
    entries = []  # (key, word, freq)
    for i in range(n):
        wo, ko, freq, wl, kl = struct.unpack_from("<IIIBB", recs, i * 14)
        word = data[wo : wo + wl].decode()
        key = data[ko : ko + kl].decode()
        entries.append((key, word, freq))
    return entries


def main():
    path = sys.argv[1]
    entries = load(path)
    by_key = defaultdict(list)
    by_word = defaultdict(list)
    for key, word, freq in entries:
        by_key[key].append((word, freq))
        by_word[word].append((key, freq))

    print(f"entries={len(entries)} words={len(by_word)}")
    if len(sys.argv) > 2:
        for w in sys.argv[2:]:
            for key, freq in by_word.get(w, []):
                print(f"  word {w}\t{key}\tfreq={freq}\tp={freq/SCALE:.3e}")
            if w not in by_word:
                print(f"  word {w}\tNOT IN DICT")
        return

    # 交互模式省略；默认 dump 指定键
    for key in ("sha'bi", "sha", "bi"):
        top = sorted(by_key.get(key, []), key=lambda x: -x[1])[:10]
        print(f"key {key}: {top}")


if __name__ == "__main__":
    main()
