#!/usr/bin/env python3
"""词频改判据的原型验证：不改词表与读音，只对 main.pyd 词条重算词频。

逐来源 (count, df) 输入来自 spike 调研的分解产物（每行 `词\tcount\tdf`，首行 `# tokens=N`），
等价于 build.py 里 load_stats 合并后逐词查表。变体只改 FineWeb-2 那一路的取值规则，
其余来源照旧线性插值。输出同格式 tsv（word freq key 排序）与新 pyd，可直接喂 evaluate.py
和引擎 bench。

    --scheme raw        完全不封顶（对照:垃圾模板回潮的代价）
    --scheme cap5       现行 SPIKE=5, LONE=0.5e-6（复算自检，应还原 freq）
    --scheme soft:A     超出封顶的部分按 A 线性折扣保留（如 soft:0.1）
    --scheme floor:F    现行封顶 + 口语词频下限表 F（tsv: 词\\t每十亿次计数）

用法: rescale.py --pyd main-600k.pyd --sources srcdir/ --weights W.json \
        --scheme soft:0.1 --floor colloquial-freq.tsv --out out/
"""
import argparse
import json
import os
import struct
import sys
from collections import defaultdict

MAGIC = b"PYD2\x00\x00\x00\x01"
SCALE = 1_000_000_000
SPIKE = 5.0
LONE = 0.5e-6
SPIKY = "fineweb2"          # 按文件名前缀判 spiky，与 build.py is_spiky 一致
WEB_SOURCES = ("cc1", "cc2")  # basename 以 cc 开头的按网页来源，仅影响 supported，不在此用


def load_pyd(path):
    blob = open(path, "rb").read()
    assert blob[:8] == MAGIC
    n, dlen = struct.unpack_from("<II", blob, 8)
    entries = []
    for i in range(n):
        wo, ko, fr, wl, kl = struct.unpack_from("<IIIBB", blob, 16 + i * 14)
        w = blob[16 + n * 14 + wo:16 + n * 14 + wo + wl].decode()
        k = blob[16 + n * 14 + ko:16 + n * 14 + ko + kl].decode()
        entries.append((w, k, fr))
    return entries


def load_sources(sdir):
    """srcdir 下每个 *.tsv 是一路来源: 词 -> (count, df)，首行带 tokens。"""
    sources = {}
    for name in sorted(os.listdir(sdir)):
        if not name.endswith(".tsv"):
            continue
        src = name[:-4]
        table = {}
        with open(os.path.join(sdir, name), encoding="utf-8") as f:
            tokens = int(next(f).split("=")[1])
            for line in f:
                w, c, d = line.rstrip("\n").split("\t")
                table[w] = (int(c), int(d))
        sources[src] = {"tokens": tokens, "table": table}
    return sources


def pack_pyd(entries, path):
    data = bytearray()
    recs = bytearray()
    last_key, key_off = None, 0
    for word, key, freq in entries:
        wb, kb = word.encode(), key.encode()
        word_off = len(data)
        data += wb
        if key != last_key:
            key_off = len(data)
            data += kb
            last_key = key
        recs += struct.pack("<IIIBB", word_off, key_off, min(freq, 0xFFFFFFFF), len(wb), len(kb))
    order = sorted(range(len(entries)),
                   key=lambda i: ("".join(s[0] for s in entries[i][1].split("'")), i))
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<II", len(entries), len(data)))
        f.write(recs)
        f.write(data)
        f.write(struct.pack(f"<{len(order)}I", *order))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyd", required=True)
    ap.add_argument("--sources", required=True, help="每来源一个词频 tsv 的目录")
    ap.add_argument("--weights", required=True, help="来源名→混合权重 json（build_report.mix）")
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--floor", default="", help="口语词频下限 tsv（可叠加在任何 scheme 上）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = {k.replace(".pkl", ""): v for k, v in json.load(open(args.weights)).items()}
    sources = load_sources(args.sources)
    entries = load_pyd(args.pyd)
    print(f"[load] entries={len(entries)} sources={ {k: len(v['table']) for k, v in sources.items()} }",
          flush=True)

    # 每个词的封顶前/后 share 只算一次。
    kind, _, arg = args.scheme.partition(":")
    alpha = float(arg) if arg else 0.0
    floor_words = {}
    if args.floor:
        for line in open(args.floor, encoding="utf-8"):
            if line.strip() and not line.startswith("#"):
                w, f = line.rstrip("\n").split("\t")[:2]
                floor_words[w] = int(f) / SCALE

    cache = {}
    def probs(word):
        if word in cache:
            return cache[word]
        ref = 0.0
        base = 0.0
        fw = 0.0
        for src, info in sources.items():
            count = info["table"].get(word, (0, 0))[0]
            if not count:
                continue
            share = count / info["tokens"]
            if src == SPIKY:
                fw = share
            else:
                ref = max(ref, share)
                base += share * weights.get(src, 0.0)
        bound = max(ref * SPIKE, LONE)
        capped = min(fw, bound)
        if kind == "raw":
            used = fw
        elif kind == "cap5":
            used = capped
        elif kind == "soft":
            used = capped + alpha * max(fw - bound, 0.0)
        elif kind == "pow":
            used = capped if fw <= bound else bound * (fw / bound) ** alpha
        else:
            sys.exit(f"未知 scheme {kind}")
        cache[word] = (base + capped * weights.get(SPIKY, 0.0),
                       base + used * weights.get(SPIKY, 0.0))
        return cache[word]

    # 词的读音占比沿用原词典：freq_i_new = freq_i * p_eff / p_act，
    # 对多读音词等价于 p_eff·share_i，不用显式求 share。
    out_entries = []
    for w, k, fr in entries:
        p_act, p_new = probs(w)
        p_eff = max(p_new, floor_words.get(w, 0.0))
        new_fr = fr if p_act <= 0 else max(1, round(fr * p_eff / p_act))
        out_entries.append((w, k, new_fr))
    out_entries.sort(key=lambda e: (e[1], -e[2], e[0]))

    os.makedirs(args.out, exist_ok=True)
    tsv = os.path.join(args.out, f"main-{kind}{arg and '-' + arg}.tsv")
    pyd = tsv[:-4] + ".pyd"
    with open(tsv, "w", encoding="utf-8") as f:
        for w, k, fr in out_entries:
            f.write(f"{w}\t{fr}\t{k}\n")
    pack_pyd(out_entries, pyd)
    old = {(w, k): fr for w, k, fr in entries}
    changed = sum(1 for w, k, fr in out_entries if old.get((w, k)) != fr)
    print(f"[done] {tsv} changed={changed} entries={len(out_entries)}", flush=True)


if __name__ == "__main__":
    main()
