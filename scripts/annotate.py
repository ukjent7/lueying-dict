#!/usr/bin/env python3
"""上下文注音：用 g2pW 给真实语料逐字定音，聚合成「词 → 读音分布」。

输出供 build.py 使用：词典里的读音来自词在句子里的实际读法，语料没见过的词才
退回词级词表与最长匹配。只统计含多音字的词组，单音字词的读音由字表决定。

用法（需要 GPU）：python3 annotate.py --max-chars 60000000 --out stats/readings.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
import unicodedata
from collections import Counter, defaultdict

import jieba
from g2pw import G2PWConverter

from legal_syllables import LEGAL_SYLLABLES
from scan import MAX_DOC, MAX_WORD, HANZI_RUN, lccc, ultra

RUNS = 8192
TONE = re.compile(r"\d")
# g2pW 训练数据带台湾读音，输入法按普通话规范收敛。
TAIWAN = {("和", "han"): "he", ("熟", "shou"): "shu"}


def syllable(ch: str, raw: str | None) -> str | None:
    if not raw:
        return None
    s = TONE.sub("", raw).lower().replace("ü", "v").replace("u:", "v")
    s = TAIWAN.get((ch, s), s)
    s = {"lue": "lve", "nue": "nve"}.get(s, s)
    return s if s in LEGAL_SYLLABLES else None


def mixed(limit_chars: int):
    """按 7:3 交替读书面与对话语料，避免读音分布只反映一种语域。"""
    streams = [ultra(), lccc()]
    pattern = (0, 0, 1, 0, 0, 1, 0, 0, 1, 0)
    seen = 0
    i = 0
    while seen < limit_chars and streams:
        text = next(streams[pattern[i % len(pattern)]], None)
        i += 1
        if text is None:
            break
        for run in HANZI_RUN.findall(text[:MAX_DOC]):
            if 4 <= len(run) <= 120:
                seen += len(run)
                yield run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=60_000_000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    conv = G2PWConverter(style="pinyin", enable_non_tradional_chinese=True, use_cuda=True,
                         batch_size=512, num_workers=0, turnoff_tqdm=True)
    poly = {row[0] for row in conv.polyphonic_chars}
    chars: dict[str, Counter] = defaultdict(Counter)
    grams: dict[str, Counter] = defaultdict(Counter)
    done = 0
    started = time.time()
    mark = started

    def flush(runs: list[str]) -> None:
        nonlocal done, mark
        for run, raw in zip(runs, conv(runs)):
            if len(raw) != len(run):
                continue
            syl = [syllable(ch, r) for ch, r in zip(run, raw)]
            for ch, s in zip(run, syl):
                if s and ch in poly:
                    chars[ch][s] += 1
            seq = jieba.lcut(run, HMM=True)
            at = [0]
            for w in seq:
                at.append(at[-1] + len(w))
            for i in range(len(seq)):
                for j in (i + 1, i + 2, i + 3):
                    if j >= len(at) or at[j] - at[i] > MAX_WORD:
                        break
                    piece = run[at[i]:at[j]]
                    keys = syl[at[i]:at[j]]
                    if len(piece) >= 2 and all(keys) and any(c in poly for c in piece):
                        grams[piece]["'".join(keys)] += 1
            done += len(run)
        if time.time() - mark > 600:
            mark = time.time()
            save()
            print(f"[g2p] chars={done/1e6:.1f}M grams={len(grams)} {time.time()-started:.0f}s", flush=True)

    def save() -> None:
        tmp = args.out + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"chars": dict(chars), "grams": dict(grams), "seen": done}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, args.out)

    batch: list[str] = []
    for run in mixed(args.max_chars):
        batch.append(run)
        if len(batch) >= RUNS:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    save()
    print(f"[done] chars={done} grams={len(grams)} {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
