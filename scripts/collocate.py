#!/usr/bin/env python3
"""搭配表：用成品词典切分语料，按收缩后的 PMI 选词对，导出 LYCL。

与词典同词表、同语料来源，引擎查到的词对就是词典切出来的词对。运行期前文最多
3 个字，左词更长的词对永远查不到，不占容量。评测句不参与构建。

用法：python3 collocate.py output/main-400k.tsv --max-chars 200000000 \
        --out output
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import pickle
import struct
import sys

from scan import HANZI_RUN, MAX_DOC, lccc, ultra

MAGIC = b"LYCL"
VERSION = 1
MAX_LEFT = 3
LIMIT = 12_000_000
BATCH = 256
MIN_COUNT = 3
END = "$"


def fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def segment(run: str, vocab: set[str], longest: int) -> list[str]:
    words = []
    i = 0
    while i < len(run):
        for j in range(min(len(run), i + longest), i, -1):
            if j == i + 1 or run[i:j] in vocab:
                words.append(run[i:j])
                i = j
                break
    return words


def worker(queue, path: str, vocab: set[str], longest: int) -> None:
    uni: dict[str, int] = {}
    pairs: dict[str, int] = {}
    floor = 1
    tokens = total = 0
    while (batch := queue.get()) is not None:
        for run in batch:
            words = segment(run, vocab, longest)
            tokens += len(words)
            for word in words:
                uni[word] = uni.get(word, 0) + 1
            for left, right in zip(words, words[1:] + [END]):
                total += 1
                if len(left) <= MAX_LEFT or right == END:
                    key = f"{left}\0{right}"
                    pairs[key] = pairs.get(key, 0) + 1
        if len(pairs) > LIMIT:
            floor += 1
            pairs = {k: v for k, v in pairs.items() if v >= floor}
    with open(path, "wb") as f:
        pickle.dump((uni, pairs, tokens, total), f, protocol=pickle.HIGHEST_PROTOCOL)


def export(scored: list[tuple[float, float, str]], limit: int, path: str) -> None:
    table: dict[int, int] = {}
    for _, score, key in scored[:limit]:
        h = fnv1a(key.encode())
        value = max(-32768, min(32767, round(score * 1000)))
        table[h] = max(table.get(h, value), value)
    items = sorted(table.items())
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<III", VERSION, len(items), 0))
        f.write(struct.pack(f"<{len(items)}Q", *(h for h, _ in items)))
        f.write(struct.pack(f"<{len(items)}h", *(v for _, v in items)))
    print(f"[export] {path} {len(items)} 条 {os.path.getsize(path)/2**20:.2f} MiB", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dict")
    parser.add_argument("--max-chars", type=int, default=200_000_000)
    parser.add_argument("--limits", default="1000000")
    parser.add_argument("--clips", default="6", help="PMI 上限，逗号分隔")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    with open(args.dict, encoding="utf-8") as f:
        vocab = {line.split("\t", 1)[0] for line in f}
    longest = min(8, max(map(len, vocab)))

    queue = mp.Queue(maxsize=args.workers * 8)
    parts = [f"{args.out}/colloc.part{i}" for i in range(args.workers)]
    procs = [mp.Process(target=worker, args=(queue, p, vocab, longest)) for p in parts]
    for proc in procs:
        proc.start()

    streams = [ultra(), lccc()]
    seen = i = 0
    batch: list[str] = []
    while seen < args.max_chars and streams:
        stream = streams[i % len(streams)]
        i += 1
        text = next(stream, None)
        if text is None:
            streams.remove(stream)
            continue
        for run in HANZI_RUN.findall(text[:MAX_DOC]):
            if len(run) >= 2:
                batch.append(run)
                seen += len(run)
        if len(batch) >= BATCH:
            queue.put(batch)
            batch = []
    queue.put(batch)
    for _ in procs:
        queue.put(None)
    for proc in procs:
        proc.join()

    uni: dict[str, int] = {}
    pairs: dict[str, int] = {}
    tokens = total = 0
    for path in parts:
        with open(path, "rb") as f:
            u, p, t, m = pickle.load(f)
        for k, v in u.items():
            uni[k] = uni.get(k, 0) + v
        for k, v in p.items():
            pairs[k] = pairs.get(k, 0) + v
        tokens += t
        total += m
        os.remove(path)
    print(f"[count] tokens={tokens} pairs={len(pairs)}", flush=True)

    for clip in (float(x) for x in args.clips.split(",")):
        scored = []
        for key, count in pairs.items():
            if count < MIN_COUNT:
                continue
            left, right = key.split("\0")
            c_right = total / 10 if right == END else uni.get(right, count)
            pmi = math.log2((count / total) / max(uni.get(left, count) / tokens * c_right / tokens, 1e-12))
            score = min(clip, pmi * count / (count + 5.0))
            if score > 0:
                scored.append((count * max(0.1, score), score, key))
        scored.sort(reverse=True)
        print(f"[score] clip={clip} kept={len(scored)} at_cap={sum(1 for s in scored if s[1] >= clip)}", flush=True)
        for limit in (int(x) for x in args.limits.split(",")):
            export(scored, limit, f"{args.out}/collocation-{limit // 1000}k-c{clip:g}.bin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
