#!/usr/bin/env python3
"""搭配表：用成品词典切分语料，按收缩后的 PMI 选词对，导出 LYCL。

与词典同词表、同语料来源，引擎查到的词对就是词典切出来的词对。运行期前文最多
3 个字，左词更长的词对永远查不到，不占容量。评测句不参与构建。

用法：python3 collocate.py output/main-400k.tsv --max-chars 200000000 \
        --out output
"""

from __future__ import annotations

import argparse
import bisect
import math
import multiprocessing as mp
import os
import pickle
import random
import struct

from scan import HANZI_RUN, MAX_DOC, lccc, run, send, ultra

MAGIC = b"LYCL"
VERSION = 2
HEADER_LEN = 16
RECORD_LEN = 5  # u32 指纹 + u8 分值
SCALE = 255 / 6  # 分值 0~6 线性映射到 0~255,读端按同一尺度还原
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


def fingerprint(key: str) -> int:
    """词对指纹:FNV-1a 64 位哈希取高 32 位,与 Rust 端 gram.rs 的 fingerprint 一致。"""
    return fnv1a(key.encode()) >> 32


def quantize(score: float) -> int:
    return max(0, min(255, round(score * SCALE)))


def export(scored: list[tuple[float, float, str]], limit: int, path: str) -> None:
    # 布局:16 字节头 + count 个升序 u32 指纹 + count 个 u8 分值。同指纹取最高分。
    best: dict[int, float] = {}
    for _, score, key in scored[:limit]:
        fp = fingerprint(key)
        best[fp] = max(best.get(fp, 0.0), score)
    items = sorted(best.items())
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<III", VERSION, len(items), 0))
        f.write(struct.pack(f"<{len(items)}I", *(fp for fp, _ in items)))
        f.write(bytes(quantize(score) for _, score in items))
    print(f"[export] {path} {len(items)} 条 {os.path.getsize(path)/2**20:.2f} MiB", flush=True)
    verify(path, scored[:limit], best)


def verify(
    path: str, pairs: list[tuple[float, float, str]], best: dict[int, float], samples: int = 1000
) -> None:
    """读回文件并随机抽词对:指纹须严格升序,二分查到的量化分值须与构建时一致。"""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != MAGIC or struct.unpack_from("<I", data, 4)[0] != VERSION:
        raise RuntimeError(f"{path}: 魔数或版本不符")
    count = struct.unpack_from("<I", data, 8)[0]
    if len(data) != HEADER_LEN + RECORD_LEN * count:
        raise RuntimeError(f"{path}: 长度 {len(data)} 与条数 {count} 不符")
    fps = struct.unpack_from(f"<{count}I", data, HEADER_LEN)
    qs = data[HEADER_LEN + 4 * count :]
    if any(a >= b for a, b in zip(fps, fps[1:])):
        raise RuntimeError(f"{path}: 指纹未严格升序")
    sample = random.Random(0).sample(pairs, min(samples, len(pairs)))
    for _, _, key in sample:
        fp = fingerprint(key)
        i = bisect.bisect_left(fps, fp)
        if i == count or fps[i] != fp:
            raise RuntimeError(f"读回缺失: {key!r}")
        if qs[i] != quantize(best[fp]):
            raise RuntimeError(f"分值不一致: {key!r} 读回 {qs[i]}")
    print(f"[verify] {path} 随机抽查 {len(sample)} 对通过", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dict")
    parser.add_argument("--max-chars", type=int, default=200_000_000)
    parser.add_argument("--limits", default="1000000")
    parser.add_argument("--clips", default="6", help="PMI 上限，逗号分隔")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

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
            send(queue, batch)
            batch = []
    send(queue, batch)
    for _ in procs:
        send(queue, None)
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


if __name__ == "__main__":
    run(main)
