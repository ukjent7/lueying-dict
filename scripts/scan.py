#!/usr/bin/env python3
"""语料扫描：流式读取一个来源，统计 1/2/3 元词组的词次与文档频次。

每个来源独立产出一份统计，由 build.py 按语域权重合并。值把词次和文档频次
压进一个整数 (词次 << 32 | 文档数)，同一批键只存一份，内存减半；超过上限时
丢弃低频键（Lossy Counting），只影响长尾。

用法：python3 scan.py --source ultra --max-chars 500000000 --out stats/ultra.pkl
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import multiprocessing as mp
import os
import pickle
import re
import time
import urllib.request

ONE = 1 << 32
LIMIT = 7_000_000
MAX_DOC = 4000
BATCH = 64
MAX_WORD = 6
HANZI_RUN = re.compile(r"[一-鿿]+")
# 只有出现这些高频繁体字才走较慢的繁简转换。
TRAD = re.compile("[這們個為來時國會對於說過還學發經應樣開從點當間現長見關與頭將實體問話給後裡讓]")


def lossy(table: dict, floor: int) -> tuple[dict, int]:
    if len(table) <= LIMIT:
        return table, floor
    while True:
        floor += 1
        kept = {k: v for k, v in table.items() if (v >> 32) >= floor}
        if len(kept) <= LIMIT // 2:
            return kept, floor


def worker(queue, path: str) -> None:
    import jieba
    from zhconv import convert

    jieba.initialize()
    tables = [{}, {}, {}]
    floors = [1, 1, 1]
    docs = chars = tokens = 0
    mark = 0
    while (batch := queue.get()) is not None:
        for text in batch:
            if TRAD.search(text):
                text = convert(text, "zh-hans")
            seen: list[set] = [set(), set(), set()]
            for run in HANZI_RUN.findall(text):
                seq = jieba.lcut(run, HMM=True)
                tokens += len(seq)
                chars += len(run)
                for i, word in enumerate(seq):
                    grams = (word, word + seq[i + 1] if i + 1 < len(seq) else "",
                             word + seq[i + 1] + seq[i + 2] if i + 2 < len(seq) else "")
                    for n, gram in enumerate(grams):
                        # 单个 token 是分词器认定的整词，长词组才受长度上限约束。
                        if not gram or (n and len(gram) > MAX_WORD):
                            continue
                        table = tables[n]
                        if gram in seen[n]:
                            table[gram] += ONE
                        else:
                            seen[n].add(gram)
                            table[gram] = table.get(gram, 0) + ONE + 1
            docs += 1
        for n in (1, 2):
            tables[n], floors[n] = lossy(tables[n], floors[n])
        if chars - mark >= 50_000_000:
            mark = chars
            dump(path, tables, docs, chars, tokens)
    dump(path, tables, docs, chars, tokens)


def dump(path: str, tables: list, docs: int, chars: int, tokens: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"grams": tables, "docs": docs, "chars": chars, "tokens": tokens}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def ultra():
    from datasets import load_dataset

    for row in load_dataset("openbmb/Ultra-FineWeb", split="zh", streaming=True):
        yield row.get("content") or row.get("text") or ""


def fineweb2(since: str = "2021"):
    from datasets import load_dataset

    for row in load_dataset("HuggingFaceFW/fineweb-2", name="cmn_Hani", split="train", streaming=True).shuffle(seed=7, buffer_size=2000):
        if row["date"] >= since:
            yield row["text"]


def lccc():
    url = "https://huggingface.co/datasets/silver/lccc/resolve/main/lccc_base_train.jsonl.gz"
    req = urllib.request.Request(url, headers={"User-Agent": "lueying-dict"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in io.TextIOWrapper(gzip.GzipFile(fileobj=resp), encoding="utf-8"):
            row = json.loads(raw)
            dialog = row if isinstance(row, list) else row["dialog"]
            yield "，".join(s.replace(" ", "") for s in dialog)


SOURCES = {"ultra": ultra, "fineweb2": fineweb2, "lccc": lccc}


def merge(paths: list[str], out: str) -> None:
    total = None
    for path in paths:
        with open(path, "rb") as f:
            part = pickle.load(f)
        if total is None:
            total = part
            continue
        for mine, theirs in zip(total["grams"], part["grams"]):
            for key, value in theirs.items():
                mine[key] = mine.get(key, 0) + value
        for name in ("docs", "chars", "tokens"):
            total[name] += part[name]
    with open(out, "wb") as f:
        pickle.dump(total, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[done] {out} docs={total['docs']} chars={total['chars']} tokens={total['tokens']} "
          f"grams={[len(t) for t in total['grams']]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--max-chars", type=int, default=500_000_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    queue = mp.Queue(maxsize=args.workers * 8)
    parts = [f"{args.out}.part{i}" for i in range(args.workers)]
    procs = [mp.Process(target=worker, args=(queue, p)) for p in parts]
    for proc in procs:
        proc.start()

    sent = docs = 0
    batch: list[str] = []
    started = time.time()
    try:
        for text in SOURCES[args.source]():
            text = text[:MAX_DOC]
            cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
            if cjk < (4 if args.source == "lccc" else 20):
                continue
            batch.append(text)
            sent += cjk
            docs += 1
            if len(batch) >= BATCH:
                queue.put(batch)
                batch = []
            if docs % 20000 == 0:
                print(f"[scan] {args.source} docs={docs} chars={sent/1e6:.0f}M {time.time()-started:.0f}s", flush=True)
            if sent >= args.max_chars:
                break
    except Exception as exc:  # noqa: BLE001 - 流中断时保留已统计的部分
        print(f"[warn] stream stopped: {exc}", flush=True)
    if batch:
        queue.put(batch)
    for _ in procs:
        queue.put(None)
    for proc in procs:
        proc.join()
    merge(parts, args.out)
    for path in parts:
        os.remove(path)


if __name__ == "__main__":
    main()
