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
import random
import re
import time
import urllib.request

ONE = 1 << 32
LIMIT = 4_500_000
MAX_DOC = 4000
BATCH = 64
MAX_WORD = 6
HANZI_RUN = re.compile(r"[一-鿿]+")
# 只有出现这些高频繁体字才走较慢的繁简转换。
HANZI_CHAR = re.compile(r"[\u4e00-\u9fff]")
SENTENCE_END = re.compile(r"[，。！？；]")
WARC_HOST = re.compile(rb"WARC-Target-URI: https?://(?:www\.)?([^/\s:]+)")
# 采样站点里占比最大的两类垃圾：博彩和色情，词频会被它们的固定话术污染。
SPAM = re.compile("博彩|彩票|百家乐|时时彩|澳门|娱乐城|棋牌|赌场|开户|av|AV|成人|裸聊|一夜情|人妻|无码|三级片|代孕|发票|私服|外围")
SITE_CAP = 30
SEEN_LINES: set[int] = set()
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


def fetch(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "lueying-dict"})
    return urllib.request.urlopen(req, timeout=120)


def clean_page(body: str) -> str:
    """行级清洗：留下成句的中文，导航、标题列表和重复模板行都不是语言样本。"""
    kept = []
    for line in body.split("\n"):
        han = len(HANZI_CHAR.findall(line))
        if han < 12 or han / len(line) < 0.5 or not SENTENCE_END.search(line):
            continue
        digest = hash(line)
        if digest in SEEN_LINES:
            continue
        SEEN_LINES.add(digest)
        kept.append(line)
    if len(SEEN_LINES) > 8_000_000:
        SEEN_LINES.clear()
    return "\n".join(kept)


def commoncrawl(seed: int = 0, per_crawl: int = 400):
    """2025–2026 快照的 WET 文本。其余语料都停在 2023 年前后，这里是新词的来源。

    每站限额让词频反映「多少站在用」，单站刷词进不了词典；只取统计量，不留原文。
    """
    crawls = [c["id"] for c in json.load(fetch("https://index.commoncrawl.org/collinfo.json"))
              if c["id"] >= "CC-MAIN-2025"]
    rng = random.Random(seed)
    files = []
    for crawl in crawls:
        with fetch(f"https://data.commoncrawl.org/crawl-data/{crawl}/wet.paths.gz") as resp:
            paths = gzip.decompress(resp.read()).decode().split()
        files += rng.sample(paths, per_crawl)
    rng.shuffle(files)
    print(f"[cc] crawls={len(crawls)} files={len(files)}", flush=True)
    sites: dict[str, int] = {}
    for path in files:
        try:
            with fetch("https://data.commoncrawl.org/" + path) as resp:
                records = gzip.GzipFile(fileobj=resp).read().split(b"WARC/1.0\r\n")
        except Exception as exc:  # noqa: BLE001 - 单个分片失败不影响抽样
            print(f"[warn] {path}: {exc}", flush=True)
            continue
        for record in records:
            head, _, body = record.partition(b"\r\n\r\n")
            match = WARC_HOST.search(head)
            if not match or len(body) < 600:
                continue
            text = body.decode("utf-8", "ignore")
            if len(HANZI_CHAR.findall(text[:2000])) < 200:
                continue
            host = match.group(1).decode("ascii", "ignore")
            if sites.get(host, 0) >= SITE_CAP or SPAM.search(text):
                continue
            page = clean_page(text)
            if len(page) >= 200:
                sites[host] = sites.get(host, 0) + 1
                yield page


SOURCES = {"ultra": ultra, "fineweb2": fineweb2, "lccc": lccc, "commoncrawl": commoncrawl}


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
    parser.add_argument("--seed", type=int, default=0, help="commoncrawl 抽样种子")
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
        stream = commoncrawl(args.seed) if args.source == "commoncrawl" else SOURCES[args.source]()
        for text in stream:
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
            if docs % 5000 == 0:
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
