#!/usr/bin/env python3
"""词典门禁：只用与构建无关的金标准，同时检查词形和读音。

词形覆盖看不出注音错误（女生 → nu'sheng 仍算「覆盖」），所以门禁以「词形和拼音
都对」为准。任何一项不达标退出码非零。

用法：python3 evaluate.py output/main-400k.tsv --eval-dir ../../dicts \
        --context ../chinese-ime-lm/eval/sentences-ctx-v1/sampled.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import zlib
from collections import Counter, defaultdict

HANZI = re.compile(r"^[一-鿿]+$")
# 高频词读音探针：任何一条错了都说明注音阶段出了系统性问题。
PROBES = {
    "女生": "nv'sheng", "考虑": "kao'lv", "效率": "xiao'lv", "旅游": "lv'you", "法律": "fa'lv",
    "绿色": "lv'se", "都是": "dou'shi", "首都": "shou'du", "给你": "gei'ni", "供给": "gong'ji",
    "地铁": "di'tie", "地方": "di'fang", "长大": "zhang'da", "长度": "chang'du", "音乐": "yin'yue",
    "快乐": "kuai'le", "了解": "liao'jie", "银行": "yin'hang", "行走": "xing'zou", "重要": "zhong'yao",
    "重新": "chong'xin", "还是": "hai'shi", "还钱": "huan'qian", "觉得": "jue'de", "睡觉": "shui'jiao",
    "朝阳": "zhao'yang", "朝代": "chao'dai", "着急": "zhao'ji", "睡着": "shui'zhao", "调查": "diao'cha",
    "厦门": "xia'men", "大厦": "da'sha", "弹琴": "tan'qin", "子弹": "zi'dan", "这个": "zhe'ge",
}
# 词形金标偏旧：600k 同键召回 87%，端到端已超过旧词典，门限只防退步。
GATES = {"probes": 1.0, "recall": 0.85, "reach": 0.985, "reading": 0.985}


def norm(key: str) -> str:
    return key.replace("lve", "lue").replace("nve", "nue").replace("'", "")


def load(path: str) -> dict[str, dict[str, int]]:
    words: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, freq, key = line.rstrip("\n").split("\t")
            words[word][key] = int(freq)
    return words


def rows(path: str) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        return [x.rstrip("\n").split("\t") for x in f if x.strip() and not x.startswith("#")]


def word_metrics(words: dict, pairs: list[list[str]]) -> dict:
    index: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for word, keys in words.items():
        for key, freq in keys.items():
            index[norm(key)].append((freq, word))
    for items in index.values():
        items.sort(reverse=True)
    hit = top1 = top5 = form = 0
    for pinyin, gold in pairs:
        form += gold in words
        ranked = [w for _, w in index.get(norm(pinyin), [])]
        if gold in ranked:
            rank = ranked.index(gold)
            hit += 1
            top1 += rank == 0
            top5 += rank < 5
    n = len(pairs)
    return {"form": form / n, "recall": hit / n, "top1": top1 / n, "top5": top5 / n}


def reachable(words: dict, text: str, pinyin: str) -> int | None:
    """用词形和拼音都对的词拼出整句，返回最少段数；拼不出返回 None。"""
    target = norm(pinyin.lower())
    keys = {}
    best: list[dict[int, int]] = [{} for _ in range(len(text) + 1)]
    best[0][0] = 0
    for i in range(len(text)):
        for pos, segs in best[i].items():
            for j in range(i + 1, min(len(text), i + 10) + 1):
                piece = text[i:j]
                if piece not in words:
                    continue
                if piece not in keys:
                    keys[piece] = {norm(k) for k in words[piece]}
                for key in keys[piece]:
                    if target.startswith(key, pos):
                        end = pos + len(key)
                        if best[j].get(end, 1 << 30) > segs + 1:
                            best[j][end] = segs + 1
    return best[len(text)].get(len(target))


def reading_accuracy(words: dict, cases: list[dict]) -> tuple[float, Counter]:
    """最长匹配切出的多字词里，词典读音与金标准一致的比例。"""
    good = total = 0
    wrong: Counter[str] = Counter()
    for case in cases:
        text, syl = case["gold"], case["readings"]
        i = 0
        while i < len(text):
            for j in range(min(len(text), i + 8), i + 1, -1):
                if text[i:j] in words:
                    total += 1
                    want = norm("".join(syl[i:j]))
                    if any(norm(k) == want for k in words[text[i:j]]):
                        good += 1
                    else:
                        wrong[text[i:j]] += 1
                    i = j
                    break
            else:
                i += 1
    return good / max(total, 1), wrong


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dicts", nargs="+")
    parser.add_argument("--eval-dir", default="../../dicts")
    parser.add_argument("--context", default="../chinese-ime-lm/eval/sentences-ctx-v1/sampled.jsonl.gz")
    parser.add_argument("--new-words", default="wordlists/new_words.tsv")
    args = parser.parse_args()

    pairs = rows(os.path.join(args.eval_dir, "eval-words.tsv"))
    gold_words = {p[1] for p in pairs}
    leaked = sorted(gold_words & {r[0] for r in rows(args.new_words)})
    sentences = [r for r in rows(os.path.join(args.eval_dir, "eval-shared-v1.tsv"))
                 if HANZI.match(r[3])]
    with gzip.open(args.context, "rt", encoding="utf-8") as f:
        # 偶数 unit 是 build.py 的开发集，这里只用奇数半。
        cases = [c for c in map(json.loads, f) if c["style"] == "full" and HANZI.match(c["gold"])
                 and len(c["gold"]) == len(c["readings"]) and zlib.crc32(c["unit"].encode()) % 2]

    failed = bool(leaked)
    if leaked:
        print(f"[fail] 人工新词混入评测词: {leaked}")
    for path in args.dicts:
        words = load(path)
        probes = [w for w, k in PROBES.items() if k not in words.get(w, {})]
        metric = word_metrics(words, pairs)
        segs = [reachable(words, r[3], r[2]) for r in sentences]
        ok = [s for s, r in zip(segs, sentences) if s is not None]
        reach = len(ok) / len(sentences)
        reading, wrong = reading_accuracy(words, cases)
        syl = Counter(s for keys in words.values() for k in keys for s in k.split("'"))
        result = {"probes": 1 - len(probes) / len(PROBES), "recall": metric["recall"],
                  "reach": reach, "reading": reading}
        bad = [k for k, floor in GATES.items() if result[k] < floor]
        bad += ["nv/lv"] if not (syl["nv"] and syl["lv"]) else []
        failed |= bool(bad)
        print(f"{os.path.basename(path)}: 词 {len(words)} 词形 {metric['form']:.2%} 同键召回 {metric['recall']:.2%} "
              f"top1 {metric['top1']:.2%} top5 {metric['top5']:.2%} 可达 {reach:.2%} 词读音 {reading:.2%} "
              f"探针 {result['probes']:.0%} {'通过' if not bad else '失败: ' + ','.join(bad)}")
        if probes:
            print(f"  探针错误: {[(w, list(words.get(w, {}))) for w in probes]}")
        print(f"  读音错误样本: {wrong.most_common(20)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
