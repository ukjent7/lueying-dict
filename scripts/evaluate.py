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
import random
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
BOOTSTRAP = 1000


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


def word_units(words: dict, pairs: list[list[str]]) -> list[tuple[int, int, int, int]]:
    """每对金标一个 unit：(词形对, 同键命中, top1, top5)。bootstrap 按 unit 重采样。"""
    index: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for word, keys in words.items():
        for key, freq in keys.items():
            index[norm(key)].append((freq, word))
    for items in index.values():
        items.sort(reverse=True)
    units = []
    for pinyin, gold in pairs:
        ranked = [w for _, w in index.get(norm(pinyin), [])]
        form = int(gold in words)
        hit = int(gold in ranked)
        rank = ranked.index(gold) if hit else -1
        units.append((form, hit, int(rank == 0), int(0 <= rank < 5)))
    return units


def word_metrics(words: dict, pairs: list[list[str]]) -> dict:
    units = word_units(words, pairs)
    n = max(len(units), 1)
    return {"form": sum(u[0] for u in units) / n, "recall": sum(u[1] for u in units) / n,
            "top1": sum(u[2] for u in units) / n, "top5": sum(u[3] for u in units) / n}


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


def poly_chars(words: dict) -> set[str]:
    """词典里出现过不止一种读音的字。宏平均只在这些字上取，对齐 g2pW 的逐字口径。"""
    opts: dict[str, set[str]] = defaultdict(set)
    for word, keys in words.items():
        for key in keys:
            for ch, s in zip(word, key.split("'")):
                opts[ch].add(norm(s))
    return {ch for ch, seen in opts.items() if len(seen) > 1}


def reading_units(words: dict, cases: list[dict], poly: set[str]) -> tuple[list, Counter]:
    """每个 case 的 (命中词计数, 多音字计数) 及错误样本。

    词/字计数是 {key: (对, 总)}，bootstrap 时合并重采样后的 case 再算宏平均。
    """
    units = []
    wrong: Counter[str] = Counter()
    for case in cases:
        text, syl = case["gold"], case["readings"]
        by_word: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        by_char: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        i = 0
        while i < len(text):
            for j in range(min(len(text), i + 8), i + 1, -1):
                if text[i:j] not in words:
                    continue
                word = text[i:j]
                want = norm("".join(syl[i:j]))
                raw_keys = list(words[word])
                keys = [norm(k) for k in raw_keys]
                ok = want in keys
                by_word[word][0] += ok
                by_word[word][1] += 1
                if not ok:
                    wrong[word] += 1
                for off, ch in enumerate(word):
                    if ch not in poly:
                        continue
                    gold_syl = norm(syl[i + off])
                    by_char[ch][0] += int(any(len(k.split("'")) > off and
                                              norm(k.split("'")[off]) == gold_syl for k in raw_keys))
                    by_char[ch][1] += 1
                i = j
                break
            else:
                i += 1
        units.append((dict(by_word), dict(by_char)))
    return units, wrong


def reading_stats(units: list) -> tuple[float, float, float]:
    """返回 (微平均, 词宏平均, 多音字宏平均)。"""
    # 微平均按全部命中词算，不受单字读音口径影响
    micro_good = sum(c[0] for per_word, _ in units for c in per_word.values())
    micro_total = sum(c[1] for per_word, _ in units for c in per_word.values())
    words: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    chars: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for per_word, per_char in units:
        for k, (g, t) in per_word.items():
            words[k][0] += g
            words[k][1] += t
        for k, (g, t) in per_char.items():
            chars[k][0] += g
            chars[k][1] += t
    macro_word = sum(g / t for k, (g, t) in words.items() if t and len(k) >= 2) / max(
        sum(1 for k, (_, t) in words.items() if t and len(k) >= 2), 1)
    macro_char = sum(g / t for g, t in chars.values() if t) / max(sum(1 for _, t in chars.values() if t), 1)
    return micro_good / max(micro_total, 1), macro_word, macro_char


def reading_accuracy(words: dict, cases: list[dict]) -> tuple[float, Counter]:
    """最长匹配切出的多字词里，词典读音与金标准一致的比例（微平均）。"""
    units, wrong = reading_units(words, cases, poly_chars(words))
    return reading_stats(units)[0], wrong


def bootstrap(values: list[float], rounds: int = BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    """按 unit 有放回重采样，返回 95% 置信区间。"""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return stats[int(0.025 * rounds)], stats[int(0.975 * rounds) - 1]


def reading_ci(units: list, index: int, rounds: int, seed: int) -> tuple[float, float]:
    """宏平均的 bootstrap：重采样 case，合并计数后再取宏平均。"""
    rng = random.Random(seed)
    n = len(units)
    vals = []
    for _ in range(rounds):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        vals.append(reading_stats(sample)[index])
    vals.sort()
    return vals[int(0.025 * rounds)], vals[int(0.975 * rounds) - 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dicts", nargs="+")
    parser.add_argument("--eval-dir", default="../../dicts")
    parser.add_argument("--context", default="../chinese-ime-lm/eval/sentences-ctx-v1/sampled.jsonl.gz")
    parser.add_argument("--new-words", default="wordlists/new_words.tsv")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP, help="bootstrap 重采样次数")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap 随机种子")
    parser.add_argument("--report", default="", help="把指标与置信区间写成 JSON")
    parser.add_argument("--macro-gate", type=float, default=None,
                        help="reading_macro_char 门限；不传只报数不设线")
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

    report: dict[str, dict] = {}
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
        units = word_units(words, pairs)
        runits, wrong = reading_units(words, cases, poly_chars(words))
        reading, macro_word, macro_char = reading_stats(runits)
        cis = {"recall": bootstrap([float(u[1]) for u in units], args.bootstrap, args.seed),
               "top1": bootstrap([float(u[2]) for u in units], args.bootstrap, args.seed),
               "top5": bootstrap([float(u[3]) for u in units], args.bootstrap, args.seed),
               "reach": bootstrap([float(s is not None) for s in segs], args.bootstrap, args.seed),
               "reading": reading_ci(runits, 0, args.bootstrap, args.seed),
               "macro_word": reading_ci(runits, 1, args.bootstrap, args.seed),
               "macro_char": reading_ci(runits, 2, args.bootstrap, args.seed)}
        syl = Counter(s for keys in words.values() for k in keys for s in k.split("'"))
        result = {"probes": 1 - len(probes) / len(PROBES), **metric,
                  "reach": reach, "reading": reading, "macro_word": macro_word,
                  "macro_char": macro_char}
        gates = dict(GATES)
        if args.macro_gate is not None:
            gates["macro_char"] = args.macro_gate
        bad = [k for k, floor in gates.items() if result[k] < floor]
        bad += ["nv/lv"] if not (syl["nv"] and syl["lv"]) else []
        failed |= bool(bad)
        print(f"{os.path.basename(path)}: 词 {len(words)} 词形 {metric['form']:.2%} 同键召回 {metric['recall']:.2%} "
              f"top1 {metric['top1']:.2%} top5 {metric['top5']:.2%} 可达 {reach:.2%} 词读音 {reading:.2%} "
              f"宏词 {macro_word:.2%} 宏字 {macro_char:.2%} "
              f"探针 {result['probes']:.0%} {'通过' if not bad else '失败: ' + ','.join(bad)}")
        print("  CI95 " + " ".join(f"{k}=[{lo:.2%},{hi:.2%}]" for k, (lo, hi) in cis.items()))
        report[os.path.basename(path)] = {
            "words": len(words),
            "metrics": {k: {"value": result[k], "ci": list(cis[k])} for k in cis}}
        if probes:
            print(f"  探针错误: {[(w, list(words.get(w, {}))) for w in probes]}")
        print(f"  读音错误样本: {wrong.most_common(20)}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"gates": GATES, "tiers": report}, f, ensure_ascii=False, indent=2)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
