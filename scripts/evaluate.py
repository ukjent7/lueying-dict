#!/usr/bin/env python3
"""词典门禁：只用与构建无关的金标准，同时检查词形和读音。

词形覆盖看不出注音错误（女生 → nu'sheng 仍算「覆盖」），所以门禁以「词形和拼音
都对」为准。任何一项不达标退出码非零。

指标分两个口径：微平均（每个出现等权）与宏平均（每个词 / 每个多音字等权）。宏平均
对齐 g2pW 的 macro-by-character：整体准确率被高频词稀释，长尾多音字的错误只有逐字
平均看得见。每项指标都给 95% bootstrap 置信区间；给了 --baseline 时再做配对 bootstrap，
判据从「是否过线」升级成「是否显著优于上一个 Release」。

用法：python3 evaluate.py output/main-400k.tsv --eval-dir ../../dicts \
        --context ../chinese-ime-lm/eval/sentences-ctx-v1/sampled.jsonl.gz
      python3 evaluate.py output/main-400k.tsv ... --baseline prev.json --report now.json
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
# 聚合口径：pooled 按出现合并（微平均），macro 先按词/字取率再平均（宏平均）。
MACRO = {"macro_word", "macro_char"}
METRICS = ("form", "recall", "top1", "top5", "reach", "reading", "macro_word", "macro_char")


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


def collect(words: dict, pairs: list[list[str]], sentences: list[list[str]],
            cases: list[dict], poly: set[str]) -> tuple[dict[str, list], Counter]:
    """把两半金标准拆成「unit → (对, 总)」，bootstrap 按 unit 重采样。

    unit 是分析的最小独立单位：词形/召回类是「拼音+词」，可达是「句子」，词读音微平均是
    「句子」，宏词/宏字是「词」/「多音字」。配对比较时两侧按同一 unit 对齐。
    """
    index: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for word, keys in words.items():
        for key, freq in keys.items():
            index[norm(key)].append((freq, word))
    for items in index.values():
        items.sort(reverse=True)
    units: dict[str, list[tuple[str, int, int]]] = {name: [] for name in METRICS}
    for i, (pinyin, gold) in enumerate(pairs):
        # 位置也进 key：金标准里同一个词可能出现多次，配对比较要能一一对上。
        key = f"{i}|{norm(pinyin)}|{gold}"
        ranked = [w for _, w in index.get(norm(pinyin), [])]
        hit = gold in ranked
        rank = ranked.index(gold) if hit else -1
        units["form"].append((key, int(gold in words), 1))
        units["recall"].append((key, int(hit), 1))
        units["top1"].append((key, int(rank == 0), 1))
        units["top5"].append((key, int(0 <= rank < 5), 1))
    for i, row in enumerate(sentences):
        units["reach"].append((f"{i}|{row[0]}", int(reachable(words, row[3], row[2]) is not None), 1))

    wrong: Counter[str] = Counter()
    by_word: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_char: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ci, case in enumerate(cases):
        text, syl = case["gold"], case["readings"]
        good = total = 0
        i = 0
        while i < len(text):
            for j in range(min(len(text), i + 8), i + 1, -1):
                word = text[i:j]
                if word not in words:
                    continue
                raw_keys = list(words[word])
                want = norm("".join(syl[i:j]))
                ok = want in [norm(k) for k in raw_keys]
                total += 1
                good += ok
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
        units["reading"].append((f"{ci}|{case['unit']}", good, total))
    for word, (g, t) in by_word.items():
        if len(word) >= 2 and t:
            units["macro_word"].append((word, g, t))
    for ch, (g, t) in by_char.items():
        if t:
            units["macro_char"].append((ch, g, t))
    return units, wrong


def aggregate(units: list[tuple[str, int, int]], name: str, picks: list[int] | None = None) -> float:
    """按口径聚合：pooled 是 Σ对/Σ总，macro 是先取每个 unit 的率再平均。"""
    if picks is None:
        picks = range(len(units))
    if name in MACRO:
        vals = [units[i][1] / units[i][2] for i in picks if units[i][2]]
        return sum(vals) / max(len(vals), 1)
    good = sum(units[i][1] for i in picks)
    total = sum(units[i][2] for i in picks)
    return good / max(total, 1)


def percentile(values: list[float], rounds: int) -> tuple[float, float]:
    values.sort()
    return values[int(0.025 * rounds)], values[int(0.975 * rounds) - 1]


def bootstrap_ci(units: list[tuple[str, int, int]], name: str, rounds: int, seed: int) -> tuple[float, float]:
    """按 unit 有放回重采样，返回 95% 置信区间。"""
    if not units:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(units)
    vals = [aggregate(units, name, [rng.randrange(n) for _ in range(n)]) for _ in range(rounds)]
    return percentile(vals, rounds)


def paired_diff(now: list[tuple[str, int, int]], base: dict[str, tuple[int, int]], name: str,
                rounds: int, seed: int) -> tuple[int, float, tuple[float, float]]:
    """配对 bootstrap：两侧按同一 unit 重采样，返回 (对齐后的 unit 数, 差值, 95% CI)。

    只在两侧都有的 unit 上比，避免「词典变大 → 分母变大」把差值带偏。
    """
    common = [(k, g, t) for k, g, t in now if k in base]
    if not common:
        return 0, 0.0, (0.0, 0.0)
    mine = common
    theirs = [(k, base[k][0], base[k][1]) for k, _, _ in common]
    rng = random.Random(seed)
    n = len(common)
    vals = []
    for _ in range(rounds):
        picks = [rng.randrange(n) for _ in range(n)]
        vals.append(aggregate(mine, name, picks) - aggregate(theirs, name, picks))
    lo, hi = percentile(vals, rounds)
    return n, aggregate(mine, name) - aggregate(theirs, name), (lo, hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dicts", nargs="+")
    parser.add_argument("--eval-dir", default="../../dicts")
    parser.add_argument("--context", default="../chinese-ime-lm/eval/sentences-ctx-v1/sampled.jsonl.gz")
    parser.add_argument("--new-words", default="wordlists/new_words.tsv")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP, help="bootstrap 重采样次数")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap 随机种子")
    parser.add_argument("--report", default="", help="把指标与置信区间写成 JSON")
    parser.add_argument("--baseline", default="", help="上一个 Release 的 --report JSON，做配对显著性检验")
    parser.add_argument("--macro-gate", type=float, default=None,
                        help="reading_macro_char 门限；不传只报数不设线")
    parser.add_argument("--gain-metrics", default="",
                        help="要求显著提升的指标（逗号分隔），如 recall,macro_char")
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

    previous = {}
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            previous = json.load(f).get("tiers", {})
    gain = {x.strip() for x in args.gain_metrics.split(",") if x.strip()}

    failed = bool(leaked)
    if leaked:
        print(f"[fail] 人工新词混入评测词: {leaked}")
    report: dict[str, dict] = {}
    for path in args.dicts:
        words = load(path)
        probes = [w for w, k in PROBES.items() if k not in words.get(w, {})]
        units, wrong = collect(words, pairs, sentences, cases, poly_chars(words))
        values = {name: aggregate(units[name], name) for name in METRICS}
        cis = {name: bootstrap_ci(units[name], name, args.bootstrap, args.seed) for name in METRICS}
        result = {"probes": 1 - len(probes) / len(PROBES), **values}
        gates = dict(GATES)
        if args.macro_gate is not None:
            gates["macro_char"] = args.macro_gate
        bad = [k for k, floor in gates.items() if result[k] < floor]
        syl = Counter(s for keys in words.values() for k in keys for s in k.split("'"))
        bad += ["nv/lv"] if not (syl["nv"] and syl["lv"]) else []
        failed |= bool(bad)
        print(f"{os.path.basename(path)}: 词 {len(words)} 词形 {values['form']:.2%} 同键召回 {values['recall']:.2%} "
              f"top1 {values['top1']:.2%} top5 {values['top5']:.2%} 可达 {values['reach']:.2%} "
              f"词读音 {values['reading']:.2%} 宏词 {values['macro_word']:.2%} "
              f"宏字 {values['macro_char']:.2%} 探针 {result['probes']:.0%} "
              f"{'通过' if not bad else '失败: ' + ','.join(bad)}")
        print("  CI95 " + " ".join(f"{k}=[{lo:.2%},{hi:.2%}]" for k, (lo, hi) in cis.items()))

        tier = {"words": len(words),
                "metrics": {k: {"value": values[k], "ci": list(cis[k]), "units": len(units[k])}
                            for k in METRICS},
                "units": {k: [list(u) for u in units[k]] for k in METRICS}}
        base = previous.get(os.path.basename(path))
        if base:
            diffs = {}
            regressions = []
            for name in METRICS:
                old = {u: (g, t) for u, g, t in base["units"][name]}
                seen, diff, (lo, hi) = paired_diff(units[name], old, name, args.bootstrap, args.seed)
                diffs[name] = {"diff": diff, "ci": [lo, hi], "paired": seen,
                               "significant": lo > 0 or hi < 0}
                if hi < 0:
                    regressions.append(name)
            print("  Δ基线 " + " ".join(f"{k}={v['diff']:+.2%}[{v['ci'][0]:+.2%},{v['ci'][1]:+.2%}]"
                                       for k, v in diffs.items()))
            missing = [k for k in gain if not (diffs[k]["ci"][0] > 0)]
            bad += [f"退步:{k}" for k in regressions] + [f"未见提升:{k}" for k in missing]
            failed |= bool(regressions or missing)
            tier["baseline"] = diffs
        report[os.path.basename(path)] = tier
        if probes:
            print(f"  探针错误: {[(w, list(words.get(w, {}))) for w in probes]}")
        print(f"  读音错误样本: {wrong.most_common(20)}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"gates": GATES, "tiers": report}, f, ensure_ascii=False)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
