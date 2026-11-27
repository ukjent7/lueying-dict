#!/usr/bin/env python3
"""Decompose the old reading metric and compare both arms on fixed gold spans."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from eval5 import cases, cluster, load_rows, norm, paired

import jieba
from jieba_tune import tune


def load_wordkeys(path):
    lex = {}
    for word, key, _ in load_rows(path):
        lex.setdefault(word, set()).add(norm(key))
    return lex


def greedy(gold, syl, lex):
    i, out = 0, {}
    while i < len(gold):
        for j in range(min(len(gold), i + 8), i + 1, -1):
            word = gold[i:j]
            if word in lex:
                out[(i, j, word)] = norm("".join(syl[i:j])) in lex[word]
                i = j
                break
        else:
            i += 1
    return out


def evaluate(data, words, groups):
    sums = Counter(greedy_common_novel_errors=0, greedy_control_only_errors=0,
                   greedy_treatment_only_errors=0)
    deltas = [[], []]
    removed, added = Counter(), Counter()
    for case in data:
        gold, syl = case["gold"], case["readings"]
        a, b = (greedy(gold, syl, lex) for lex in words)
        for k in a.keys() & b.keys():
            sums["greedy_common_spans"] += 1
            sums["greedy_common_novel_errors"] += a[k] and not b[k]
        for spans, other, prefix, errors in ((a, b, "control", removed), (b, a, "treatment", added)):
            for k in sorted(spans.keys() - other.keys()):
                sums[f"greedy_{prefix}_only_spans"] += 1
                if not spans[k]:
                    sums[f"greedy_{prefix}_only_errors"] += 1
                    errors[k[2]] += 1
        hits = [0, 0]
        n = pos = 0
        for word in jieba.lcut(gold, HMM=True):
            end = pos + len(word)
            key = norm("".join(syl[pos:end]))
            pos = end
            if len(word) < 2:
                continue
            n += 1
            for i, lex in enumerate(words):
                hits[i] += key in lex.get(word, ())
        for i, h in enumerate(hits):
            deltas[i].append((0, h, h, n, 0))
    a, b = map(np.array, deltas)
    fixed = paired(b, a, groups, 5000, 0)
    return {"greedy_counts": dict(sums), "added_errors": added.most_common(15),
            "removed_errors": removed.most_common(15), "fixed_segmentation_lexical_tokens": int(a[:, 3].sum()),
            "fixed_segmentation_control": int(a[:, 1].sum()),
            "fixed_segmentation_treatment": int(b[:, 1].sum()),
            "fixed_segmentation_delta": fixed["diff"]["char"], "fixed_segmentation_ci": fixed["ci"]["char"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("treatment")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    tune()
    words = [load_wordkeys(path) for path in (args.control, args.treatment)]
    result = {name: evaluate(data, words, cluster(name, data)) for name, data in cases().items()}
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
