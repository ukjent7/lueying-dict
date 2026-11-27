#!/usr/bin/env python3
"""Round-4 A/B ruler with corrected missing-output score and document bootstrap."""

import argparse
import importlib.util
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HOME = Path("/root/work/patches5")
OLD = Path("/root/work/patches4/harness4")
sys.path.insert(0, str(OLD))
from arms import build_decoder, load_rows, word_metrics
from decode2 import MAX_LEFT, PENALTY, SCALE, load_cases, norm

CTX = "/root/work/sampled.jsonl.gz"
FW = "/root/work3/a7/ctx-2024.jsonl.gz"
V1 = "/root/work/eval/eval-words.tsv"
V2 = OLD / "eval-words-v2.tsv"
PAIR_COUNTS = "/root/work4/rs4/pairs-fw2-full.pkl"


def cases():
    return {"ctx-v1-odd": load_cases(CTX), "fw2-2024": load_cases(FW)}


def pairs(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip() and not line.startswith("#")]


def read_rows(path, legacy_keys=False):
    rows = load_rows(path)
    if legacy_keys:
        return rows
    return [(w, "'".join(map(norm, key.split("'"))), f) for w, key, f in rows]


def units(dec, rows):
    result = np.empty((len(rows), 5), dtype=np.int64)
    for i, case in enumerate(rows):
        gold = case["gold"]
        got = dec.decode([norm(s) for s in case["readings"]])
        if got and len(got) != len(gold):
            raise ValueError(f'{case["id"]}: decoded a misaligned entry {got!r}')
        char = sum(a == b for a, b in zip(got, gold))
        result[i] = got == gold, char, char + len(gold) - len(got), len(gold), not got
    return result


def absolute(data):
    exact, char, legacy, length, empty = data.sum(axis=0).tolist()
    return {"exact": exact / len(data), "char": char / length, "char_legacy": legacy / length,
            "correct": exact, "n": len(data), "empty": empty}


def paired(now, base, groups, rounds, seed):
    unique = dict.fromkeys(groups)
    index = {group: i for i, group in enumerate(unique)}
    ids = np.asarray([index[g] for g in groups])
    a, b = absolute(now), absolute(base)
    metrics = ("exact", "char", "char_legacy")
    diff = {k: a[k] - b[k] for k in metrics}
    plus = np.count_nonzero(now[:, 0] > base[:, 0]).item()
    minus = np.count_nonzero(now[:, 0] < base[:, 0]).item()
    summary = np.zeros((len(unique), 5), dtype=np.int64)
    np.add.at(summary, ids, np.column_stack((now[:, :3] - base[:, :3], np.ones(len(ids)), now[:, 3])))
    draw = np.random.default_rng(seed)
    draws = []
    for start in range(0, rounds, 128):
        idx = draw.integers(len(unique), size=(min(128, rounds - start), len(unique)))
        sample = summary[idx].sum(axis=1)
        draws.append(sample[:, :3] / sample[:, [3, 4, 4]])
    lo, hi = np.quantile(np.concatenate(draws), [0.025, 0.975], axis=0)
    return {"diff": diff, "ci": {k: [float(lo[i]), float(hi[i])] for i, k in enumerate(metrics)},
            "gained": plus, "lost": minus, "clusters": len(unique)}


def cluster(name, rows):
    if name == "fw2-2024":
        return [c["unit"].rsplit("-", 1)[0] for c in rows]
    return [c["unit"] for c in rows]


def decode_report(dec, datasets, baselines, rounds, seed):
    result, all_units = {}, {}
    for name, rows in datasets.items():
        u = all_units[name] = units(dec, rows)
        item = result[name] = absolute(u)
        if baselines is not None:
            b = baselines[name]
            item.update(paired(u, b, cluster(name, rows), rounds, seed))
        else:
            item["clusters"] = len(set(cluster(name, rows)))
    return result, all_units


def lexical(args):
    datasets = cases()
    words_eval = {"v1": pairs(V1), "v2": pairs(V2)}
    results = {}
    baseline = None
    for spec in args.dicts:
        name, path = spec.split("=", 1)
        dec, words, best = build_decoder(read_rows(path, args.legacy_keys))
        report, u = decode_report(dec, datasets, baseline, args.bootstrap, args.seed)
        if baseline is None:
            baseline = u
        report["words"] = len(words)
        for version, gold in words_eval.items():
            report[version] = word_metrics(words, gold)
        results[name] = report
        print(json.dumps({name: report}), flush=True)
        del dec, words, best
    return results


class ContextDecoder:
    """Same round-4 unigram and pair terms; delay the winner until after context."""

    def __init__(self, words, keep, table, beta):
        freq = defaultdict(list)
        for word, keys in words.items():
            for key, f in keys.items():
                freq[key].append((f, word))
        self.edges = {k: [(word, math.log(max(f, 1) / SCALE) - PENALTY)
                          for f, word in sorted(hits, key=lambda p: (-p[0], p[1]))[:keep]]
                      for k, hits in freq.items()}
        self.table = table
        self.beta = beta

    def decode(self, syl):
        n = len(syl)
        grid = [{} for _ in range(n + 1)]
        grid[0][""] = (0.0, "", 0)
        for i in range(n):
            here = grid[i]
            if not here:
                continue
            for end in range(i + 1, min(i + 10, n) + 1):
                edges = self.edges.get("'".join(syl[i:end]), ())
                dest = grid[end]
                for w, lp in edges:
                    for prev, (sc, _, _) in here.items():
                        add = (self.beta * self.table.get(prev + "\0" + w, 0.0)
                               if prev and len(prev) <= MAX_LEFT else 0.0)
                        score = sc + lp + add
                        if w not in dest or score > dest[w][0]:
                            dest[w] = (score, prev, i)
        if not grid[n]:
            return ""
        words = []
        w = max(grid[n], key=lambda k: grid[n][k][0])
        while n:
            words.append(w)
            _, w, n = grid[n][w]
        return "".join(reversed(words))


def load_collocate(path):
    spec = importlib.util.spec_from_file_location("exp_collocate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collocation(args):
    mod = load_collocate(args.collocate)
    datasets = cases()
    dict_path = args.dicts[0]
    rows = read_rows(dict_path, args.legacy_keys)
    legacy, words, best = build_decoder(rows)
    del rows, best
    baseline, output = None, {}
    if not args.backoff:
        report, baseline = decode_report(legacy, datasets, None, args.bootstrap, args.seed)
        output["legacy"] = report
        print(json.dumps({"legacy": report}), flush=True)
    del legacy
    with open(PAIR_COUNTS, "rb") as f:
        data = pickle.load(f)
    quantized = lambda ranking: {k: mod.quantize(score) / mod.SCALE for _, score, k in ranking}
    tables, rank = {}, (("ordinary", None),)
    if not args.backoff:
        rank += (("priority", mod.homophone_pressure(dict_path)),)
    for name, pressure in rank:
        ranked = mod.score_pairs(data["uni"], data["counts"], data["tokens"], data["total"], 6.0, pressure)
        ranked = ranked[:args.limit]
        tables[name] = quantized(ranked)
        if args.write_tables:
            mod.export(ranked, args.limit, str(args.report.with_name(name + "-fw2.bin")))
        if args.backoff:
            tuned = load_collocate(args.backoff).backoff_scores(
                ranked, data["uni"], data["counts"], data["tokens"], data["total"], 6.0)
            tables["dirichlet"] = quantized(tuned)
            if args.write_tables:
                mod.export(tuned, args.limit, str(args.report.with_name("dirichlet-fw2.bin")))
        del ranked
    del data, rank
    # A/B capacity uses the same fixed vocabulary, K, score, beta, and capacity.
    # Additional K=1/K=6 runs expose what round 4's oracle saw.
    frozen = None
    options = (("ordinary-k1", 1, "ordinary"), ("ordinary-k6", 6, "ordinary"),
               ("priority-k6", 6, "priority"))
    if args.backoff:
        options = (("ordinary-k6", 6, "ordinary"), ("dirichlet-k6", 6, "dirichlet"))
    for name, k, tab in options:
        dec = ContextDecoder(words, k, tables[tab], args.beta)
        report, u = decode_report(dec, datasets, baseline if tab == "ordinary" else frozen,
                                  args.bootstrap, args.seed)
        if name == "ordinary-k6":
            frozen = u
        output[name] = report
        print(json.dumps({name: report}), flush=True)
    a, b = tables.values()
    shared = a.keys() & b.keys()
    output["meta"] = {"beta": args.beta, "limit": args.limit, "counts": PAIR_COUNTS,
                      "common_pairs": len(shared),
                      "common_pairs_with_different_scores": sum(a[k] != b[k] for k in shared)}
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicts", nargs="+", required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--collocate", type=Path)
    ap.add_argument("--backoff", type=Path, help="Compare a score-only variant instead of capacity")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=420000)
    ap.add_argument("--write-tables", action="store_true")
    ap.add_argument("--legacy-keys", action="store_true")
    args = ap.parse_args()
    values = collocation(args) if args.collocate else lexical(args)
    args.report.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
