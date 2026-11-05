#!/usr/bin/env python3
"""掠影词典构建：语料统计 + 上下文读音 → 多档 main.tsv / main.pyd。

词条与词频只来自 scan.py 的语料统计，混合权重在开发集上学出来；读音优先取 annotate.py
在真实句子上得到的分布，其次是词级词表（phrase-pinyin-data、CC-CEDICT），最后
按最长匹配拼接。词典档位由同一条效用排序截断：

    效用 = 混合概率 × 词表可信度 × 解码必要性

解码必要性来自拿掉这个词后的整句解码：同一拼音仍能解出这个词，说明它可由更短
的词拼出（「的影响」= 的 + 影响），收录它只增加体积，效用打折。这样碎片词自然
沉底，体积与准确率落在同一条曲线上。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import re
import struct
import sys
import unicodedata
import zlib
from collections import Counter, defaultdict

from legal_syllables import LEGAL_SYLLABLES

MAGIC = b"PYD2\x00\x00\x00\x01"
HANZI = re.compile(r"^[一-鿿]+$")
HANZI_RUN = re.compile("[\u4e00-\u9fff]+")
MAX_LEN = 10
# 与引擎 WORD_PENALTY 同义：路径每多一个词的对数代价。
WORD_PENALTY = 2.2
SCALE = 1_000_000_000
REDUNDANT = 0.05
TRUSTED = 4.0
SPAM_CAP = 500 / SCALE
# FineWeb-2 混着成人站模板：自慰 286 ppm，其余来源都不到 4。高出其余来源最高值 SPIKE 倍的
# 部分不算，只有它一家有的词最多算 LONE，这样真词回到真实词频，模板串掉出词典。
SPIKE = 5.0
LONE = 0.5e-6
MIN_SHARE = 0.08
MIN_OBSERVED = 4
AMBIGUOUS = 0.97
# 规范字表也收的异读，g2pW 却几乎只给它：血液 → xie'ye。
MAINLAND = {("血", "xie"): "xue", ("熟", "shou"): "shu"}
NEED_LIMIT = 600_000
WEB_MIN_DF = 8
# 单字读音占比下限：规范字表收录的读音即使语料罕见也要能打出来。
CHAR_FLOOR = 0.002
COMBINING_V = {"ǖ": "v", "ǘ": "v", "ǚ": "v", "ǜ": "v", "ü": "v"}


def toneless(raw: str) -> str:
    """带调拼音 → 引擎音节。ü 要在去调之前处理，否则 ǚ 会被分解成 u。"""
    s = "".join(COMBINING_V.get(ch, ch) for ch in raw.strip().lower()).replace("u:", "v")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[0-9]", "", s)
    return {"lue": "lve", "nue": "nve", "r": "er"}.get(s, s)


def split_pinyin(text: str) -> list[str]:
    return [toneless(x) for x in re.split(r"[\s,，、]+", text.strip()) if x.strip()]


def legal_key(word: str, syllables: list[str]) -> str | None:
    if len(word) == len(syllables) and all(s in LEGAL_SYLLABLES for s in syllables):
        return "'".join(syllables)
    return None


def load_8105(path: str) -> dict[str, list[str]]:
    chars: dict[str, list[str]] = {}
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3 or len(parts[1].strip()) != 1:
                continue
            ch = parts[1].strip()
            found = [s for s in dict.fromkeys(split_pinyin(parts[2])) if s in LEGAL_SYLLABLES]
            if found:
                chars[ch] = found
    # 这三个字规范读音是成音节鼻音，键盘上按常用近音输入。
    for ch, fallback in {"呒": ["wu"], "呣": ["mu"], "嗯": ["en"]}.items():
        chars.setdefault(ch, fallback)
    return chars


def load_pinyin_data(path: str) -> dict[str, list[str]]:
    chars: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            m = re.match(r"U\+([0-9A-Fa-f]+):\s*([^#]+)", raw)
            if m:
                found = [s for s in dict.fromkeys(split_pinyin(m.group(2))) if s in LEGAL_SYLLABLES]
                if found:
                    chars[chr(int(m.group(1), 16))] = found
    return chars


def load_phrases(path: str) -> dict[str, set[str]]:
    words: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for raw in f:
            if raw.startswith("#") or ":" not in raw:
                continue
            word, pinyin = (x.strip() for x in raw.split("#")[0].split(":", 1))
            if HANZI.match(word) and 2 <= len(word) <= MAX_LEN:
                key = legal_key(word, split_pinyin(pinyin))
                if key:
                    words[word].add(key)
    return words


def load_cedict(path: str, words: dict[str, set[str]]) -> set[str]:
    """CC-CEDICT 只取简体普通词；专名首字母大写，不当作通用词表。"""
    common: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for raw in f:
            m = re.match(r"\S+ (\S+) \[([^\]]+)\]", raw)
            if not m or raw.startswith("#"):
                continue
            word, pinyin = m.groups()
            if not HANZI.match(word) or not 2 <= len(word) <= MAX_LEN:
                continue
            key = legal_key(word, split_pinyin(pinyin.replace("5", "").lower()))
            if key:
                words[word].add(key)
                if not pinyin[0].isupper():
                    common.add(word)
    return common


def load_tsv(path: str) -> list[list[str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip() and not line.startswith("#")]


def load_stats(path: str) -> tuple[dict[str, tuple[int, int]], int]:
    """一个来源的词次与文档频次。1/2/3 元表里同一字串的次数相加，低频长尾不进内存。"""
    with open(path, "rb") as f:
        stats = pickle.load(f)
    merged: dict[str, tuple[int, int]] = {}
    for table in stats["grams"]:
        for word, packed in table.items():
            count, df = packed >> 32, packed & 0xFFFFFFFF
            if count >= 2 or len(word) == 1:
                old = merged.get(word, (0, 0))
                merged[word] = (old[0] + count, max(old[1], df))
    print(f"[stats] {path} tokens={stats['tokens']/1e6:.0f}M chars={stats['chars']/1e6:.0f}M "
          f"grams={len(merged)}", flush=True)
    return merged, stats["tokens"]


def dev_tokens(path: str, exclude: str) -> Counter:
    """开发集词频：ctx-v1 按 unit 取偶数半，另一半留给 evaluate.py。各来源等权。"""
    import jieba

    with open(exclude, encoding="utf-8") as f:
        held_out = {line.rstrip("\n").split("\t")[3] for line in f
                    if not line.startswith("#") and line.count("\t") >= 3}
    groups: dict[str, Counter] = defaultdict(Counter)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for case in map(json.loads, f):
            if zlib.crc32(case["unit"].encode()) % 2 or case["gold"] in held_out:
                continue
            for text in (case["gold"], case.get("context") or ""):
                for run in HANZI_RUN.findall(text):
                    groups[case["source"]].update(jieba.lcut(run, HMM=True))
    tokens: Counter = Counter()
    for group in groups.values():
        total = sum(group.values())
        for word, count in group.items():
            tokens[word] += count / total
    print(f"[dev] groups={ {k: sum(v.values()) for k, v in groups.items()} } types={len(tokens)}", flush=True)
    return tokens


def learn_weights(paths: list[str], tokens: Counter, rounds: int = 200) -> list[float]:
    """EM 求混合权重：让线性插值的 unigram 在开发集上似然最大。"""
    probs: list[dict[str, float]] = []
    for path in paths:
        table, total = load_stats(path)
        probs.append({w: table[w][0] / total for w in tokens if w in table})
    weights = [1.0 / len(paths)] * len(paths)
    mass = sum(tokens.values())
    for _ in range(rounds):
        resp = [0.0] * len(paths)
        for word, count in tokens.items():
            parts = [lam * p.get(word, 0.0) for lam, p in zip(weights, probs)]
            norm = sum(parts)
            if norm > 0:
                for i, part in enumerate(parts):
                    resp[i] += count * part / norm
        total = sum(resp) or mass
        weights = [r / total for r in resp]
    return weights


def load_seeds(paths: list[str]) -> set[str]:
    """社区词库补语料还没沉淀下来的新词。只取词形，读音和词频仍由本项目计算。"""
    words: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                word = line.split("\t", 1)[0]
                if "\t" in line and 2 <= len(word) <= 6 and HANZI.match(word):
                    words.add(word)
    return words


def is_spiky(path: str) -> bool:
    return os.path.basename(path).startswith("fineweb")


class Lexicon:
    """混合语料概率：各来源先归一到词次比例，再按学到的权重线性插值。

    Common Crawl 未经人工筛选，只在它一家出现的词要有更多站点支持才算成词。
    """

    def __init__(self, paths: list[str], weights: list[float], min_count: int, min_df: int):
        self.p: dict[str, float] = defaultdict(float)
        self.supported: set[str] = set()
        reference: dict[str, float] = defaultdict(float)
        # FineWeb-2 最后算，它要拿其余来源当参照。
        for path, weight in sorted(zip(paths, weights), key=lambda s: is_spiky(s[0])):
            table, tokens = load_stats(path)
            floor = WEB_MIN_DF if os.path.basename(path).startswith("cc") else min_df
            spiky = is_spiky(path)
            for word, (count, df) in table.items():
                share = count / tokens
                if spiky:
                    share = min(share, max(reference[word] * SPIKE, LONE))
                else:
                    reference[word] = max(reference[word], share)
                self.p[word] += share * weight
                if count >= min_count and df >= floor:
                    self.supported.add(word)
            del table


def standardize(observed: dict[str, Counter], official: dict[str, list[str]]) -> dict[str, Counter]:
    """把 g2pW 的台湾读音收敛到普通话：规范字表没有的读音改成表里的唯一读音。"""
    def syllable(ch: str, s: str) -> str | None:
        s = MAINLAND.get((ch, s), s)
        legal = official.get(ch)
        if not legal or s in legal:
            return s
        return legal[0] if len(legal) == 1 else None

    result: dict[str, Counter] = defaultdict(Counter)
    for word, seen in observed.items():
        for key, count in seen.items():
            keys = [syllable(ch, s) for ch, s in zip(word, key.split("'"))]
            if all(keys):
                result[word]["'".join(keys)] += count
    return dict(result)


def read_keys(word: str, observed: dict[str, Counter], phrases: dict[str, set[str]],
              primary: dict[str, str]) -> list[tuple[str, float]]:
    """词的读音及占比。语料观测与词表合并，都没有时最长匹配拼接。"""
    seen = observed.get(word, Counter())
    total = sum(seen.values())
    if total < MIN_OBSERVED:
        seen, total = Counter(), 0
    # 词表读音保底占一半：观测带模型偏差，词表又漏口语音（还款、都会），两边都留。
    merged = seen + Counter({key: max(total, 1) for key in phrases.get(word, ())})
    if merged:
        total = sum(merged.values())
        kept = [(k, c / total) for k, c in merged.most_common() if c / total >= MIN_SHARE]
        norm = sum(s for _, s in kept)
        if kept:
            return [(k, s / norm) for k, s in kept]
    if len(word) == 1:
        return [(primary[word], 1.0)] if word in primary else []
    parts: list[str] = []
    i = 0
    while i < len(word):
        for j in range(min(len(word), i + 6), i, -1):
            piece = word[i:j]
            found = None
            if j - i == 1:
                found = primary.get(piece)
            elif piece in phrases:
                found = sorted(phrases[piece])[0]
            elif piece in observed and sum(observed[piece].values()) >= MIN_OBSERVED:
                found = observed[piece].most_common(1)[0][0]
            if found:
                parts.append(found)
                i = j
                break
        else:
            return []
    return [("'".join(parts), 1.0)]


def necessity(entries: dict[str, list[tuple[str, float]]], p: dict[str, float]) -> dict[str, float]:
    """拿掉词本身后解码它的拼音：解回原词就是冗余。

    解码用与引擎同形的 unigram 路径分：Σ ln P(词) − 词数 × WORD_PENALTY。
    """
    best: dict[tuple[str, ...], list[tuple[float, str]]] = defaultdict(list)
    for word, keys in entries.items():
        lp = math.log(max(p.get(word, 0.0), 1e-12))
        for key, share in keys:
            best[tuple(key.split("'"))].append((lp + math.log(max(share, 1e-6)), word))
    for key, items in best.items():
        items.sort(reverse=True)
        del items[2:]

    def decode(syl: tuple[str, ...], banned: str) -> str:
        n = len(syl)
        score: list[float] = [-1e18] * (n + 1)
        back: list[tuple[int, str]] = [(0, "")] * (n + 1)
        score[0] = 0.0
        for i in range(n):
            if score[i] <= -1e18:
                continue
            for j in range(i + 1, min(n, i + 6) + 1):
                for lp, word in best.get(syl[i:j], ()):
                    if word == banned:
                        continue
                    cand = score[i] + lp - WORD_PENALTY
                    if cand > score[j]:
                        score[j] = cand
                        back[j] = (i, word)
                    break
        if score[n] <= -1e18:
            return ""
        out = []
        at = n
        while at:
            at, word = back[at][0], back[at][1]
            out.append(word)
        return "".join(reversed(out))

    result: dict[str, float] = {}
    for word, keys in entries.items():
        if len(word) == 1:
            result[word] = 1.0
            continue
        syl = tuple(keys[0][0].split("'"))
        result[word] = 0.0 if decode(syl, word) == word else 1.0
    return result


def pack_pyd(entries: list[tuple[str, str, int]], path: str) -> None:
    """PYD2：记录 14 字节，相邻同键共用字符串，签名表只存下标（与 crates/dict 同构）。"""
    data = bytearray()
    recs = bytearray()
    last_key, key_off = None, 0
    for word, key, freq in entries:
        wb, kb = word.encode(), key.encode()
        word_off = len(data)
        data += wb
        if key != last_key:
            key_off = len(data)
            data += kb
            last_key = key
        recs += struct.pack("<IIIBB", word_off, key_off, min(freq, 0xFFFFFFFF), len(wb), len(kb))
    order = sorted(range(len(entries)),
                   key=lambda i: ("".join(s[0] for s in entries[i][1].split("'")), i))
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<II", len(entries), len(data)))
        f.write(recs)
        f.write(data)
        f.write(struct.pack(f"<{len(order)}I", *order))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", nargs="+", required=True, help="各来源统计；文件名 cc 开头按网页来源处理")
    parser.add_argument("--dev", required=True, help="开发集 sampled.jsonl.gz，用来学混合权重")
    parser.add_argument("--exclude", required=True, help="eval-shared-v1.tsv，从开发集里剔除")
    parser.add_argument("--readings", required=True, help="annotate.py 的输出")
    parser.add_argument("--cache", required=True, help="8105.txt pinyin-data.txt phrase-pinyin.txt cedict.gz")
    parser.add_argument("--new-words", default="wordlists/new_words.tsv")
    parser.add_argument("--colloquial", default="wordlists/colloquial.tsv")
    parser.add_argument("--blocked", default="wordlists/blocked.txt")
    parser.add_argument("--seeds", nargs="*", default=[], help="新词候选：Rime 词库文件，只取词形")
    parser.add_argument("--sizes", default="200000,400000,600000,800000", help="多字词条档位")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--need", default="", help="导出待注音词表后退出")
    parser.add_argument("--redundant", type=float, default=REDUNDANT, help="解码得出的词的效用折扣")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cache = args.cache
    official = load_8105(f"{cache}/8105.txt")
    pinyin_data = load_pinyin_data(f"{cache}/pinyin-data.txt")
    phrases = load_phrases(f"{cache}/phrase-pinyin.txt")
    common = set(phrases) | load_cedict(f"{cache}/cedict.gz", phrases) | load_seeds(args.seeds)
    new_words = {row[0]: row for row in load_tsv(args.new_words)}
    print(f"[load] official={len(official)} pinyin_data={len(pinyin_data)} phrases={len(phrases)} "
          f"common={len(common)} new={len(new_words)}", flush=True)

    with open(args.readings, "rb") as f:
        observed_all = pickle.load(f)
    # 不足 MIN_OBSERVED 的观测 read_keys 不会用，先丢掉：几百万个 Counter 占内存最多。
    observed = standardize({w: c for w, c in observed_all["grams"].items()
                            if sum(c.values()) >= MIN_OBSERVED}, official)
    char_seen = standardize(observed_all["chars"], official)
    del observed_all

    weights = learn_weights(args.stats, dev_tokens(args.dev, args.exclude))
    print("[mix]", {os.path.basename(s): round(w, 4) for s, w in zip(args.stats, weights)}, flush=True)
    lex = Lexicon(args.stats, weights, args.min_count, args.min_df)
    p = lex.p

    if args.need:
        # 实际读音已经一边倒的字（的、在、是）不算多音，别把 GPU 花在它们身上。
        def ambiguous(ch: str, opts: list[str]) -> bool:
            seen = char_seen.get(ch)
            total = sum(seen.values()) if seen else 0
            return len(opts) > 1 and (total < 50 or max(seen.values()) / total < AMBIGUOUS)

        poly = {ch for ch, opts in {**pinyin_data, **official}.items() if ambiguous(ch, opts)}
        words = [w for w in lex.supported if 2 <= len(w) <= 6 and HANZI.match(w)
                 and w not in phrases and any(ch in poly for ch in w)]
        words.sort(key=lambda w: -p[w])
        with open(args.need, "w", encoding="utf-8") as f:
            f.write("\n".join(words[:NEED_LIMIT]))
        print(f"[need] {min(len(words), NEED_LIMIT)}/{len(words)} {words[:12]}", flush=True)
        return 0

    # 单字：规范字表全收；读音占比取语料观测，未观测的规范读音保留下限。
    primary: dict[str, str] = {}
    char_keys: dict[str, list[tuple[str, float]]] = {}
    for ch in set(official) | set(pinyin_data):
        # 规范字表按字母排序，常用读音的先后看 pinyin-data。
        usual = pinyin_data.get(ch, [])
        options = sorted(official.get(ch) or usual,
                         key=lambda s: usual.index(s) if s in usual else len(usual))
        seen = char_seen.get(ch, Counter())
        total = sum(seen.values())
        shares = {s: (seen.get(s, 0) / total if total else 0.0) for s in options}
        for s, c in seen.items():
            if total and c / total >= MIN_SHARE and s not in shares:
                shares[s] = c / total
        if not total:
            shares = {s: 1.0 / (i + 1) ** 2 for i, s in enumerate(options)}
        ordered = sorted(shares, key=lambda s: -shares[s])
        primary[ch] = ordered[0]
        norm = sum(max(v, CHAR_FLOOR) for v in shares.values())
        char_keys[ch] = [(s, max(shares[s], CHAR_FLOOR) / norm) for s in ordered]

    # 多字候选：语料支持的词 + 词表收录的词 + 人工新词。
    candidates: dict[str, list[tuple[str, float]]] = {}
    floor = min((v for v in p.values() if v > 0), default=1e-10)
    pool = {w for w in lex.supported if 2 <= len(w) <= MAX_LEN} | {w for w in common if len(w) <= 6}
    pool |= set(new_words)
    for word in pool:
        if not HANZI.match(word) or not all(ch in primary for ch in word):
            continue
        keys = read_keys(word, observed, phrases, primary)
        if keys:
            candidates[word] = keys
    for word in common:
        p[word] = max(p.get(word, 0.0), floor)
    for word, row in new_words.items():
        p[word] = max(p.get(word, 0.0), int(row[1]) / SCALE)
    # 网页语料里垃圾站的刷词清不干净：词表收录的词频封顶（别压过 自卫），别的直接不收。
    with open(args.blocked, encoding="utf-8") as f:
        spam = re.compile("|".join(w for line in f if not line.startswith("#") for w in line.split()))
    blocked = [w for w in candidates if spam.search(w)]
    for word in blocked:
        if word in common:
            p[word] = min(p[word], SPAM_CAP)
        else:
            del candidates[word]
    print(f"[select] candidates={len(candidates)} blocked={len(blocked)}", flush=True)

    singles = {ch: keys for ch, keys in char_keys.items() if ch in official}
    need = necessity({**singles, **candidates}, p)
    utility = {w: p[w] * (need[w] or args.redundant) * (TRUSTED if w in common else 1.0)
               for w in candidates}
    ranked = sorted(candidates, key=lambda w: (-utility[w], w))
    redundant = sum(1 for w in candidates if not need[w])
    print(f"[select] redundant={redundant} ({redundant/len(candidates):.1%})", flush=True)

    colloquial = load_tsv(args.colloquial)
    report = {}
    for size in (int(x) for x in args.sizes.split(",")):
        chosen = set(ranked[:size]) | set(new_words)
        rows: dict[tuple[str, str], int] = {}
        for ch, keys in singles.items():
            base = max(p.get(ch, 0.0), 1e-9)
            for key, share in keys:
                rows[(ch, key)] = max(1, round(base * share * SCALE))
        for word in chosen:
            for key, share in candidates.get(word, []):
                rows[(word, key)] = max(1, round(p[word] * share * SCALE))
        for row in new_words.values():
            if len(row) >= 3:
                rows.setdefault((row[0], row[2]), max(1, round(p[row[0]] * SCALE)))
        for word, key in ((r[0], r[1]) for r in colloquial):
            best = max((f for (w, _), f in rows.items() if w == word), default=0)
            if best:
                rows[(word, key)] = best
        entries = sorted(((w, k, f) for (w, k), f in rows.items()), key=lambda e: (e[1], -e[2], e[0]))
        tag = f"{size // 1000}k"
        with open(f"{args.out}/main-{tag}.tsv", "w", encoding="utf-8") as f:
            for word, key, freq in entries:
                f.write(f"{word}\t{freq}\t{key}\n")
        pack_pyd(entries, f"{args.out}/main-{tag}.pyd")
        mib = os.path.getsize(f"{args.out}/main-{tag}.pyd") / 2**20
        report[tag] = {"entries": len(entries), "mib": round(mib, 2),
                       "words": len({w for w, _, _ in entries})}
        print(f"[pack] {tag} entries={len(entries)} {mib:.2f} MiB", flush=True)
    with open(f"{args.out}/build_report.json", "w", encoding="utf-8") as f:
        json.dump({"tiers": report, "candidates": len(candidates), "redundant": redundant,
                   "mix": dict(zip(map(os.path.basename, args.stats), weights))}, f,
                  ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
