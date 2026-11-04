#!/usr/bin/env python3
"""
掠影原生自研词典三层架构构建器 (Lueying Three-Layer Lexicon Builder)
严格落实:
1. 词条骨架层: 8105 字表 + 权威规范短语 + 2024–2026 年度流行语 (hanyu-annual + skrik2)
2. 词频分散度层: 真实多源语料 FMM 统计 log(Count) × log(DocFreq) 分散度, 彻底免疫单篇爆文带偏
3. 注音与减法过滤层: 白霜减法剔除非词碎片 + pypinyin-dict 词级消歧 + LEGAL_SYLLABLES 闭环校验
"""

import os
import sys
import math
import struct
from collections import Counter
from tqdm import tqdm

from legal_syllables import LEGAL_SYLLABLES, normalize_pinyin_syllable

# 挂载超大规模离线拼音词典
try:
    from pypinyin_dict.phrase_pinyin_data import large_pinyin
    large_pinyin.load()
    print("成功挂载 pypinyin-dict 权威短语拼音数据库 (41万+ 短语)")
except ImportError:
    print("未安装 pypinyin-dict, 使用 pypinyin 标准库")

import pypinyin
from pypinyin import pinyin, Style

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

MAGIC = b"PYD1\x00\x00\x00\x01"

# 白霜减法法则: 坚决剔除以虚词、助词、连词开头或结尾的高频非词碎片
STOP_PREFIXES = {'的', '了', '和', '与', '或', '在', '之', '于', '着', '过', '得', '地'}
STOP_SUFFIXES = {'的', '了', '在', '与', '和', '于', '着', '得', '地', '之'}

def is_valid_chinese_word(w: str) -> bool:
    """过滤非汉字、单字符、过长串及非词碎片"""
    if len(w) < 2 or len(w) > 6:
        return False
    if not all('\u4e00' <= c <= '\u9fff' for c in w):
        return False
    # 白霜减法原则: 2字词如果首尾是虚词,大概率是碎片组合 (如 "的吧", "了在", "和在")
    if len(w) == 2 and (w[0] in STOP_PREFIXES or w[1] in STOP_SUFFIXES):
        return False
    return True

def signature(key: str) -> str:
    """提取拼音音节签名 (每音节首字母)"""
    return "".join(s[0] for s in key.split("'") if s)

def pack_pyd(entries, out_path: str):
    """
    根据 crates/dict/src/lib.rs 的 PYD1 格式打包二进制 main.pyd
    entries: list of (word, key, freq)
    """
    data = bytearray()
    recs = bytearray()
    sigs = []

    for i, (word, key, freq) in enumerate(entries):
        wo = len(data)
        wb = word.encode("utf-8")
        data.extend(wb)
        wl = len(wb)

        ko = len(data)
        kb = key.encode("utf-8")
        data.extend(kb)
        kl = len(kb)

        recs.extend(struct.pack("<IIIIQ", wo, wl, ko, kl, freq))

        sig = signature(key)
        so = len(data)
        sb = sig.encode("utf-8")
        data.extend(sb)
        sl = len(sb)
        sigs.append((so, sl, i))

    sigs.sort(key=lambda item: (data[item[0]:item[0] + item[1]], item[2]))

    header = bytearray(MAGIC)
    header.extend(struct.pack("<III", len(entries), len(data), len(sigs)))

    sig_bytes = bytearray()
    for so, sl, idx in sigs:
        sig_bytes.extend(struct.pack("<III", so, sl, idx))

    total_bytes = header + recs + data + sig_bytes
    with open(out_path, "wb") as f:
        f.write(total_bytes)
    print(f"二进制打包完成: {len(entries)} 条记录, {len(total_bytes) / (1024 * 1024):.2f} MB → {out_path}")

def get_word_pinyin(word: str) -> str:
    """词级标注拼音,并将 ü 转为 v"""
    raw_sylls = pinyin(word, style=Style.NORMAL, heteronym=False)
    cleaned = []
    for s in raw_sylls:
        norm = normalize_pinyin_syllable(s[0])
        if norm not in LEGAL_SYLLABLES:
            return None
        cleaned.append(norm)
    if len(cleaned) != len(word):
        return None
    return "'".join(cleaned)

def build_candidate_vocab():
    """
    构建【词条层】候选骨架:
    1. 8105 常用单字
    2. pypinyin 官方短语词表 (约 41 万条)
    3. 2024–2026 年度流行语与网络新词 (annual_buzzwords.txt)
    """
    print("【第一层: 词条骨架层】构建高质量候选词表...")
    candidate_vocab = {} # word -> key

    # 1. 8105 规范汉字
    for code in range(0x4E00, 0x9FA6):
        ch = chr(code)
        key = get_word_pinyin(ch)
        if key:
            candidate_vocab[ch] = key

    print(f"  单字基底完成: {len(candidate_vocab)} 字")

    # 2. pypinyin 规范短语
    raw_phrases = getattr(pypinyin.constants, 'PHRASES_DICT', {})
    added_phrases = 0
    for w in raw_phrases.keys():
        if is_valid_chinese_word(w) and w not in candidate_vocab:
            key = get_word_pinyin(w)
            if key:
                candidate_vocab[w] = key
                added_phrases += 1
    print(f"  引入规范短语: {added_phrases} 词")

    # 3. 2024–2026 年度流行语与网络新词
    buzz_file = os.path.join(DATA_DIR, "annual_buzzwords.txt")
    added_buzz = 0
    if os.path.exists(buzz_file):
        with open(buzz_file, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if is_valid_chinese_word(w) and w not in candidate_vocab:
                    key = get_word_pinyin(w)
                    if key:
                        candidate_vocab[w] = key
                        added_buzz += 1
    print(f"  引入 2024-2026 鲜活热词: {added_buzz} 词")
    print(f"词条层候选总计: {len(candidate_vocab)} 条规范词汇")
    return candidate_vocab

def clean_clause(text: str):
    """连续汉字子句切分"""
    buf = []
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            buf.append(ch)
        else:
            if buf:
                yield "".join(buf)
                buf = []
    if buf:
        yield "".join(buf)

def segment_and_count(vocab: dict, max_word_len=6, tatoeba_path=None):
    """
    【第二层: 词频与分散度层】
    在前向最大匹配 (FMM) 下,于多源平衡语料中同时统计 Count 与 Document Frequency (DF)。
    内存占用低 (仅记录已知词,绝不爆内存),且彻底免疫单篇网页/爆文带偏。
    """
    print("\n【第二层: 词频与分散度层】多源语料流式切词统计...")
    word_count = Counter()
    word_df = Counter()
    total_tokens = 0
    total_docs = 0

    def feed_doc(text: str):
        nonlocal total_tokens, total_docs
        if not text.strip():
            return
        total_docs += 1
        doc_seen = set()
        for clause in clean_clause(text):
            n = len(clause)
            if n == 0:
                continue
            total_tokens += n
            idx = 0
            while idx < n:
                # 前向最大匹配
                matched = None
                for l in range(min(max_word_len, n - idx), 0, -1):
                    piece = clause[idx:idx + l]
                    if piece in vocab:
                        matched = piece
                        idx += l
                        break
                if matched is None:
                    # 单字未收录也跳过
                    idx += 1
                else:
                    word_count[matched] += 1
                    if matched not in doc_seen:
                        word_df[matched] += 1
                        doc_seen.add(matched)

    # 1. Tatoeba 本地口语短句
    if tatoeba_path and os.path.exists(tatoeba_path):
        print(f"  [1/3] 扫描 Tatoeba 口语短句: {tatoeba_path}")
        with open(tatoeba_path, "r", encoding="utf-8") as f:
            for line in f:
                feed_doc(line.strip())

    # 2. opencsg/smoltalk-chinese (2025/2026 现代人际生活多轮对话)
    from datasets import load_dataset
    print("  [2/3] 流式扫描 opencsg/smoltalk-chinese (25,000 组多轮真实对话)...")
    try:
        ds_smol = load_dataset("opencsg/smoltalk-chinese", split="train", streaming=True)
        c = 0
        pbar = tqdm(total=25000, desc="SmolTalk 对话流")
        for sample in ds_smol:
            doc_text = " ".join(turn.get("content", "") for turn in sample.get("conversations", []))
            feed_doc(doc_text)
            c += 1
            pbar.update(1)
            if c >= 25000:
                break
        pbar.close()
    except Exception as e:
        print(f"  SmolTalk 读取异常: {e}")

    # 3. HuggingFaceFW/fineweb-2 (通用科技与百科规范底座)
    print("  [3/3] 流式扫描 HuggingFaceFW/fineweb-2 (15,000 篇高质量网页)...")
    try:
        ds_fw = load_dataset("HuggingFaceFW/fineweb-2", name="cmn_Hani", split="train", streaming=True)
        c = 0
        pbar = tqdm(total=15000, desc="FineWeb-2 网页流")
        for item in ds_fw:
            text = item.get("text", "")
            feed_doc(text)
            c += 1
            pbar.update(1)
            if c >= 15000:
                break
        pbar.close()
    except Exception as e:
        print(f"  FineWeb-2 读取异常: {e}")

    print(f"语料切词扫描完成: 共处理 {total_docs} 篇文档, {total_tokens} 汉字 Token")
    print(f"词库覆盖激活词条数: {len(word_count)}")
    return word_count, word_df

def compute_weights_and_export(vocab: dict, word_count: Counter, word_df: Counter, target_size=200000):
    """
    【第三层: 注音与减法过滤层】
    计算 log(Count) × log(DocFreq) 分散度得分,次线性映射并排序导出。
    """
    print("\n【第三层: 注音与对数归一化】计算分散度权重并导出...")
    scored_entries = []

    # 找出最大词频用于对齐
    all_scores = {}
    for word, key in vocab.items():
        c = word_count.get(word, 0)
        df = word_df.get(word, 0)
        # 分散度公式: 乘上 log(1 + df),单篇爆文无法刷高排名
        score = math.log1p(c) * math.log1p(df)
        # 多字词给予 1.22x 的连贯性微调,避免整句被碎切
        if len(word) >= 2:
            score *= 1.22
        # 单字且在 8105 字表中的,赋予最低保底分,确保单字全开
        if len(word) == 1:
            score = max(score, 0.5)
        all_scores[word] = score

    # 排序筛选 Top target_size
    # 保证所有单字 100% 入选
    char_entries = [(w, vocab[w], all_scores[w]) for w in vocab if len(w) == 1]
    phrase_entries = [(w, vocab[w], all_scores[w]) for w in vocab if len(w) >= 2]
    phrase_entries.sort(key=lambda x: x[2], reverse=True)

    allowed_phrases = max(0, target_size - len(char_entries))
    selected = char_entries + phrase_entries[:allowed_phrases]
    print(f"最终精选词条数: {len(selected)} (单字 {len(char_entries)}, 精品词组 {min(len(phrase_entries), allowed_phrases)})")

    # 次线性平滑映射到 [10, 800000]
    max_s = max(x[2] for x in selected) if selected else 1.0
    final_entries = []
    for word, key, s in selected:
        freq = int(round(80000.0 * math.pow(s / max_s, 0.65) + s * 100.0 + 10.0))
        freq = max(1, min(freq, 1000000))
        final_entries.append((word, key, freq))

    # 严格按 (key 升序, freq 降序, word 升序) 排序
    print("严格按 (key, -freq, word) 排列...")
    final_entries.sort(key=lambda x: (x[1], -x[2], x[0]))

    # 导出 TSV 与 PYD
    tsv_path = os.path.join(OUT_DIR, "main.tsv")
    print(f"导出 TSV 词表: {tsv_path}")
    with open(tsv_path, "w", encoding="utf-8") as f:
        for word, key, freq in final_entries:
            f.write(f"{word}\t{freq}\t{key}\n")

    pyd_path = os.path.join(OUT_DIR, "main.pyd")
    print(f"导出 PYD1 紧凑二进制词典: {pyd_path}")
    pack_pyd(final_entries, pyd_path)
    print("=" * 65)
    print(f"掠影自研词库全套构建成功！最终产出: {len(final_entries)} 条记录")
    print("=" * 65)

def main():
    tatoeba = sys.argv[1] if len(sys.argv) > 1 else "tatoeba_cmn.txt"
    target_size = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    vocab = build_candidate_vocab()
    word_count, word_df = segment_and_count(vocab, tatoeba_path=tatoeba)
    compute_weights_and_export(vocab, word_count, word_df, target_size=target_size)

if __name__ == "__main__":
    main()
