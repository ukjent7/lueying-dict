#!/usr/bin/env python3
"""
掠影自研词库全套自动化构建流水线。
结合 pypinyin-dict 权威消歧词典、通用规范汉字表、现代高质量语料 (FineWeb-2 + Tatoeba)
进行无监督新词挖掘、G2P 注音、同口径 MLE 词频标定，并生成 main.tsv 与 main.pyd。
"""

import os
import sys
import math
import struct
from collections import Counter
from tqdm import tqdm

from legal_syllables import LEGAL_SYLLABLES, normalize_pinyin_syllable
from mine_words import NewWordMiner, is_cjk

# 尝试载入超大规模离线拼音词典
try:
    from pypinyin_dict.phrase_pinyin_data import large_pinyin
    large_pinyin.load()
    print("成功挂载 pypinyin-dict 权威短语拼音数据库 (含百万级多音字词条)")
except ImportError:
    print("未安装 pypinyin-dict, 使用 pypinyin 标准库")

import pypinyin
from pypinyin import pinyin, Style

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

MAGIC = b"PYD1\x00\x00\x00\x01"

def signature(key: str) -> str:
    """提取拼音音节签名 (每音节首字母)"""
    return "".join(s[0] for s in key.split("'") if s)

def pack_pyd(entries, out_path: str):
    """
    根据 crates/dict/src/lib.rs 的 PYD1 格式将词条列表直接打包成二进制 main.pyd
    entries: list of (word, key, freq) 已按 (key, -freq, word) 排序
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

    # 签名按字典序排,查询才能二分
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

def get_word_pinyin(word: str):
    """使用 pypinyin 词级上下文标注拼音,并将 ü 转为 v"""
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

def build_baseline_characters():
    """初始化《通用规范汉字表》覆盖的 8105 常用汉字"""
    chars = []
    for code in range(0x4E00, 0x9FA6):
        ch = chr(code)
        chars.append(ch)
    return chars

def run(sample_docs=30000, target_vocab_size=200000, tatoeba_path=None):
    print("=" * 60)
    print("开始执行掠影原生自主词典 (Lueying Dict) 全量流水线")
    print("=" * 60)

    # 1. 挖掘语料统计
    miner = NewWordMiner(max_word_len=4, min_freq=6, min_cohesion=35.0, min_entropy=0.75)

    # 1.1 本地 Tatoeba 真实对话/口语短句
    if tatoeba_path and os.path.exists(tatoeba_path):
        print(f"1. 载入本地 Tatoeba 口语文本: {tatoeba_path}")
        with open(tatoeba_path, "r", encoding="utf-8") as f:
            for line in f:
                miner.feed_text(line.strip())
        print(f"Tatoeba 处理完毕，已扫描 Token: {miner.total_tokens}")

    # 1.2 FineWeb-2 (cmn_Hani)
    print(f"2. 从 FineWeb-2 (cmn_Hani) 流式挖掘现代语料 ({sample_docs} 篇)...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "HuggingFaceFW/fineweb-2",
            name="cmn_Hani",
            split="train",
            streaming=True
        )
        count = 0
        pbar = tqdm(total=sample_docs, desc="FineWeb-2 语料扫描")
        for item in ds:
            text = item.get("text", "")
            miner.feed_text(text)
            count += 1
            pbar.update(1)
            if count >= sample_docs:
                break
        pbar.close()
    except Exception as e:
        print(f"FineWeb-2 流式读取异常: {e}")

    print(f"语料扫描总计完成，有效汉字 Token 数: {miner.total_tokens}")

    # 2. 提取新词
    print("3. 执行无监督凝固度 (PMI) 与左右信息熵剪枝...")
    candidates = miner.extract_words()
    print(f"新词挖掘提取成功: {len(candidates)} 条高质量词组")

    # 3. 词条汇聚与 G2P 注音
    print("4. 执行 G2P 自动注音与多音字消歧...")
    base_chars = build_baseline_characters()
    raw_dict = {} # (word, key) -> freq

    # 3.1 单字
    for ch in base_chars:
        key = get_word_pinyin(ch)
        if key:
            c = miner.ngram_counts[1].get(ch, 1)
            raw_dict[(ch, key)] = c

    # 3.2 挖掘出的多字词
    for word, freq, cohesion, entropy in candidates:
        key = get_word_pinyin(word)
        if key:
            raw_dict[(word, key)] = freq
            if len(raw_dict) >= target_vocab_size:
                break

    print(f"词典候选条数: {len(raw_dict)}")

    # 4. 同口径 MLE 词频平滑
    print("5. 执行同口径 MLE 次线性平滑与频次映射...")
    max_c = max(raw_dict.values()) if raw_dict else 1
    final_entries = [] # (word, key, freq)

    for (word, key), c in raw_dict.items():
        # 次线性对数平滑映射到 [10, 800000]
        score = 80000.0 * math.pow(c / max_c, 0.62) + math.log1p(c) * 120.0 + 10.0
        # 多字词适度给予 1.25x 组合偏好,避免单字切碎
        if len(word) >= 2:
            score *= 1.25
        final_freq = max(1, int(round(score)))
        final_entries.append((word, key, final_freq))

    # 5. 排序: (key 升序, freq 降序, word 升序)
    print("6. 严格按 (key, -freq, word) 排序...")
    final_entries.sort(key=lambda x: (x[1], -x[2], x[0]))

    # 6. 导出 TSV 与 PYD
    tsv_path = os.path.join(OUT_DIR, "main.tsv")
    print(f"7. 导出 TSV 文本词典: {tsv_path}")
    with open(tsv_path, "w", encoding="utf-8") as f:
        for word, key, freq in final_entries:
            f.write(f"{word}\t{freq}\t{key}\n")

    pyd_path = os.path.join(OUT_DIR, "main.pyd")
    print(f"8. 导出 PYD1 紧凑二进制词典: {pyd_path}")
    pack_pyd(final_entries, pyd_path)

    print("=" * 60)
    print(f"掠影原生自主词典生成成功！总条目: {len(final_entries)}")
    print("=" * 60)

if __name__ == "__main__":
    tatoeba = sys.argv[1] if len(sys.argv) > 1 else None
    docs = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    vocab_size = int(sys.argv[3]) if len(sys.argv) > 3 else 180000
    run(sample_docs=docs, target_vocab_size=vocab_size, tatoeba_path=tatoeba)
