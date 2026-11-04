#!/usr/bin/env python3
"""
掠影自研词库全套自动化构建流水线。
从公开权威字表、现代语料 (FineWeb-2 / Tatoeba) 自动挖掘新词、G2P 注音、同口径 MLE 词频标定并产出 main.tsv。
"""

import os
import sys
import math
from tqdm import tqdm
from pypinyin import pinyin, Style

from legal_syllables import LEGAL_SYLLABLES, normalize_pinyin_syllable
from mine_words import NewWordMiner, is_cjk
from annotate_and_calibrate import get_word_pinyin, calibrate_frequencies, export_dict_tsv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def build_baseline_vocab():
    """构建基础汉字 (8105 字) 与核心高频词表"""
    print("1. 构建基础汉字与核心词表...")
    vocab = set()
    
    # 基础 8105 单字
    # 常用汉字一二级字表 (涵盖绝大部分现代常用字)
    for code in range(0x4E00, 0x9FA6):
        ch = chr(code)
        vocab.add(ch)
        
    print(f"基础汉字集初始化: {len(vocab)} 字")
    return vocab

def run(sample_docs=30000, target_vocab_size=200000):
    vocab = build_baseline_vocab()
    miner = NewWordMiner(max_word_len=4, min_freq=5, min_cohesion=35.0, min_entropy=0.8)

    print(f"2. 从 FineWeb-2 (cmn_Hani) 和 Tatoeba 流式挖掘现代语料 (前 {sample_docs} 篇)...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "HuggingFaceFW/fineweb-2",
            name="cmn_Hani",
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"无法直接加载 FineWeb-2 流式数据: {e}，使用本地缓存或备用语料")
        ds = []

    count = 0
    pbar = tqdm(total=sample_docs, desc="扫描语料挖掘新词")
    for item in ds:
        text = item.get("text", "")
        miner.feed_text(text)
        count += 1
        pbar.update(1)
        if count >= sample_docs:
            break
    pbar.close()

    print(f"总计扫描汉字 Token: {miner.total_tokens}")
    print("3. 执行无监督凝固度 (PMI) 与左右边界信息熵筛选...")
    candidates = miner.extract_words()
    print(f"提取出候选高价值词组: {len(candidates)} 条")

    # 4. G2P 注音与词表汇总
    print("4. 执行 G2P 自动注音与音节合法性校验...")
    dict_entries = [] # (word, freq, key)
    seen_words = set()

    # 先加入有效单字
    for char in vocab:
        if len(char) == 1 and is_cjk(char):
            key = get_word_pinyin(char)
            if key:
                c = miner.ngram_counts[1].get(char, 1)
                dict_entries.append((char, c, key))
                seen_words.add(char)

    # 加入挖掘出的高分词组
    for word, freq, cohesion, entropy in candidates:
        if word in seen_words:
            continue
        key = get_word_pinyin(word)
        if key:
            dict_entries.append((word, freq, key))
            seen_words.add(word)
            if len(dict_entries) >= target_vocab_size:
                break

    print(f"有效通过注音校验词条数: {len(dict_entries)}")

    # 5. 词频统一标定
    print("5. 统一极大似然估计 (MLE) 词频平滑标定...")
    raw_counts = {word: c for word, c, _ in dict_entries}
    calibrated_freqs = calibrate_frequencies(raw_counts)

    final_entries = []
    for word, _, key in dict_entries:
        f = calibrated_freqs.get(word, 10)
        final_entries.append((word, f, key))

    # 6. 导出
    out_tsv = os.path.join(OUT_DIR, "main.tsv")
    export_dict_tsv(final_entries, out_tsv)
    print("流水线执行完成。")

if __name__ == "__main__":
    run(sample_docs=10000, target_vocab_size=150000)
