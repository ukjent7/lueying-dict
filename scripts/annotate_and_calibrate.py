#!/usr/bin/env python3
"""
拼音自动注音、合法性校验与统一极大似然估计 (MLE) 词频标定。
产出符合掠影规范的 main.tsv。
"""

import math
from pypinyin import pinyin, Style
from legal_syllables import LEGAL_SYLLABLES, normalize_pinyin_syllable

def get_word_pinyin(word: str) -> str:
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

def calibrate_frequencies(word_counts: dict, oral_boost_words: set = None) -> dict:
    """
    统一极大似然估计与温和次线性缩放:
    将原始语料出现频次映射到输入法最优的 [10, 1000000] 整数频次空间。
    """
    if oral_boost_words is None:
        oral_boost_words = set()
    
    calibrated = {}
    if not word_counts:
        return calibrated
    
    max_c = max(word_counts.values())
    for word, c in word_counts.items():
        # 次线性对数平滑: 避免高频词过大压死候选,也避免长尾词过小
        # f = 10000 * (c / max_c)^0.65
        score = 80000.0 * math.pow(c / max_c, 0.62) + math.log1p(c) * 150.0 + 10.0
        
        # 针对口语/高频常用词适度给予 1.5 ~ 2.5 倍温和先验
        if word in oral_boost_words:
            score *= 2.0
            
        calibrated[word] = max(1, int(round(score)))
    return calibrated

def export_dict_tsv(word_keys: list, out_path: str):
    """
    排序并导出 main.tsv
    排序准则:按 (key 字典序升序, freq 降序, word 字典序) 严格排列。
    """
    # word_keys 条目: (word, freq, key)
    word_keys.sort(key=lambda x: (x[2], -x[1], x[0]))
    
    with open(out_path, "w", encoding="utf-8") as f:
        for word, freq, key in word_keys:
            f.write(f"{word}\t{freq}\t{key}\n")
    print(f"成功导出掠影原生词典: {len(word_keys)} 条记录 → {out_path}")
