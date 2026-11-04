#!/usr/bin/env python3
"""
基于无监督凝固度 (PMI 木桶原理) 与左右边界信息熵的新词发现算法。
用于从 FineWeb-2 / 对话语料中全自动挖掘现代口语与科技热词。
"""

import math
from collections import Counter, defaultdict

def is_cjk(ch: str) -> bool:
    return '\u4e00' <= ch <= '\u9fff'

def clean_clause(text: str):
    """把文本按标点/空格切分为连续的纯汉字子句"""
    buf = []
    for ch in text:
        if is_cjk(ch):
            buf.append(ch)
        else:
            if buf:
                yield "".join(buf)
                buf = []
    if buf:
        yield "".join(buf)

class NewWordMiner:
    def __init__(self, max_word_len=4, min_freq=8, min_cohesion=40.0, min_entropy=0.85):
        self.max_word_len = max_word_len
        self.min_freq = min_freq
        self.min_cohesion = min_cohesion
        self.min_entropy = min_entropy
        
        self.ngram_counts = [Counter() for _ in range(max_word_len + 2)]
        self.left_context = defaultdict(Counter)
        self.right_context = defaultdict(Counter)
        self.total_tokens = 0

    def feed_text(self, text: str):
        for clause in clean_clause(text):
            n = len(clause)
            if n < 2:
                continue
            self.total_tokens += n
            # 统计 1-gram 到 (max_word_len + 1)-gram
            for length in range(1, min(n + 1, self.max_word_len + 2)):
                for i in range(n - length + 1):
                    piece = clause[i:i + length]
                    self.ngram_counts[length][piece] += 1
            
            # 统计 2 到 max_word_len 的左右边界上下文
            for length in range(2, min(n + 1, self.max_word_len + 1)):
                for i in range(n - length + 1):
                    word = clause[i:i + length]
                    if i > 0:
                        self.left_context[word][clause[i - 1]] += 1
                    if i + length < n:
                        self.right_context[word][clause[i + length]] += 1

    def _calc_entropy(self, counter: Counter) -> float:
        total = sum(counter.values())
        if total <= 1:
            return 0.0
        ent = 0.0
        for count in counter.values():
            p = count / total
            ent -= p * math.log(p)
        return ent

    def _calc_cohesion(self, word: str) -> float:
        k = len(word)
        count_w = self.ngram_counts[k].get(word, 0)
        if count_w == 0 or self.total_tokens == 0:
            return 0.0
        pw = count_w / self.total_tokens

        # 木桶原理:取所有切分点下的最小互信息比值
        min_pmi = float("inf")
        for i in range(1, k):
            left = word[:i]
            right = word[i:]
            c_l = self.ngram_counts[len(left)].get(left, 1)
            c_r = self.ngram_counts[len(right)].get(right, 1)
            p_l = c_l / self.total_tokens
            p_r = c_r / self.total_tokens
            pmi_ratio = pw / (p_l * p_r)
            if pmi_ratio < min_pmi:
                min_pmi = pmi_ratio
        return min_pmi

    def extract_words(self):
        """筛选符合凝固度与左右熵阈值的高价值词汇"""
        candidates = []
        for length in range(2, self.max_word_len + 1):
            for word, freq in self.ngram_counts[length].items():
                if freq < self.min_freq:
                    continue
                cohesion = self._calc_cohesion(word)
                if cohesion < self.min_cohesion:
                    continue
                
                left_ent = self._calc_entropy(self.left_context[word])
                right_ent = self._calc_entropy(self.right_context[word])
                entropy = min(left_ent, right_ent)
                if entropy < self.min_entropy:
                    continue
                
                candidates.append((word, freq, cohesion, entropy))

        # 按频次与综合置信度排序
        candidates.sort(key=lambda x: (x[1] * math.log(1.0 + x[2]) * (1.0 + x[3])), reverse=True)
        return candidates
