"""把新词增量表喂给 jieba。

默认词典停在 2012,新词不喂就会被碎切:全量扫描的词频、开发集口径、注音
上下文全会偏。频率用 suggest_freq 按当前词典反算,强制整词胜过碎切;只影响
含这些字的这几条确切序列,与词表「不抢高频基础词」的先验不冲突。
"""

import jieba

PATH = "../wordlists/new_words.tsv"


def tune(path: str = PATH) -> None:
    for line in open(path, encoding="utf-8"):
        if line.startswith("#"):
            continue
        word = line.split("\t")[0].strip()
        if word:
            jieba.suggest_freq(word, tune=True)
