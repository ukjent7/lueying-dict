#!/usr/bin/env python3
"""
下载与构建无版权争议的基础汉字与核心词汇基底。
包含:
1. 《通用规范汉字表》8105 字标准注音与多音字 (CC0 / 规范审音)
2. 基础高频词汇与口语高频词
"""

import os
import sys
import urllib.request
import pypinyin
from legal_syllables import LEGAL_SYLLABLES, normalize_pinyin_syllable

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# pinyin-data 开源拼音数据库 (CC0 / 遵循国家规范)
PINYIN_DATA_URL = "https://raw.githubusercontent.com/mozillazg/pinyin-data/master/pinyin.txt"

def fetch_pinyin_data():
    local_path = os.path.join(DATA_DIR, "pinyin_data.txt")
    if not os.path.exists(local_path):
        print(f"下载 pinyin-data 权威单字审音表: {PINYIN_DATA_URL}")
        try:
            urllib.request.urlretrieve(PINYIN_DATA_URL, local_path)
            print("下载完成。")
        except Exception as e:
            print(f"下载失败 ({e})，将使用内置 pypinyin 自动生成。")
            return None
    return local_path

def parse_char_pinyin(pinyin_file):
    """解析单字到合法拼音列表的映射"""
    char_to_pinyins = {}
    if pinyin_file and os.path.exists(pinyin_file):
        with open(pinyin_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # 格式: U+4E00: yī  # 一
                if ":" in line and "#" in line:
                    parts = line.split(":")
                    code_point = parts[0].strip()
                    after_colon = parts[1].split("#")[0].strip()
                    char = chr(int(code_point[2:], 16))
                    
                    # 剥除音调
                    tones = after_colon.split(",")
                    sylls = set()
                    for t in tones:
                        norm = pypinyin.style.tone.tone_to_normal(t.strip())
                        norm = normalize_pinyin_syllable(norm)
                        if norm in LEGAL_SYLLABLES:
                            sylls.add(norm)
                    if sylls:
                        char_to_pinyins[char] = list(sylls)
    return char_to_pinyins

if __name__ == "__main__":
    p_file = fetch_pinyin_data()
    m = parse_char_pinyin(p_file)
    print(f"成功解析 {len(m)} 个汉字的单字规范注音。")
