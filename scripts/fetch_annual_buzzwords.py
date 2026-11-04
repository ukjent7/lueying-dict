#!/usr/bin/env python3
"""
抓取 2024–2026 年度流行语、网络新词与生活高频词库。
来源:
1. zispace/hanyu-annual-buzzwords (国家汉语盘点、咬文嚼字、语言文字周报、B站弹幕、搜狗Top100)
2. skrik2/lexicon (网络流行新词、聊天常用语)
"""

import os
import re
import json
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

ZISPACE_FILES = [
    "汉语盘点-年度流行语.txt",
    "汉语盘点-年度字词.txt",
    "汉语盘点-中国媒体十大流行语.txt",
    "咬文嚼字-年度十大流行语.txt",
    "语言文字周报-年度十大网络流行语.txt",
    "搜狗-年度流行语Top100.txt",
    "B站-年度弹幕.txt",
    "上海市青少年研究中心-年度青年十大热词.txt",
]

ZISPACE_RAW_BASE = "https://raw.githubusercontent.com/zispace/hanyu-annual-buzzwords/main"

SKRIK2_URLS = {
    "skrik2_buzz": "https://raw.githubusercontent.com/skrik2/lexicon/main/%E7%BD%91%E7%BB%9C%E6%B5%81%E8%A1%8C%E6%96%B0%E8%AF%8D.json",
    "skrik2_chat": "https://raw.githubusercontent.com/skrik2/lexicon/main/%E8%81%8A%E5%A4%A9%E5%B8%B8%E7%94%A8%E8%AF%AD.json",
}

def is_all_cjk(text: str) -> bool:
    return len(text) >= 2 and all('\u4e00' <= c <= '\u9fff' for c in text)

def fetch_buzzwords() -> set:
    words = set()

    # 1. zispace 汉语年度榜单
    print("1. 正在抓取 zispace 2024-2026 年度字词榜单...")
    for filename in ZISPACE_FILES:
        url = f"{ZISPACE_RAW_BASE}/{urllib.parse.quote(filename)}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode('utf-8', errors='ignore')
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # 剥除序号、说明等: "1. 显眼包" -> "显眼包"
                    cleaned = re.sub(r'^[0-9\.\、\s\-\*]+', '', line)
                    # 剥除括号补充说明: "拉布布 (Labubu)" -> "拉布布"
                    cleaned = re.sub(r'[\(（\[【].*?[\)）\]】]', '', cleaned).strip()
                    # 分词或逗号分隔
                    tokens = re.split(r'[,，、\s\t]+', cleaned)
                    for tok in tokens:
                        tok = tok.strip()
                        if 2 <= len(tok) <= 8 and is_all_cjk(tok):
                            words.add(tok)
        except Exception as e:
            print(f"  抓取 {filename} 跳过或失败: {e}")

    # 2. skrik2/lexicon 新词与聊天常用语
    print("2. 正在抓取 skrik2/lexicon 网络流行新词与聊天常用语...")
    for name, url in SKRIK2_URLS.items():
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                items = json.loads(resp.read().decode('utf-8', errors='ignore'))
                for item in items:
                    tok = item.get("token", "").strip()
                    if 2 <= len(tok) <= 8 and is_all_cjk(tok):
                        words.add(tok)
        except Exception as e:
            print(f"  抓取 {name} 跳过或失败: {e}")

    out_path = os.path.join(DATA_DIR, "annual_buzzwords.txt")
    sorted_words = sorted(words)
    with open(out_path, "w", encoding="utf-8") as f:
        for w in sorted_words:
            f.write(f"{w}\n")
    print(f"成功汇总 2024-2026 年度流行新词: {len(sorted_words)} 条 → {out_path}")
    return words

if __name__ == "__main__":
    fetch_buzzwords()
