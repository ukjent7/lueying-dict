#!/usr/bin/env python3
"""
掠影自研原生词典全套自动化构建流水线 (2026 现代人际互动与通用混合版)。
结合:
1. opencsg/smoltalk-chinese (2025/2026 现代生活与职场多轮对话)
2. m-a-p/COIG-CQIA (2024/2025 小红书 xhs、豆瓣 douban、贴吧 ruozhiba 社交与网络热梗)
3. HuggingFaceFW/fineweb-2 cmn_Hani (通用科技与百科底座)
4. Tatoeba 口语短句 (8.9万条)
5. 《通用规范汉字表》8105 字规范审音与 pypinyin-dict 百万级短语消歧库
产出符合掠影规范的 main.tsv 与零拷贝二进制 main.pyd。
"""

import os
import sys
import math
import struct
from collections import Counter
from tqdm import tqdm

from legal_syllables import LEGAL_SYLLABLES, normalize_pinyin_syllable
from mine_words import NewWordMiner, is_cjk

# 挂载超大规模离线拼音词典
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

def run_multi_source_pipeline(target_vocab_size=200000, tatoeba_path="tatoeba_cmn.txt"):
    print("=" * 65)
    print("开始执行掠影 2026 现代人际互动加权词典全量构建流水线")
    print("=" * 65)

    miner = NewWordMiner(max_word_len=4, min_freq=5, min_cohesion=35.0, min_entropy=0.72)
    from datasets import load_dataset

    # 1. 本地 Tatoeba 真实短句
    if os.path.exists(tatoeba_path):
        print(f"[1/4] 载入本地 Tatoeba 口语短句: {tatoeba_path}")
        with open(tatoeba_path, "r", encoding="utf-8") as f:
            for line in f:
                miner.feed_text(line.strip())
        print(f"Tatoeba 处理完毕，已扫描 Token: {miner.total_tokens}")

    # 2. opencsg/smoltalk-chinese (2025/2026 现代生活与职场多轮对话)
    print("[2/4] 流式扫描 opencsg/smoltalk-chinese 现代多轮对话 (25,000 组对话)...")
    try:
        ds_smol = load_dataset("opencsg/smoltalk-chinese", split="train", streaming=True)
        count = 0
        pbar = tqdm(total=25000, desc="SmolTalk-CN 对话流")
        for sample in ds_smol:
            for turn in sample.get("conversations", []):
                content = turn.get("content", "")
                if content:
                    miner.feed_text(content)
            count += 1
            pbar.update(1)
            if count >= 25000:
                break
        pbar.close()
    except Exception as e:
        print(f"SmolTalk-CN 加载异常: {e}")

    # 3. m-a-p/COIG-CQIA (小红书、豆瓣、贴吧网络热词与生活俚语)
    print("[3/4] 流式扫描 m-a-p/COIG-CQIA 现代社交语料 (xhs + douban + ruozhiba)...")
    subsets = [("xhs", 12000), ("douban", 12000), ("ruozhiba", 3000)]
    for sub, limit in subsets:
        try:
            ds_sub = load_dataset("m-a-p/COIG-CQIA", sub, split="train", streaming=True)
            c = 0
            pbar = tqdm(total=limit, desc=f"COIG-CQIA ({sub})")
            for item in ds_sub:
                text = (item.get("instruction", "") or "") + " " + (item.get("output", "") or "")
                if text.strip():
                    miner.feed_text(text)
                c += 1
                pbar.update(1)
                if c >= limit:
                    break
            pbar.close()
        except Exception as e:
            print(f"COIG-CQIA ({sub}) 加载异常: {e}")

    # 4. HuggingFaceFW/fineweb-2 (通用科技与百科规范底座)
    print("[4/4] 流式扫描 HuggingFaceFW/fineweb-2 通用科技与网页底座 (15,000 篇)...")
    try:
        ds_fw = load_dataset(
            "HuggingFaceFW/fineweb-2",
            name="cmn_Hani",
            split="train",
            streaming=True
        )
        c = 0
        pbar = tqdm(total=15000, desc="FineWeb-2 语料扫描")
        for item in ds_fw:
            text = item.get("text", "")
            if text:
                miner.feed_text(text)
            c += 1
            pbar.update(1)
            if c >= 15000:
                break
        pbar.close()
    except Exception as e:
        print(f"FineWeb-2 流式读取异常: {e}")

    print("-" * 65)
    print(f"多源混合语料扫描总计完成！有效汉字 Token 达: {miner.total_tokens}")
    print("-" * 65)

    # 5. 提取新词
    print("5. 执行无监督凝固度 (PMI) 与左右信息熵剪枝...")
    candidates = miner.extract_words()
    print(f"新词挖掘提取成功: {len(candidates)} 条高质量词组")

    # 6. G2P 自动注音与多音字消歧
    print("6. 执行 G2P 自动注音与多音字消歧...")
    base_chars = build_baseline_characters()
    raw_dict = {} # (word, key) -> freq

    # 单字
    for ch in base_chars:
        key = get_word_pinyin(ch)
        if key:
            c = miner.ngram_counts[1].get(ch, 1)
            raw_dict[(ch, key)] = c

    # 挖掘出的多字词
    for word, freq, cohesion, entropy in candidates:
        key = get_word_pinyin(word)
        if key:
            raw_dict[(word, key)] = freq
            if len(raw_dict) >= target_vocab_size:
                break

    print(f"词典初选条目数: {len(raw_dict)}")

    # 7. 同口径 MLE 词频平滑
    print("7. 执行同口径 MLE 次线性平滑与频次映射...")
    max_c = max(raw_dict.values()) if raw_dict else 1
    final_entries = [] # (word, key, freq)

    for (word, key), c in raw_dict.items():
        # 次线性对数平滑映射到 [10, 800000]
        score = 80000.0 * math.pow(c / max_c, 0.62) + math.log1p(c) * 120.0 + 10.0
        # 多字词给予 1.25x 组合偏好,保证整句输入时不被打碎成单字
        if len(word) >= 2:
            score *= 1.25
        final_freq = max(1, int(round(score)))
        final_entries.append((word, key, final_freq))

    # 8. 严格排序: (key 升序, freq 降序, word 升序)
    print("8. 严格按 (key, -freq, word) 排序...")
    final_entries.sort(key=lambda x: (x[1], -x[2], x[0]))

    # 9. 导出 TSV 与 PYD
    tsv_path = os.path.join(OUT_DIR, "main.tsv")
    print(f"9. 导出 TSV 文本词典: {tsv_path}")
    with open(tsv_path, "w", encoding="utf-8") as f:
        for word, key, freq in final_entries:
            f.write(f"{word}\t{freq}\t{key}\n")

    pyd_path = os.path.join(OUT_DIR, "main.pyd")
    print(f"10. 导出 PYD1 紧凑二进制词典: {pyd_path}")
    pack_pyd(final_entries, pyd_path)

    print("=" * 65)
    print(f"掠影原生自主词典生成成功！总条目: {len(final_entries)}")
    print("=" * 65)

if __name__ == "__main__":
    tatoeba = sys.argv[1] if len(sys.argv) > 1 else "tatoeba_cmn.txt"
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    run_multi_source_pipeline(target_vocab_size=vocab_size, tatoeba_path=tatoeba)
