# 掠影词典 (Lueying Dict)

掠影输入法的自研词库。词条和词频来自语料统计，读音来自真实句子上的上下文注音，
构建不读取任何第三方输入法词库。

## 流水线

| 阶段 | 脚本 | 产物 |
|---|---|---|
| 语料统计 | `scripts/scan.py` | 每个来源一份词次与文档频次 |
| 上下文注音 | `scripts/annotate.py` | 词的实际读音分布（g2pW，需要 GPU） |
| 选词打包 | `scripts/build.py` | 多档 `main-<档位>.tsv` / `.pyd` |
| 门禁 | `scripts/evaluate.py` | 词形、读音、可达性检查 |

各阶段通过中间文件衔接，换掉一个阶段（新的注音模型、新的语料）不影响其余部分。
语料阶段和注音阶段在 Colab 上用 `colab` CLI 跑。

## 选词

语料分四类：网页基础层（Ultra-FineWeb、FineWeb-2）、对话（LCCC）、2025–2026 新文本
（Common Crawl）。混合权重用 EM 在开发集上学，不手工指定；新词随新语料进入词频。

所有档位由同一条效用排序截断：混合概率 × 词表可信度 × 解码必要性。解码必要性来自
拿掉这个词后的整句解码，同一拼音仍能解出原词就说明它可以由更短的词拼出，收录它
只增加体积。

## 构建

```bash
python3 scripts/scan.py --source ultra --max-chars 600000000 --out stats/ultra.pkl
python3 scripts/annotate.py --max-chars 80000000 --out stats/readings.pkl
python3 scripts/build.py --stats stats/*.pkl --readings stats/readings.pkl --cache cache \
  --dev sampled.jsonl.gz --exclude eval-shared-v1.tsv
python3 scripts/evaluate.py output/main-400k.tsv
```

`cache/` 放 `8105.txt`、`pinyin-data.txt`、`phrase-pinyin.txt`、`cedict.gz`。

## 新词

`wordlists/new_words.tsv` 只收公开榜单上的新词，不收评测集里的词；门禁会检查两者
没有交集。

## 许可

词典以 GPL-3.0-or-later 随掠影分发，数据来源见 [NOTICE](NOTICE)。
