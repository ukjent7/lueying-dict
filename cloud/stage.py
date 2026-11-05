"""Kaggle 内核主体。pipeline.py 在文件开头写入 STAGE、REF 与 ARGS。"""

import glob
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict

OUT = "/kaggle/working"
REPO = "/kaggle/temp/lueying-dict"
CACHE = "/kaggle/temp/cache"
STATS = ("fineweb2", "lccc", "ultra", "cc1", "cc2")


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def find(name: str) -> str:
    """内核产物优先于数据集里的同名文件。"""
    paths = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return sorted(paths, key=lambda path: "lueying-dict-work" in path)[0]


def checkout() -> None:
    sh(f"git init -q {REPO} && cd {REPO} && "
       f"git fetch -q --depth 1 https://github.com/ukjent7/lueying-dict {REF} && git checkout -q FETCH_HEAD")
    os.chdir(f"{REPO}/scripts")
    # Kaggle 上传数据集时会解压 .gz，所以数据集里改了扩展名，这里还原。
    os.makedirs(CACHE, exist_ok=True)
    for source, target in (("8105.txt", "8105.txt"), ("pinyin-data.txt", "pinyin-data.txt"),
                           ("phrase-pinyin.txt", "phrase-pinyin.txt"), ("cedict-gz.bin", "cedict.gz"),
                           ("sampled-gz.bin", "sampled.jsonl.gz")):
        shutil.copy(find(source), f"{CACHE}/{target}")


def corpus_args(readings: str) -> str:
    stats = " ".join(find(f"{name}.pkl") for name in STATS)
    return (f"--stats {stats} --dev {CACHE}/sampled.jsonl.gz --exclude {find('eval-shared-v1.tsv')} "
            f"--readings {readings} --cache {CACHE} "
            f"--new-words ../wordlists/new_words.tsv --colloquial ../wordlists/colloquial.tsv "
            f"--blocked ../wordlists/blocked.txt")


def scan() -> None:
    sh("pip install -q jieba zhconv datasets")
    checkout()
    seed = f"--seed {ARGS['seed']}" if "seed" in ARGS else ""
    sh(f"python scan.py --source {ARGS['source']} --max-chars {ARGS['chars']} {seed} --workers 4 "
       f"--out {OUT}/{ARGS['name']}.pkl")


def need() -> None:
    sh("pip install -q jieba zhconv")
    checkout()
    sh(f"python build.py {corpus_args(find('readings-seed.pkl'))} --need {OUT}/need.txt --out {OUT}/tmp")
    shutil.rmtree(f"{OUT}/tmp", ignore_errors=True)


def annotate() -> None:
    sh("pip install -q g2pw jieba zhconv datasets")
    # g2pw 依赖 CPU 版 onnxruntime，它和 GPU 版共用包目录，必须清掉重装。
    sh("pip uninstall -y -q onnxruntime onnxruntime-gpu || true")
    sh("pip install -q --force-reinstall --no-deps onnxruntime-gpu")
    sh("pip install -q nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 "
       "nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12")
    checkout()
    hours = ARGS["hours"]
    words = find("need.txt")
    # 两卡共用一份模型缓存会读到半截文件，先让一张卡把模型下载完。
    sh("python -c \"from g2pw import G2PWConverter; G2PWConverter(style='pinyin')\"")
    procs = [subprocess.Popen([sys.executable, "annotate.py", "--need", words, "--hours", str(hours),
                               "--shard", f"{i}/2", "--out", f"{OUT}/part{i}.pkl"],
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))) for i in (0, 1)]
    # 语料流卡住时不空等：检查点每 10 分钟落盘一次，合并现有结果即可。
    end = time.time() + hours * 3600 + 1800
    for proc in procs:
        try:
            proc.wait(timeout=max(1, end - time.time()))
        except subprocess.TimeoutExpired:
            proc.terminate()
    chars, grams, seen = defaultdict(Counter), defaultdict(Counter), 0
    for path in glob.glob(f"{OUT}/part*.pkl"):
        with open(path, "rb") as f:
            part = pickle.load(f)
        seen += part["seen"]
        for target, source in ((chars, part["chars"]), (grams, part["grams"])):
            for key, value in source.items():
                target[key].update(value)
        os.remove(path)
    if not seen:
        raise SystemExit("两张卡都没有产出")
    with open(f"{OUT}/readings.pkl", "wb") as f:
        pickle.dump({"chars": dict(chars), "grams": dict(grams), "seen": seen}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print("merged", seen, len(grams), flush=True)


def build() -> None:
    sh("pip install -q jieba zhconv datasets")
    checkout()
    sh(f"python build.py {corpus_args(find('readings.pkl'))} --sizes {ARGS['sizes']} --out {OUT}")
    tier = f"{int(ARGS['default']) // 1000}k"
    gate = "" if ARGS["gate"] else " || true"
    sh(f"python evaluate.py {OUT}/main-{tier}.tsv --eval-dir {os.path.dirname(find('eval-words.tsv'))} "
       f"--context {CACHE}/sampled.jsonl.gz --new-words ../wordlists/new_words.tsv{gate}")
    limit = ARGS["pairs"]
    sh(f"python collocate.py {OUT}/main-{tier}.tsv --max-chars {ARGS['chars']} --limits {limit} "
       f"--clips 6 --workers 4 --out {OUT}/colloc")
    shutil.move(f"{OUT}/colloc/collocation-{limit // 1000}k-c6.bin", f"{OUT}/collocation.bin")
    shutil.rmtree(f"{OUT}/colloc")
    shutil.copy(f"{OUT}/main-{tier}.pyd", f"{OUT}/main.pyd")
    for path in glob.glob(f"{OUT}/main-*.tsv"):
        os.remove(path)


{"scan": scan, "need": need, "annotate": annotate, "build": build}[STAGE]()
