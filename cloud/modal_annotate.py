"""spike1 词典重建的 Modal 移植：annotate(2×T4 GPU) + build(大内存 CPU)。

语义与 cloud/stage.py 完全一致，改动只在外层编排：
- annotate：两个 annotate.py 子进程按 crc32 分片各取一半句子，各占一张 T4
  （CUDA_VISIBLE_DEVICES 分卡，与 Kaggle 同构）；语料流（ultra+lccc 7:3）、
  --enough/--hours/--max-chars 语义、part*.pkl 合并（Counter 相加、seen 求和）
  逐行照抄 stage.py，输出 readings.pkl 进 Volume。
- build：build.py/evaluate.py/collocate.py 用同一批参数与静态输入；stats pkl、
  cache 文件、readings.pkl 全部走 Volume，产物写回 /vol/dist 再取回本地。

Volume 布局（lueying-dict）：
    need.txt                          need 内核产物
    readings.pkl                      annotate 合并产物
    stats/{fineweb2,lccc,ultra,cc1,cc2}.pkl
    cache/{8105.txt,pinyin-data.txt,phrase-pinyin.txt,cedict.gz,
           sampled.jsonl.gz,eval-shared-v1.tsv,eval-words.tsv}
    dist/                             build 产物

用法：
    modal run --detach cloud/modal_annotate.py                 # 全量 annotate（3h）
    modal run cloud/modal_annotate.py --hours 0.1 \
        --max-chars 20000000 --prefix smoke-                   # 小冒烟
    modal run --detach cloud/modal_annotate.py::build          # 构建
"""

import os
import subprocess
import sys
import time

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOLUME_NAME = "lueying-dict"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("lueying-dict")

# 与 stage.py annotate() 同一套 pip：g2pw 拖来 CPU 版 onnxruntime，必须卸载后
# --force-reinstall --no-deps 换成 GPU 版，再补 cu12 用户态库。
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    # torch 2.9 的 PyPI 默认轮是 cu128：拖来的正好是与 stage.py 相同的 cu12 库。
    # g2pw 只把 torch 当 DataLoader 用，CUDA 推理全走 onnxruntime。
    .pip_install("g2pw>=0.1", "jieba>=0.42", "zhconv>=1.4", "datasets>=4.0",
                 "torch==2.9.*")
    .run_commands(
        "pip uninstall -y -q onnxruntime onnxruntime-gpu || true",
        # ORT 1.32 起改要 CUDA 13；锁回与 cu12 库配套的 1.23。
        "pip install -q --force-reinstall --no-deps onnxruntime-gpu==1.23.*",
    )
    .pip_install(
        "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12",
        "nvidia-cufft-cu12", "nvidia-curand-cu12", "nvidia-cuda-nvrtc-cu12",
    )
    .add_local_dir(f"{REPO}/scripts", "/repo/scripts", ignore=["__pycache__"])
    .add_local_dir(f"{REPO}/wordlists", "/repo/wordlists")
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .pip_install("jieba>=0.42", "zhconv>=1.4", "datasets>=4.0")
    .add_local_dir(f"{REPO}/scripts", "/repo/scripts", ignore=["__pycache__"])
    .add_local_dir(f"{REPO}/wordlists", "/repo/wordlists")
)

SEEDS = (
    "https://raw.githubusercontent.com/amzxyz/rime-wanxiang/"
    "b41495a9d257920eca6d5f50ef2fb693d6d6f779/dicts/jichu.dict.yaml",
    "https://raw.githubusercontent.com/gaboolic/rime-frost/"
    "3ba4f1df224823079b947ce1553441c36097b124/cn_dicts/base.dict.yaml",
    "https://raw.githubusercontent.com/gaboolic/rime-frost/"
    "3ba4f1df224823079b947ce1553441c36097b124/cn_dicts/ext.dict.yaml",
    "https://raw.githubusercontent.com/gaboolic/rime-frost/"
    "3ba4f1df224823079b947ce1553441c36097b124/cn_dicts/tencent.dict.yaml",
)


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


@app.function(image=gpu_image, gpu="T4:2", cpu=4.0, memory=16384,
              timeout=5 * 3600, volumes={"/vol": volume})
def annotate(hours: float = 3.0, max_chars: int = 2_000_000_000,
             prefix: str = "") -> None:
    """stage.py annotate()：双分片各吃一半句子，各占一张 T4（与 Kaggle 一致）。"""
    import glob
    import pickle
    from collections import Counter, defaultdict

    os.chdir("/repo/scripts")
    # 两个进程各自的 G2PWModel 缓存先由单进程建好，避免并发下载读到半截文件。
    # g2pw 自己的 download_model 把 zip 解到 G2PWModel/ 再 move 到同名目录必然
    # ENOTEMPTY（zip 顶层目录变了），这里直接解到目标位置，效果与成功路径相同。
    sh("python -c \""
       "import os, requests, zipfile; "
       "from io import BytesIO; "
       "z = zipfile.ZipFile(BytesIO(requests.get("
       "'https://huggingface.co/YC-Chen/g2pW/resolve/main/G2PWModel-v2-onnx.zip').content)); "
       "z.extractall('.')\"")
    sh("python -c \"from g2pw import G2PWConverter; G2PWConverter(style='pinyin')\"")
    # get_available_providers 会列出加载失败的 provider；必须验证会话实际用上了 CUDA。
    sh("python -c \"import onnxruntime; onnxruntime.preload_dlls(); "
       "s = onnxruntime.InferenceSession('G2PWModel/g2pw.onnx', "
       "providers=['CUDAExecutionProvider']); "
       "assert s.get_providers()[0] == 'CUDAExecutionProvider', s.get_providers(); "
       "print('cuda ok')\"")
    procs = [
        subprocess.Popen(
            [sys.executable, "annotate.py", "--need", "/vol/need.txt",
             "--hours", str(hours), "--max-chars", str(max_chars),
             "--shard", f"{i}/2", "--out", f"/vol/{prefix}part{i}.pkl"],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)))
        for i in (0, 1)
    ]
    # 语料流卡住时不空等；分片进程自己每 10 分钟落盘，这里只催 Volume 落库。
    end = time.time() + hours * 3600 + 1800
    while time.time() < end:
        if all(proc.poll() is not None for proc in procs):
            break
        volume.commit()
        time.sleep(300)
    for proc in procs:
        try:
            proc.wait(timeout=max(1, end - time.time()))
        except subprocess.TimeoutExpired:
            proc.terminate()
    volume.commit()
    # 合并逻辑与 stage.py 相同。
    chars, grams, seen = defaultdict(Counter), defaultdict(Counter), 0
    for path in glob.glob(f"/vol/{prefix}part*.pkl"):
        with open(path, "rb") as f:
            part = pickle.load(f)
        seen += part["seen"]
        for target, source in ((chars, part["chars"]), (grams, part["grams"])):
            for key, value in source.items():
                target[key].update(value)
        os.remove(path)
    if not seen:
        raise RuntimeError("两个分片都没有产出")
    with open(f"/vol/{prefix}readings.pkl", "wb") as f:
        pickle.dump({"chars": dict(chars), "grams": dict(grams), "seen": seen}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    volume.commit()
    print("merged", seen, len(grams), flush=True)


@app.function(image=cpu_image, cpu=8.0, memory=65536, timeout=8 * 3600,
              volumes={"/vol": volume})
def build() -> None:
    """stage.py build()：参数全部与 spike1 构建一致，输入改走 Volume。"""
    import glob
    import shutil

    os.chdir("/repo/scripts")
    out = "/work/out"
    os.makedirs("/work/seeds", exist_ok=True)
    os.makedirs(out, exist_ok=True)
    seeds = []
    for i, url in enumerate(SEEDS):
        seeds.append(f"/work/seeds/seed{i}.yaml")
        sh(f"curl -fsSL --retry 3 -o {seeds[-1]} {url}")
    stats = " ".join(f"/vol/stats/{name}.pkl"
                     for name in ("fineweb2", "lccc", "ultra", "cc1", "cc2"))
    corpus_args = (
        f"--stats {stats} --dev /vol/cache/sampled.jsonl.gz "
        f"--exclude /vol/cache/eval-shared-v1.tsv --readings /vol/readings.pkl "
        f"--cache /vol/cache --new-words ../wordlists/new_words.tsv "
        f"--colloquial ../wordlists/colloquial.tsv --blocked ../wordlists/blocked.txt "
        f"--spike-scheme pow --colloquial-freq ../wordlists/colloquial_freq.tsv "
        f"--lex-floor 300 --trusted-redundant 0.3"
    )
    sh(f"python build.py {corpus_args} --seeds {' '.join(seeds)} "
       f"--sizes 200000,400000,600000,800000 --out {out}")
    # gate=True：评测不挂 || true，门槛不过直接失败。
    sh(f"python evaluate.py {out}/main-600k.tsv --eval-dir /vol/cache "
       f"--context /vol/cache/sampled.jsonl.gz --new-words ../wordlists/new_words.tsv")
    sh(f"python collocate.py {out}/main-600k.tsv --max-chars 200000000 "
       f"--limits 1000000 --clips 6 --workers 4 --out /work/colloc")
    os.makedirs("/vol/dist", exist_ok=True)
    shutil.move("/work/colloc/collocation-1000k-c6.bin", "/vol/dist/collocation.bin")
    shutil.copy(f"{out}/main-600k.pyd", "/vol/dist/main.pyd")
    shutil.copy(f"{out}/build_report.json", "/vol/dist/build_report.json")
    for path in glob.glob(f"{out}/main-*.*"):
        shutil.copy(path, "/vol/dist/")
    volume.commit()
    print(sorted(os.listdir("/vol/dist")), flush=True)


@app.local_entrypoint()
def main(hours: float = 3.0, max_chars: int = 2_000_000_000,
         prefix: str = "") -> None:
    annotate.remote(hours=hours, max_chars=max_chars, prefix=prefix)
