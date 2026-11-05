#!/usr/bin/env python3
"""词典全流程上云：在 Kaggle 上跑 扫描 → 选词 → 注音 → 构建，产物取回 --out。

幂等：内核名带标签，已完成的阶段跳过，运行中的接着等，失败的重推。GitHub Actions
单个作业最多 6 小时：等到 --budget 分钟就以退出码 75 结束，工作流据此自己续跑。

静态输入在私有数据集 DATASET：字表与词表、开发集、评测集、readings-seed.pkl、cc1/cc2 统计。
Common Crawl 整体限流时扫不动（S3 SlowDown，换平台也一样），默认用数据集里的统计，
要更新新词时加 --cc 重扫。

用法：python3 cloud/pipeline.py --tag v0.2.0 --out dist
      python3 cloud/pipeline.py --tag smoke1 --smoke --out dist   # 千分之一规模走通流程
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

USER = "ukjent"
DATASET = f"{USER}/lueying-dict-work"
HERE = os.path.dirname(os.path.abspath(__file__))
CONTINUE = 75
# 产物名、来源、规模；cc 开头的名字让 build.py 按网页来源处理。
SCANS = {
    "fineweb2": {"source": "fineweb2", "chars": 200_000_000},
    "lccc": {"source": "lccc", "chars": 200_000_000},
    "ultra": {"source": "ultra", "chars": 100_000_000},
}
FRESH = {
    "cc1": {"source": "commoncrawl", "chars": 300_000_000, "seed": 1},
    "cc2": {"source": "commoncrawl", "chars": 300_000_000, "seed": 2},
}


def kaggle(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


class Kernel:
    def __init__(self, tag: str, stage: str, name: str, args: dict, ref: str,
                 needs: tuple[str, ...] = (), gpu: bool = False):
        self.stage, self.args, self.ref, self.gpu = stage, args, ref, gpu
        self.slug = f"lueying-{tag}-{name}"
        self.id = f"{USER}/{self.slug}"
        self.needs = [f"{USER}/lueying-{tag}-{n}" for n in needs]

    def status(self) -> str:
        result = kaggle("kernels", "status", self.id)
        found = re.search(r"KernelWorkerStatus\.(\w+)", result.stdout)
        if found:
            return found.group(1)
        # 不存在的内核有时报无权访问，有时报 404。
        if re.search(r"Cannot access kernel|404 Client Error", result.stdout + result.stderr):
            return "MISSING"
        # 网络抖动或令牌过期：别把运行中的内核当成失败重推。
        raise RuntimeError(f"{self.slug}: {result.stdout}{result.stderr}")

    def push(self) -> None:
        with open(f"{HERE}/stage.py", encoding="utf-8") as f:
            body = f.read()
        head = f"STAGE = {self.stage!r}\nREF = {self.ref!r}\nARGS = {self.args!r}\n"
        with tempfile.TemporaryDirectory() as folder:
            with open(f"{folder}/kernel.py", "w", encoding="utf-8") as f:
                f.write(head + body)
            with open(f"{folder}/kernel-metadata.json", "w", encoding="utf-8") as f:
                json.dump({"id": self.id, "title": self.slug, "code_file": "kernel.py", "language": "python",
                           "kernel_type": "script", "is_private": True, "enable_gpu": self.gpu,
                           "enable_internet": True, "dataset_sources": [DATASET],
                           "competition_sources": [], "kernel_sources": self.needs}, f)
            extra = ["--accelerator", "NvidiaTeslaT4"] if self.gpu else []
            result = kaggle("kernels", "push", "-p", folder, *extra)
        # 并发名额满时退出码也是 0，推送没成功就留给 run 下一轮重推。
        print(f"[push] {self.slug} {result.stdout.strip()} {result.stderr.strip()}", flush=True)


def run(kernels: list[Kernel], deadline: float) -> None:
    """推送一批互不依赖的内核并等全部完成。"""
    failed: set[str] = set()
    for kernel in kernels:
        if kernel.status() in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCELED"):
            kernel.push()
    waiting = list(kernels)
    while waiting:
        for kernel in list(waiting):
            try:
                state = kernel.status()
            except RuntimeError as error:
                print(f"[wait] {error}", flush=True)
                continue
            if state == "MISSING":
                kernel.push()
            if state in ("RUNNING", "QUEUED", "MISSING"):
                continue
            print(f"[done] {kernel.slug} {state}", flush=True)
            waiting.remove(kernel)
            if state != "COMPLETE":
                failed.add(kernel.slug)
        if waiting and time.time() > deadline:
            print(f"[budget] 还在等 {[k.slug for k in waiting]}", flush=True)
            sys.exit(CONTINUE)
        if waiting:
            time.sleep(60)
    if failed:
        sys.exit(f"失败的内核：{sorted(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sizes", default="200000,400000,600000,800000")
    parser.add_argument("--default", default="600000", help="main.pyd 用哪一档")
    parser.add_argument("--hours", type=float, default=3.0, help="注音时长")
    parser.add_argument("--budget", type=float, default=1e6, help="最多等多少分钟")
    parser.add_argument("--cc", action="store_true", help="重扫 Common Crawl")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    tag = re.sub(r"[^a-z0-9]+", "-", args.tag.lower()).strip("-")
    ref = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                         check=True).stdout.strip()
    scale = 0.001 if args.smoke else 1.0
    deadline = time.time() + args.budget * 60
    sources = {**SCANS, **FRESH} if args.cc else SCANS
    run([Kernel(tag, "scan", name, {**spec, "name": name, "chars": int(spec["chars"] * scale)}, ref)
         for name, spec in sources.items()], deadline)
    scans = tuple(sources)
    run([Kernel(tag, "need", "need", {}, ref, scans)], deadline)
    run([Kernel(tag, "annotate", "annotate", {"hours": 0.05 if args.smoke else args.hours}, ref,
                ("need",), gpu=True)], deadline)
    build = Kernel(tag, "build", "build",
                   {"sizes": "2000,4000" if args.smoke else args.sizes,
                    "default": "4000" if args.smoke else args.default,
                    "chars": int(200_000_000 * scale), "pairs": 10_000 if args.smoke else 1_000_000,
                    "gate": not args.smoke}, ref, (*scans, "annotate"))
    run([build], deadline)
    os.makedirs(args.out, exist_ok=True)
    result = kaggle("kernels", "output", build.id, "-p", args.out, "-o")
    if result.returncode:
        sys.exit(result.stderr)
    print(sorted(os.listdir(args.out)), flush=True)


if __name__ == "__main__":
    main()
