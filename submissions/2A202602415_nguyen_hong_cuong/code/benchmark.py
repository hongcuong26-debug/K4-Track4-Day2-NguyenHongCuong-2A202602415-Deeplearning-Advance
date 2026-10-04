"""Latency measurement with warmup, synchronization, and tail percentiles."""
from __future__ import annotations

import contextlib
import platform
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Measure a nullary callable in milliseconds."""
    if warmup < 10:
        raise ValueError("warmup must be at least 10")
    if iters < 50:
        raise ValueError("iters must be at least 50")
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()
    samples = []
    for _ in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(samples, dtype=float)
    return {"p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "mean": float(values.mean()), "n": int(iters)}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Benchmark model-only forward latency on a synthetic input."""
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype must be fp32, amp, or fp16")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    measured_model = model.to(target).eval()
    images = torch.randn(batch_size, 3, img_size, img_size, device=target)
    if dtype == "fp16":
        if target.type != "cuda":
            raise ValueError("fp16 benchmark is supported on CUDA only")
        measured_model = measured_model.half()
        images = images.half()
    sync = torch.cuda.synchronize if target.type == "cuda" else None
    amp_context = (lambda: torch.autocast(device_type=target.type, dtype=torch.float16)) \
        if dtype == "amp" and target.type == "cuda" else contextlib.nullcontext

    def forward():
        with torch.inference_mode(), amp_context():
            measured_model(images)

    result = bench(forward, warmup=warmup, iters=iters, sync=sync)
    result.update({
        "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else platform.processor() or "CPU",
        "dtype": dtype, "batch": int(batch_size), "img_size": int(img_size),
        "images_per_s": float(batch_size / (result["p50"] / 1000.0)),
        "torch": torch.__version__, "includes_preprocessing": False,
    })
    return result


def tta_latency(model, k_views: int, **kw) -> dict:
    """Measure K actual forwards and compare them with K times the one-view median."""
    if k_views < 1:
        raise ValueError("k_views must be positive")
    one = latency_report(model, **kw)
    target = torch.device(kw.get("device", "cuda"))
    batch_size, img_size = kw["batch_size"], kw["img_size"]
    dtype = kw.get("dtype", "fp32")
    measured_model = model.to(target).eval()
    images = torch.randn(batch_size, 3, img_size, img_size, device=target)
    if dtype == "fp16":
        measured_model, images = measured_model.half(), images.half()
    sync = torch.cuda.synchronize if target.type == "cuda" else None
    amp_context = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) \
        if dtype == "amp" and target.type == "cuda" else contextlib.nullcontext

    def forward_k():
        with torch.inference_mode(), amp_context():
            for _ in range(k_views):
                measured_model(images)

    actual = bench(forward_k, warmup=kw.get("warmup", 10), iters=kw.get("iters", 100), sync=sync)
    actual.update({"k_views": int(k_views), "one_view_p50": one["p50"],
                   "expected_k_x_p50": float(k_views * one["p50"]),
                   "ratio_to_one_view": float(actual["p50"] / one["p50"])})
    return actual
