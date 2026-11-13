#!/usr/bin/env python
"""Quick GPU throughput check (run alone, with no training on the GPU).

1. Matrix multiplication in fp32, tf32 and bf16 [TFLOPS]: what the GPU and this PyTorch build deliver.
2. ResNet-18 forward + backward at ACT's input (two 480x640 cameras, batch 32 → 64 images) in fp32,
   bf16 autocast and bf16 + channels_last [s/step]: the part of ACT's update that dominates.

    python scripts/act/gpu_bench.py
"""

from __future__ import annotations

import time

import torch


def timed(fn, n: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main() -> None:
    assert torch.cuda.is_available(), "no CUDA device"
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {p.name}, capability {p.major}.{p.minor}, "
          f"{p.multi_processor_count} SMs, arch list {torch.cuda.get_arch_list()}")
    n = 8192
    for name, dtype, tf32 in (("fp32", torch.float32, False), ("tf32", torch.float32, True), ("bf16", torch.bfloat16, False)):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        a, b = torch.randn(n, n, device=dev, dtype=dtype), torch.randn(n, n, device=dev, dtype=dtype)
        s = timed(lambda: a @ b)
        print(f"matmul {name}: {2 * n**3 / s / 1e12:6.1f} TFLOPS")
    torch.backends.cuda.matmul.allow_tf32 = False

    import torchvision

    torch.backends.cudnn.benchmark = True
    x = torch.randn(64, 3, 480, 640, device=dev)
    for name, amp, cl in (("fp32", False, False), ("bf16 autocast", True, False), ("bf16 + channels_last", True, True)):
        net = torchvision.models.resnet18().to(dev)
        xi = x
        if cl:
            net = net.to(memory_format=torch.channels_last)
            xi = x.contiguous(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(net.parameters(), 1e-5)

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = net(xi).float().mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        print(f"resnet18 fwd+bwd, 64 x 480x640, {name}: {timed(step, n=5, warmup=2):.3f} s/step")


if __name__ == "__main__":
    main()
