# -*- coding: utf-8 -*-
"""冒烟测试：逐层定位 PMNNCore 的 einsum 维度问题。"""
import sys, traceback, time
sys.path.insert(0, "src")
import torch
from physics.block import PMNNCore, PhysicsBlock
from physics.kuramoto_sync import build_spacetime_index, KuramotoSync, SpacetimeToeplitzCoupling, MeanFieldCoupling
from physics.standing_wave_ffn import StandingWaveFFN
from physics.drift_field import DriftField

T, H, W = 6, 32, 32
dev = "cuda"
tc, sc = build_spacetime_index(T, H, W, dev)
N = T * H * W
a0 = torch.nn.functional.softplus(torch.randn(1, N, 64, device=dev))
th0 = torch.rand(1, N, 64, device=dev) * 6.283 - 3.1415
om0 = torch.randn(1, N, 64, device=dev) * 0.5

def step(name, fn):
    try:
        r = fn()
        print(f"[PASS] {name}  -> {tuple(r.shape) if torch.is_tensor(r) else r}")
        return r
    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        sys.exit(1)

# 1. Toeplitz
step("Toeplitz", lambda: SpacetimeToeplitzCoupling().to(dev).apply_coupling(th0, tc, sc))
# 2. MeanField
step("MeanField", lambda: MeanFieldCoupling.forward(a0, th0)[0])
# 3. KuramotoSync
k = KuramotoSync().to(dev)
step("KuramotoSync", lambda: k(th0, a0, om0, torch.zeros_like(om0),
                               torch.tensor(2.0, device=dev), torch.tensor(0.5, device=dev), tc, sc)[0])
# 4. StandingWaveFFN
step("SWFFN", lambda: StandingWaveFFN(64, 256).to(dev)(a0, th0, torch.tensor(1.0, device=dev), tc, sc))
# 5. DriftField
step("DriftField", lambda: DriftField(64, 16).to(dev)(a0, th0, None))
# 6. 单个 PhysicsBlock
blk = PhysicsBlock(64).to(dev)
step("PhysicsBlock", lambda: blk(a0, th0, om0, tc, sc,
                                  kappa=torch.tensor(2.0, device=dev),
                                  T=torch.tensor(0.5, device=dev),
                                  omega_d=torch.tensor(1.0, device=dev))[0])
# 7. 完整 PMNNCore
core = PMNNCore(64, 12).to(dev)
print(f"[INFO] PMNNCore params: {sum(p.numel() for p in core.parameters())}")

def full_core():
    t0 = time.time()
    a2, th2 = core(a0, th0, om0, tc, sc)
    (a2.mean() - th2.mean()).abs().backward()
    torch.cuda.synchronize()
    print(f"  fwd+bwd {time.time()-t0:.2f}s")
    print(f"  peak VRAM {torch.cuda.max_memory_allocated()/1024/1024:.0f} MB")
    return a2

step("PMNNCore fwd+bwd", full_core)
print("[ALL PASS]")
