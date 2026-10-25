# -*- coding: utf-8 -*-
"""验证 DriftField 分块 softmax 核与全物化严格等价，并测训练口径显存。

物理公式一个字不改，只是把 O(N*M) 的全距离矩阵换成分块两遍 + 梯度检查点。
若这里过不了，说明分块合并写错了，不能进训练。
"""
import sys

sys.path.insert(0, "src")
import torch
from physics.drift_field import DriftField, softmax_kernel_field

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)


def ref_kernel(x, y, tau, self_mask):
    """全物化参考实现（同一代数展开，不分块）。"""
    x, y = x.float(), y.float()
    xs = (x ** 2).sum(-1)
    ys = (y ** 2).sum(-1)
    cross = torch.einsum("bnc,bmc->bnm", x, y)
    l = -(xs.unsqueeze(2) + ys.unsqueeze(1) - 2.0 * cross) / tau
    if self_mask:
        N = x.shape[1]
        eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
        l = l.masked_fill(eye, -1e4)
    w = torch.softmax(l, dim=-1)
    acc = torch.einsum("bnm,bmc->bnc", w, y)
    return acc - x * w.sum(-1, keepdim=True)


print("\n=== 1) 分块 vs 全物化（前向数值等价）===")
B, N, C = 2, 768, 128
x = torch.randn(B, N, C, device=dev) * 0.5
y = torch.randn(B, N, C, device=dev) * 0.5
tau = 1.0
with torch.no_grad():
    ref_attract = ref_kernel(x, y, tau, False)
    ref_repel = ref_kernel(x, x, tau, True)
    scale = ref_repel.abs().max().item() + 1e-12
    for chunk in [64, 256, 1024, 4096]:
        o_a = softmax_kernel_field(x, y, tau, False, chunk)
        o_r = softmax_kernel_field(x, x, tau, True, chunk)
        ea = (o_a - ref_attract).abs().max().item()
        er = (o_r - ref_repel).abs().max().item()
        rel = max(ea, er) / scale
        print(f"  chunk={chunk:5d}  attract={ea:.3e}  repel={er:.3e}  rel={rel:.3e}")
        assert rel < 1e-4, f"chunk={chunk} 数值不等价"
print("[PASS] 前向数值等价 (rel < 1e-4)")

print("\n=== 2) 反向梯度等价 ===")
for chunk in [256, 1024]:
    xa = (torch.randn(B, N, C, device=dev) * 0.5).requires_grad_(True)
    out = softmax_kernel_field(xa, xa, tau, True, chunk)
    out.sum().backward()
    g_chunk = xa.grad.clone()

    xb = xa.detach().clone().requires_grad_(True)
    ref = ref_kernel(xb, xb, tau, True)
    ref.sum().backward()
    err = (g_chunk - xb.grad).abs().max().item()
    print(f"  chunk={chunk:5d}  grad max diff = {err:.3e}")
    assert err < 1e-3, f"chunk={chunk} 梯度不等价"
print("[PASS] 反向梯度等价")

print("\n=== 3) 训练口径显存（DriftField fwd+bwd @ N=10240）===")
N = 10240
df = DriftField(d_model=64, n_prototypes=16, chunk=2048).to(dev)
a = torch.nn.functional.softplus(torch.randn(1, N, 64, device=dev)).requires_grad_(True)
th = (torch.rand(1, N, 64, device=dev) * 6.283 - 3.1415).requires_grad_(True)
targets = torch.randn(1, N, 128, device=dev) * 0.5  # 训练时 M = N
df.train()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
g = df(a, th, targets)
g.sum().backward()
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / 1024 ** 2
print(f"  N={N} 训练（含反向，checkpoint）峰值显存 = {peak:.0f} MB")
assert peak < 3500, f"峰值 {peak:.0f} MB 仍太高"
print("[PASS] 未 OOM")
print("\n[ALL PASS]")
