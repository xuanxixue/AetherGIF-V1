# -*- coding: utf-8 -*-
"""验证 log 域振幅演化的前向/反向可用性（不加载 Qwen，快速冒烟）。"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')
import torch
from physics.block import PhysicsBlock, PMNNCore

N, T_STEP, D = 96, 6, 64
torch.manual_seed(0)
blk = PhysicsBlock(D)
a = torch.exp(torch.randn(2, N, D) * 0.8 - 2.5)
th = torch.rand(2, N, D) * 2 * math.pi - math.pi
om = torch.zeros(2, N, D)
t = torch.arange(N).float() % T_STEP
s = torch.rand(N, 2)

a2, th2, d = blk(a, th, om, t, s,
                 kappa=torch.tensor(1.0), T=torch.tensor(0.5),
                 omega_d=torch.tensor(1.0), targets=None)
print("a_in  mean=%.4f min=%.5f max=%.3f" % (a.mean(), a.min(), a.max()))
print("a_out mean=%.4f min=%.5f max=%.3f" % (a2.mean(), a2.min(), a2.max()))
print("th delta mean=%.4f abs-mean=%.4f" % ((th2 - th).mean(), (th2 - th).abs().mean()))

gs = torch.autograd.grad(a2.sum(), list(blk.parameters()), retain_graph=True, allow_unused=True)
n_grad = sum(1 for g in gs if g is not None and torch.isfinite(g).all())
print("a-grad: %d/%d params finite" % (n_grad, len(list(blk.parameters()))))

gs2 = torch.autograd.grad(th2.sum(), list(blk.parameters()), allow_unused=True)
n_grad2 = sum(1 for g in gs2 if g is not None and torch.isfinite(g).all())
print("th-grad: %d/%d params finite" % (n_grad2, len(list(blk.parameters()))))

# 12 层堆叠数值稳定性
pmnn = PMNNCore(d_model=D, n_blocks=12)
a12, th12 = pmnn(a, th, om, t, s)
print("12-block: a mean=%.4f min=%.5f max=%.3f | th all-finite=%s"
      % (a12.mean(), a12.min(), a12.max(), bool(torch.isfinite(th12).all())))
