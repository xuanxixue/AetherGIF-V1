# -*- coding: utf-8 -*-
"""第二十八死穴探针：θ0 随机 + dθ 看不到 θ0 ⇒ 天花板量化

模型：θ_final = θ0 + dθ，θ0 ~ U(−π,π)，dθ = f(cond) 与 θ0 独立。
问：|dθ| 给到多大，cos(θ_final, θ_tar) 的上限是多少？
"""
import math, torch

torch.manual_seed(0)
N = 200000
th0 = torch.rand(N) * 2 * math.pi - math.pi
thtar = torch.rand(N) * 2 * math.pi - math.pi

def cosv(a, b):
    return torch.cos(a - b)

print("=== 上限：dθ 恰好取 '理想修正' 的一个子集（其余被 bound 截断）===")
# 理想修正量（把 th0 搬到 thtar）
ideal = torch.remainder(thtar - th0 + math.pi, 2*math.pi) - math.pi
print(f"  理想 |dθ|: mean={ideal.abs().mean():.4f}  p50={ideal.abs().median():.4f} max={ideal.abs().max():.4f}")

for bound in [math.pi/4, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]:
    d = ideal.clamp(-bound, bound)
    c = cosv(th0 + d, thtar).mean().item()
    print(f"  bound={bound:.3f}  理想截断后 cos={c:+.4f}")

print()
print("=== 现实：dθ 与 θ0 独立（dθ 是条件的函数，不是 θ0 的函数）===")
print("  dθ 只能是一个固定形状；随机 θ0 的分量无法被消去")
for K in [1, 5, 20]:
    # dθ = K 个模态叠加，模拟 per-band 结构
    for amp in [math.pi/2, math.pi, 2*math.pi]:
        d = amp * torch.sin(K * thtar + 0.7)  # 理想形状但幅值受限
        d = d.clamp(-amp, amp)
        c = cosv(th0 + d, thtar).mean().item()
        print(f"  K={K:2d} amp={amp:.3f}  cos={c:+.4f}")

print()
print("=== 对照：θ0 不随机（θ0 = dθ 直接给相位）===")
for K in [1, 5, 20]:
    d = torch.sin(K * thtar + 0.7)
    print(f"  K={K:2d}  cos(θ_final,θ_tar)={cosv(d, thtar).mean().item():+.4f}")
