# -*- coding: utf-8 -*-
"""loss_sep 符号 bug 数值验证（纯张量，不加载模型）

supervise 返回 = loss_amp + loss_ph + guard_w*guard + sep_w*loss_sep
  loss_amp ∈ [0,2]，loss_ph ∈ [0,2]，guard ≥ 0，sep_w = 1.0（默认）
日志 gen = -9.77 ⇒ 负值只能来自 loss_sep。

当前实现（cond_init.py 两处）：
    loss_sep = (log(spread_out) - log(spread_in)).clamp(max=0.0)
  spread_out = 组间散布（各频段均值之间的差异）
  spread_in  = 组内散布（每频段跨位置的方差）

设计意图（行 457 注释）：「ratio<1 时才罚」——
  ratio = spread_out/spread_in，组间散布 < 组内散布 时应当**惩罚**（正值）。
但 clamp(max=0) 把它变成了**负奖励**：
  ratio<1 时 loss_sep<0（总 loss 下降），模型被奖励维持组间无差异。

验证三种情形的 loss_sep 值与「正确形式」对比。
正确形式 = (log(spread_in) - log(spread_out)).clamp(min=0) = relu(-log(ratio))
  ratio>=1 → 0（组间分化达标，不罚）
  ratio<1  → 正罚（组间无差异，推模型去分化）
"""
import torch

def spread(pd_g):
    """pd_g: [B,N,G] 每频段相位场，返回 (spread_in, spread_out)。"""
    mu = pd_g.mean(dim=1, keepdim=True)          # [B,1,G] 跨位置均值
    spread_in = (pd_g - mu).pow(2).mean()
    spread_out = (mu - mu.mean(dim=-1, keepdim=True)).pow(2).mean()
    return spread_in, spread_out


def loss_sep_bug(pd_g):
    si, so = spread(pd_g)
    return (torch.log(so.clamp(min=1e-8)) - torch.log(si.clamp(min=1e-8))).clamp(max=0.0)


def loss_sep_fix(pd_g):
    si, so = spread(pd_g)
    return (torch.log(si.clamp(min=1e-8)) - torch.log(so.clamp(min=1e-8))).clamp(min=0.0)


B, N, G = 1, 250, 5
cases = {
    # ① 健康分化：5 个频段各自指向不同方向（设计想要的）
    "健康分化(组间差异大)": torch.stack([
        torch.full((B, N), v) + 0.05 * torch.randn(B, N)
        for v in (-1.2, -0.3, 0.4, 1.1, 1.9)], dim=-1),
    # ② 当前疑似状态：组间无差异 + 组内高频噪声（模型白嫖点）
    "组间无差异+组内噪声": torch.stack([
        0.0 + 1.2 * torch.randn(B, N) for _ in range(G)], dim=-1),
    # ③ 全常数（彻底摆烂）
    "全常数(无组内无组间)": torch.zeros(B, N, G),
}

print("=" * 78)
print("loss_sep 符号验证（B=1, N=250, G=5）")
print("=" * 78)
print(f"{'情形':22s} {'spread_in':>10s} {'spread_out':>11s} "
      f"{'ratio':>8s} {'bug版':>9s} {'正确版':>8s}")
print("-" * 78)
for name, pd_g in cases.items():
    si, so = spread(pd_g)
    ratio = (so / si.clamp(min=1e-12)).item()
    print(f"{name:22s} {si.item():10.5f} {so.item():11.6f} "
          f"{ratio:8.2e} {loss_sep_bug(pd_g).item():9.3f} "
          f"{loss_sep_fix(pd_g).item():8.3f}")

print()
print("判读：")
print("  bug 版在「组间无差异+组内噪声」给出最负值 → 模型被奖励去做噪声。")
print("  正确版在同情形给出最大正值 → 推模型去分化。")
print()
print("量级对账：gen = loss_amp + loss_ph + guard + loss_sep")
print("  若 loss_sep ≈ -12 而其余三项合计 ≈ +2.6 → gen ≈ -9.4（与日志吻合）")
