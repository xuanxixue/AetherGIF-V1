# -*- coding: utf-8 -*-
"""第4 路条件 = 3D FFT 场。探针：逐位置解码能不能重建 (θ, a)？

★ 关键设计约束：RTX3050 只有 4GB 显存。
  全连接 hidden(512) → N×D(768K) 需要 1.46GB/层，OOM。
  ⇒ 必须用**逐位置**解码器：第 n 个位置只看自己那 2D 个 FFT 系数
     + 少量全局上下文（低秩）。参数量与 N 解耦，显存 O(D)。

这才是 FFT 条件真正的价值：它**逐位置自带完整答案**，
不需要网络去「猜」全局结构 —— 这是 CLIP 向量做不到的事。
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
_p = ROOT
for _ in range(8):
    if (_p / "deps312").is_dir(): break
    if _p.parent == _p: break
    _p = _p.parent
sys.path.insert(0, str(_p))
if (_p / "deps312").is_dir(): sys.path.insert(0, str(_p / "deps312"))
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

fields = []
for gp in sorted((ROOT/"data/gif_raw").rglob("*.gif")):
    fields.append(encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).float())
tgts = torch.stack(fields).to(dev)
B, N, _ = tgts.shape
re_t = tgts[..., :D]; im_t = tgts[..., D:]
a_t = torch.sqrt(re_t**2 + im_t**2 + 1e-8); th_t = torch.atan2(im_t, re_t)
print(f"素材数={B} field={tuple(tgts.shape)}")
print(f"a_t: mean={a_t.mean():.4f} p50={a_t.median():.5f} p99={a_t.flatten().quantile(0.99):.3f}（重尾）")

_W = (torch.arange(D, device=dev) % 25)   # 含 DC(c=0→w=0)
_EDGES = torch.linspace(0, 25, 6, device=dev)
_GID = torch.bucketize(_W.float(), _EDGES[1:-1].float()).clamp(0, 4)
def _expand(x):
    """[B,N,G] → [B,N,D]，按 w=c%25 分组广播（与 cond_init.expand_to_d 同口径）"""
    B_, N_, G_ = x.shape
    out = torch.zeros(B_, N_, D, device=x.device, dtype=x.dtype)
    for g in range(G_):
        msk = (_GID == g)
        if bool(msk.any()):
            out[:, :, msk] = x[:, :, g:g+1].expand(-1, -1, int(msk.sum()))
    return out

class PosDec(nn.Module):
    """逐位置解码器：输入该位置 FFT 系数(2D) + 全局低秩上下文 → (θ̂, â)

    ★ 关键：输出是 **per-band** 的（每位置 D 组），不是标量。
      频率间空间分布毫不相似（记忆铁律：FFT 每通道振幅均值跨 727 倍），
      压成标量等于亲手抹掉全部频段结构。
    """
    G = 5   # 频段数（与 cond_init 的 COND_GROUPS 一致）
    def __init__(self, gctx=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*D + gctx, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2*self.G))
    def forward(self, coeff, g):
        # coeff [B,N,2D], g [B,gctx] → 广播
        B_, N_, _ = coeff.shape
        x = torch.cat([coeff, g.unsqueeze(1).expand(-1, N_, -1)], dim=-1)
        o = self.net(x)
        return torch.atan(o[..., :self.G]), o[..., self.G:].clamp(-8, 4)

def run(gctx, steps=600, lr=3e-4, tag=""):
    torch.manual_seed(0)
    m = PosDec(gctx=gctx).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    # 全局上下文：素材的身份向量（模拟 CLIP 条件那一路）
    g = torch.randn(B, gctx, device=dev)*0.5
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        th_g, la_g = m(tgts, g)                # [B,N,G]
        th_p = _expand(th_g); la_p = _expand(la_g)   # → [B,N,D]
        l_ph = (1 - torch.cos(th_p[:,:,1:] - th_t[:,:,1:]).mean())
        # ★ 第十六死穴的修法：log1p/MSE 关不住尖峰。
        #   FFT 幅度 p50=0.023、p99=4.647（跨 200 倍），均值化的 MSE
        #   会被中位数完全主导 ⇒ corr 0.85 但像素 std 只有 3.1（目标 80）。
        #   改用**逐通道能量比log 差**：d/da logE = 2a/E_m，
        #   梯度不随尖峰衰减，大能量通道自动获得与其份额相称的权重。
        _a_p = torch.exp(la_p[:, :, 1:]).clamp(min=1e-8)
        _a_t = a_t[:, :, 1:].clamp(min=1e-8)
        _E_p = (_a_p ** 2).mean(dim=1, keepdim=True).clamp(min=1e-8)   # [B,1,D-1]
        _E_t = (_a_t ** 2).mean(dim=1, keepdim=True).clamp(min=1e-8)
        l_am = (torch.log(_E_p) - torch.log(_E_t)).pow(2).mean()
        (l_ph + 1.0*l_am).backward(); opt.step()
    with torch.no_grad():
        th_g, la_g = m(tgts, g)
        th_p = _expand(th_g); la_p = _expand(la_g)
        c = torch.cos(th_p[:,:,1:] - th_t[:,:,1:]).mean().item()
        lp = la_p[:,:,1:]; lp = lp-lp.mean()
        lt = torch.log(a_t[:,:,1:].clamp(min=1e-6)); lt = lt-lt.mean()
        cr = ((lp*lt).sum()/(lp.norm()*lt.norm())).item()
        _ap = torch.exp(la_p[:,:,1:]); _at = a_t[:,:,1:]
        ratio = (_ap.std()/_at.std()).item()
    np_ = sum(p.numel() for p in m.parameters())/1e6
    print(f"  {tag:20s} gctx={gctx:4d}  cos={c:+.4f}  corr={cr:+.4f}  a_std比={ratio:.3f}  ({np_:.2f}M)")
    return c, cr, m

print("\n=== 逐位置解码器：输入 FFT 场本身（4 步 0 训练，验证可学性）===")
res = {}
for gctx in [0, 64]:
    res[gctx] = run(gctx, tag=f"位置级+全局{gctx}")

# 像素域 PSNR：最强判据
print("\n=== 像素域验证（用训练好的位置级解码器，不经物理栈）===")
gctx = max(res, key=lambda k: res[k][1])
c, cr, m = res[gctx]
_GCTX = torch.randn(B, gctx, device=dev)*0.5   # 与 run 内同一 seed 下的输入
with torch.no_grad():
    th_g, la_g = m(tgts, _GCTX)
    th_p = _expand(th_g); la_p = _expand(la_g)
    a_p = torch.exp(la_p).clamp(min=1e-6)
    fld_p = torch.cat([a_p*torch.cos(th_p), a_p*torch.sin(th_p)], -1)
    for i in range(B):
        fr_p = decode_field_to_frames(fld_p[i].cpu(), TS,GRID,PATCH,WIN,IMG,D, normalize=True)
        fr_t = decode_field_to_frames(tgts[i].cpu(), TS,GRID,PATCH,WIN,IMG,D, normalize=True)
        fp = np.stack(fr_p).astype(np.float64); ft = np.stack(fr_t).astype(np.float64)
        mse = ((fp-ft)**2).mean(); psnr = 10*math.log10(255.0**2/max(mse,1e-12))
        print(f"  素材{i}: PSNR={psnr:.2f} dB   像素 std out={fp.std():.2f} tar={ft.std():.2f}")
