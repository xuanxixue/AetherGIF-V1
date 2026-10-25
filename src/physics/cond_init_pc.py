# -*- coding: utf-8 -*-
"""CondInitPC —— per-channel 条件初始态调制器（第 36 个发现的直接修复）。

【证据链（全部实测）】
  1. v3600 无条件单样本 overfit：drift 拿**逐通道** target 当吸引中心 →
     wcos **0.9991** → 物理栈能保持/演化逐通道场；
  2. v4200 teacher 天花板探针（diag_teacher_ceiling.py，fix_r4 ckpt）：
     把答案 FFT 场喂给 G=5 的 fft_bypass + G=5 的 cond_init →
     wcos 仅 +0.44~0.51，**PSNR 13.76/9.64 dB ≈ 噪声基线 13.03/7.44**；
  3. 两者唯一差别 = 条件初始态与漂移锚的**逐通道 vs 5组广播**。
 ⇒ G=5 per-group 结构（125 通道共享 5 相位；content 层 w=0 的 5 个时间
   频率通道 c=0/25/50/75/100 全落 group 0、共享每位置 1 个相位）就是天花板。

【本模块】
与 CondInitModulator 同接口（forward / expand_to_d / supervise 签名兼容），
但 θ̂ / â 输出为 **[B, N, D] 逐通道**，直接干掉 25× 相位瓶颈：
  - θ̂ = arctan(to_phase(h))·2 ∈ (−π, π]（arctan×2 覆盖全圆，见
    CondInitModulator 的「arctan 折叠」教训）
  - â = softplus(to_amp(h)) > 0，dlna = log(â) − log(a0) 抵消随机 a0 抖动
  - drift 锚 = θ̂ 逐通道 → 直接分量 sin(θ̂−θ) 等价于 v3600 的逐通道吸引

【参数与显存（4GB 约束）】
  to_phase/to_amp = Linear(hidden,R) + Linear(R, N*D)。R=CONDO_PC_RANK 默认 8：
    每头 6.15M 参数 → 权重+梯度+AdamW(m,v) ≈ 98MB/头，两头 ≈ 196MB，
    叠加实测训练峰值 3.79GB ≈ 3.98GB（放得下；OOM 时降 R=4）。
  总模型 ≈ 21.5M 参数（上限 500M，余量充足）。

【 何为 R（低秩）】
  所有条件信息经 h（512 维）→ R 个标量 h_r，再由**每 (位置,通道) 一套**
  的读出基 W[n,c,r] 线性组合成逐通道预测。R 是「一次条件能调整多少个
  独立空间模式」的瓶颈：R=8 在 2 素材过拟合下绰绰有余（W 本身有
  N*D*R = 6.1M 自由度），35k 数据集阶段的 R 是后续调参点。
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 与 CondInitModulator 同口径（import 时的环境变量）
_IDEAL_CLAMP = float(os.environ.get("IDEAL_CLAMP", "") or math.pi)
_AMP_SOFT = float(os.environ.get("PC_AMP_SOFT", "0.5"))   # (a_tar/a_max)^p 软化权重幂


class CondInitPC(nn.Module):
    """条件向量 → 逐通道初始相位/幅度（替代 CondInitModulator）。"""

    def __init__(self, cond_dim: int = 6147, n_osc: int = 6144,
                 d_model: int = 125, hidden: int = 512, rank: int = 8,
                 amp_scale: float = 1.5, zero_init: bool = True):
        super().__init__()
        self.n_osc = n_osc
        self.d_model = d_model
        self.amp_scale = amp_scale
        _rank = max(1, int(os.environ.get("COND_PC_RANK", rank)))
        self.rank = _rank
        # 与 CondInitModulator 同一 proj（h 爆炸 → LayerNorm 修复一并继承）
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        # 两路逐通道输出头（低秩分解，4GB 显存约束）
        self.to_phase = nn.Sequential(
            nn.Linear(hidden, _rank, bias=False),
            nn.Linear(_rank, n_osc * d_model))
        self.to_amp = nn.Sequential(
            nn.Linear(hidden, _rank, bias=False),
            nn.Linear(_rank, n_osc * d_model))
        if zero_init:
            # 零初始化最后一层 → 起步 θ̂≡0/â≡softplus(0)≈0.69，纯随机起点平滑过渡
            for _lin in (self.to_phase[1], self.to_amp[1]):
                nn.init.zeros_(_lin.weight)
                if _lin.bias is not None:
                    nn.init.zeros_(_lin.bias)

    def forward(self, cond, return_parts: bool = False,
                n_pos: int = None, theta0: torch.Tensor = None,
                a0: torch.Tensor = None):
        """
        cond: [B, cond_dim]
        θ̂ 模式（theta0/a0 给定时，与 CondInitModulator 同接口）：
            dtheta = wrap(θ̂ − θ0)  → θ0 + dθ ≡ θ̂ (mod 2π)，随机初相位被抵消
            dlna   = log(â) − log(a0) → a0·exp(dlna) = â，随机 a0 抖动被抵消
        返回 dtheta/dlna [B,N,D]（**逐通道**），parts 含 theta_hat/a_hat。
        """
        h = self.proj(cond)                                   # [B,hidden]
        B = h.shape[0]
        N = int(n_pos) if n_pos is not None else self.n_osc
        D = self.d_model
        _pre_ph = self.to_phase(h).reshape(B, N, D)
        _pre_am = self.to_amp(h).reshape(B, N, D)
        theta_hat = torch.arctan(_pre_ph) * 2.0               # (−π, π] 全圆
        a_hat = F.softplus(_pre_am) + 1e-6                    # 正幅度
        parts = {"h": h, "theta_hat": theta_hat, "a_hat": a_hat}
        if theta0 is not None:
            dtheta = torch.remainder(theta_hat - theta0 + math.pi,
                                     2 * math.pi) - math.pi
            if a0 is not None:
                dlna = torch.log(a_hat.clamp(min=1e-6)) - torch.log(
                    a0.clamp(min=1e-6))
            else:
                dlna = torch.log(a_hat.clamp(min=1e-6))
            if return_parts:
                return dtheta, dlna, parts
            return dtheta, dlna
        # 非θ̂ 模式（兼容 warmup 路径，不常用）
        dlna = torch.log(a_hat.clamp(min=1e-6))
        if return_parts:
            return theta_hat, dlna, parts
        return theta_hat, dlna

    def expand_to_d(self, x):
        """已经是 [B,N,D]，恒等返回（接口兼容）。"""
        return x

    @staticmethod
    def supervise(dtheta, dlna, target_field, d_model: int = 125,
                  w_ph: torch.Tensor = None,
                  guard_bot: float = 0.4, guard_w: float = 1.0,
                  sep_w: float = 1.0,
                  theta0: torch.Tensor = None,
                  abs_w: float = 0.0,
                  theta_hat: torch.Tensor = None,
                  a_hat: torch.Tensor = None):
        """逐通道直接监督（θ̂ 模式下真正的教师）。

        签名与 CondInitModulator.supervise 兼容（train_guided.py 无需改调用），
        但语义全部换成逐通道：
          - 相位：wrap(θ̂ − θ_tar) 的加权 MSE（a² 逐通道归一权重，
            clamp 到 IDEAL_CLAMP 防压垮其余项）。θ̂ 已是逐通道，
            无「组圆均值折中」误差——这是与 G=5 版的本质区别。
          - 幅度：log1p 域 MSE + (a_tar/a_max)^p 软化权重
            （记忆铁律：a_tar² 加权让 DC 独占 99.9% 监督，过度）。
          - guard/sep 项返回 0：θ̂ 模式下 guard 治的是 tanh 饱和，
            PC 的 arctan×2 全圆无饱和；sep 治的是组间分化，
            PC 逐通道本就无组。abs_w 参数在 PC 中被吸收进主相位项
            （监督对象恒为 θ̂ 本身）。

        ★ 按「不注入数据先验」铁律，本监督只作用于模型自产的未来量
          （θ̂/â 是条件通路的输出），target 只作为对齐目标。
        """
        del dtheta, dlna, guard_bot, sep_w    # θ̂ 模式下不用派生物
        dev = theta_hat.device if theta_hat is not None else (
            a_hat.device if a_hat is not None else dtheta.device)
        re_t, im_t = target_field[..., :d_model], target_field[..., d_model:]
        a_t = torch.sqrt(re_t ** 2 + im_t ** 2 + 1e-8)          # [B,N,D]
        th_t = torch.atan2(im_t, re_t)
        _loss = torch.zeros((), device=dev)
        # ---- 相位：逐通道 wrap MSE，a² 逐通道归一权重 ----
        if theta_hat is not None:
            pe = torch.remainder(theta_hat - th_t + math.pi,
                                 2 * math.pi) - math.pi         # [B,N,D]
            wch = a_t[0].detach() ** 2 if a_t.shape[0] == 1 else a_t.detach() ** 2
            wch = wch.clamp(min=1e-8)
            wch = wch / wch.mean(dim=0, keepdim=True).clamp(min=1e-8)
            _loss = _loss + (wch * pe.clamp(max=_IDEAL_CLAMP).pow(2)).mean()
        # ---- 幅度：log1p 域 MSE，(a_tar/a_max)^p 软化权重 ----
        if a_hat is not None:
            la_p = torch.log1p(a_hat.clamp(min=1e-8))
            la_t = torch.log1p(a_t.clamp(min=1e-8))
            if a_t.shape[0] == 1:
                a_t_s = a_t[0]
            else:
                a_t_s = a_t
            w_a = (a_t_s / a_t_s.max().clamp(min=1e-8)).pow(_AMP_SOFT)
            _loss = _loss + (w_a * (la_p - la_t).pow(2)).mean()
        # guard_w / abs_w 为接口兼容保留（PC 语义已并入主项）
        if guard_w > 0 or abs_w > 0:
            pass
        return _loss
