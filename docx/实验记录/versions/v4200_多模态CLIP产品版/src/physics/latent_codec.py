# -*- coding: utf-8 -*-
"""物理版 VAE 编解码器（PhysicalLatentCodec）——阶段一：自编码重建验证

【来源】2026-10-21 用户「物理版 VAE」指令：
    传统 VAE「加噪 -> 去噪还原」对应物理模型「图片 -> 振子场 -> 图片」，
    VAE 骨架不变，只把计算逻辑全部换成物理的。

【定位（第三十四个发现：条件通路维度灾难）】
    场 [N, 2D=250] 每位置 250 自由度，而条件通路只能给 per-band G=5
    （相位 25 倍缺口）。本模块把 VAE 缺失的「压缩」一环补上：
      编码端（零参数纯物理）：场 -> 物理潜（content + texture 两层）
      解码端（可学习）：物理潜 -> 场（Kuramoto 均场重建 + 能量守恒分配）
    生成器（PMNN 物理栈，阶段二）只需在低维物理潜上演化，
    与条件通路的 per-band 粒度天然对齐。

【口径】与 codec/gif_codec.py 严格对齐：
    D = WIN * W_FFT（WIN=5 时间频, W_FFT=25 空间频）
    通道 c <-> h = c // W_FFT（时间频率, 0..4）, w = c % W_FFT（空间频率, 0..24）
    field = cat([z.real, z.imag])：前 D 列实部、后 D 列虚部
    rfft2 在 (win, s_dim) 上：h 轴全长（5）、w 轴减半（25）

【两层物理潜 latent = [B, N, L]】
    content 层（无损、物理态本身，[0:2*WIN]）：
        w=0 列 5 个时间频率系数的 (log1p(a_h), θ_h)
        —— 承载 97.2% 能量的「时空色彩结构」，FFT 本身就是物理变换，
           系数就是振子态，原样保留（log1p 重参数化统一量纲口径）。
    texture 层（Kuramoto 均场压缩）：
        w>0 按空间频率 w 分 N_BANDS 频段（h_split=1 时再按时间频率 h 细分），
        每组取**能量加权 Kuramoto 序参量** + 组能量：
            order = Σ_c a_c·z_c （复数）,  E = Σ_c a_c²
            r   = |order| / E      ∈ [0,1]  相干度
            ψ   = arg(order)               均值相位
            logE = log1p(E)                组能量
        纯物理、零参数（Kuramoto 序参量是均场物理量，不是数据先验）。
    可选 N_MODES>0：组内相位偏离模式 δ_c = wrap(θ_c−ψ) 的前 N_MODES 个
        **能量加权傅里叶系数**（cos/sin 分量）。N_MODES=0 纯序参量；
        N_MODES = n_group−1 时组内相位无损（线性正交变换，仍是物理）。
        这是「保真度逃生阀」：N_MODES=0 压得最狠但保真度最低，
        逐级提高可找到「压缩率 vs 保真度」的平衡点。

【解码（物理 + 可学习）】
    content：直接放回 w=0 列（无损）。
    texture 每组：
        能量守恒分配：a_c = sqrt(E · softmax(logit_c))
            —— 组能量**精确**（E 来自潜变量），组内分配 p_c 可学。
            全场逐 (时间频率 h × 频段) 能量谱在编解码中天然守恒。
        相位重建：θ_c = wrap(ψ + s(r)·η_c + δ_c^Fourier)
            s(r) = sqrt(−2 ln r)：Kuramoto 均场「相干度-展布」关系
            （von Mises 一致性：r=1 → s=0 全同相；r→0 → s=s_max 展布最大）
            η_c：可学习的组内通道展布模式，由 4 维组特征
                 (r, cosψ, sinψ, logE) 经 per-group 小网络给出。
        展布网络 Linear+SiLU+Linear —— 与 cond_init / fft_bypass 条件路径
        同口径，属**编解码通路**而非物理架构主干（项目约定内允许）。

【尺寸】默认 h_split=1, N_BANDS=5, N_MODES=0：
    n_groups = 25（5 频段 × 5 时间频率），L = 10 + 75 = 85（相对 250 压缩 2.9×）
    参数量 ~2.5 万（远低于 500M 上限；4G 显存零压力）

【验收】阶段一目标：重建 PSNR 逼近 codec 天花板（3D-FFT 往返），
    详见 scripts/train_latent_codec.py。
"""
import math
from typing import List, Tuple

import torch
import torch.nn as nn

__all__ = ["PhysicalLatentCodec"]


class _GroupDecoder(nn.Module):
    """单个 (频段 × 时间频率) 组的可学习解码头。

    输入 4 维组特征 (r, cosψ, sinψ, logE)，输出组内每通道两件事：
      amp_logit -> 能量分配（softmax 后 a_c = sqrt(E·p_c)，组能量守恒）
      eta       -> 相位展布模式（θ_c = ψ + s(r)·η_c，η ∈ (−π/2, π/2)）
    """

    def __init__(self, n_ch: int, hidden: int = 64):
        super().__init__()
        self.n_ch = n_ch
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, 2 * n_ch)
        )

    def forward(self, r, cpsi, spsi, logE):
        feats = torch.stack([r, cpsi, spsi, logE], dim=-1)      # [..., 4]
        out = self.net(feats)                                   # [..., 2*n_ch]
        amp_logit = out[..., : self.n_ch]
        eta = (math.pi / 2.0) * torch.tanh(out[..., self.n_ch:])
        return amp_logit, eta


class PhysicalLatentCodec(nn.Module):
    """物理版 VAE：场 <-> 物理潜。encode 零参数纯物理，decode 可学习。"""

    def __init__(self, d_model: int = 125, win: int = 5, n_bands: int = 5,
                 h_split: int = 1, hidden: int = 64, n_modes: int = 0,
                 phys_spread: int = 1, s_max: float = 3.0):
        """
        d_model    : FFT 系数数 D = WIN * W_FFT（125）
        win        : 时间频率数 WIN（5）
        n_bands    : 空间频率 w>0 的频段数 G（5）
        h_split    : 1=按 (频段 × 时间频率) 细分组（25 组）；0=只按频段（5 组）
        hidden     : 展布网络隐藏宽度
        n_modes    : 每组相位偏离傅里叶模式数（0=纯序参量，保真度逃生阀）
        phys_spread: 1=用 Kuramoto 相干度-展布关系 s(r)；0=自由学习（s≡1）
        s_max      : 展布上限（r 下限由 s_max 反推 r_min = e^(−s_max²/2)）
        """
        super().__init__()
        assert d_model % win == 0, "d_model 必须是 win 的整数倍"
        self.d_model = d_model
        self.win = win
        self.w_fft = d_model // win
        self.n_bands = n_bands
        self.h_split = int(h_split)
        self.n_modes = int(n_modes)
        self.phys_spread = int(phys_spread)
        self.s_max = float(s_max)
        self._r_min = math.exp(-0.5 * s_max * s_max)

        w_fft = self.w_fft
        c_ar = torch.arange(d_model, dtype=torch.long)
        w_of_c = c_ar % w_fft                 # 空间频率
        h_of_c = c_ar // w_fft                # 时间频率
        tex_mask = w_of_c >= 1                # texture = w>0
        n_tex_w = w_fft - 1                   # 24

        # content：w==0 列（c = h*w_fft），h=0..4
        self.register_buffer("cont_idx", torch.arange(win, dtype=torch.long) * w_fft)

        # 频段编号：w∈[1, w_fft) 均匀切成 n_bands 段（[0]段含 w=1..5）
        g_of_w = (torch.arange(w_fft) - 1).clamp(min=0) * n_bands // max(n_tex_w, 1)
        g_of_c = g_of_w[w_of_c]
        gid_all = g_of_c * win + h_of_c if self.h_split else g_of_c

        c_tex = c_ar[tex_mask]
        gid_tex = gid_all[tex_mask]
        self.n_groups = int(gid_tex.max().item()) + 1
        # 逐组收集通道（组内按 c 升序 = 按 w 升序，供傅里叶基用）
        self.group_chs: List[torch.Tensor] = []
        for u in range(self.n_groups):
            self.group_chs.append(c_tex[gid_tex == u])
            # non-leaf 列表 + buffer 双轨：buffer 随 .to(device) 移动
            self.register_buffer(f"_grp{u}", self.group_chs[-1])
        self.group_sizes = [int(t.numel()) for t in self.group_chs]

        # 每组可学习解码头
        self.nets = nn.ModuleList(
            [_GroupDecoder(n_ch, hidden) for n_ch in self.group_sizes]
        )

        # �里叶基（N_MODES>0 时）：每组 [n_ch, 2*n_modes]（cos 行 / sin 行）
        if self.n_modes > 0:
            for gi, n_ch in enumerate(self.group_sizes):
                assert self.n_modes < n_ch, f"N_MODES={self.n_modes} >= 组大小 {n_ch}"
                idx = torch.arange(n_ch, dtype=torch.float32)
                ks = torch.arange(1, self.n_modes + 1, dtype=torch.float32)
                ang = 2.0 * math.pi * torch.outer(ks, idx) / float(n_ch)  # [K, n_ch]
                self.register_buffer(f"_fbs{gi}", torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1))
                # [K, n_ch, 2]

        self.latent_dim = 2 * win + self.n_groups * (3 + 2 * self.n_modes)

    # ------------------------------------------------------------------ 工具
    def _grp(self, i: int) -> torch.Tensor:
        return getattr(self, f"_grp{i}")

    @staticmethod
    def polar(field: torch.Tensor, d_model: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B,N,2D] -> (a [B,N,D], θ [B,N,D])。"""
        re = field[..., :d_model]
        im = field[..., d_model:]
        return torch.sqrt(re * re + im * im), torch.atan2(im, re)

    # ------------------------------------------------------------------ 序参量（Kuramoto 均场）
    _E_FLOOR = 1e-12      # 组能量下限：只接住「平坦块 E 恰为 0」
    _ORD_FLOOR = 1e-30    # |order| 平方下限：防 sqrt(0) 梯度 inf

    @classmethod
    def _order_params(cls, zr: torch.Tensor, zi: torch.Tensor):
        """能量加权 Kuramoto 序参量。zr/zi: [..., n_ch] -> (r, psi, E)。

        order = Σ_c a_c·z_c（复数）, E = Σ_c a_c²，r = |order|/E ∈ [0,1]。
        ★ 数值陷阱（曾在冒烟里把 r 炸成 1e8）：分子分母的 eps 必须**同量级
          绑定**——平坦块 E=0 时 order 也=0，若分母 floor=1e-20 而分子
          sqrt(+1e-24)=1e-12，r = 1e-12/1e-20 = 1e8。这里分子分母同取
          1e-12 量级：死组 r → 1e-15/1e-12 = 1e-3 ≈ 0（能量为 0，相位无意义）。
        """
        ac = torch.sqrt(zr * zr + zi * zi)
        E = (ac * ac).sum(-1).clamp(min=cls._E_FLOOR)
        orr = (ac * zr).sum(-1)           # Σ_c a_c·Re(z_c)
        ori = (ac * zi).sum(-1)           # Σ_c a_c·Im(z_c)
        mag = torch.sqrt((orr * orr + ori * ori).clamp(min=cls._ORD_FLOOR))
        r = mag / E
        psi = torch.atan2(ori, orr)
        return r, psi, E
    # ------------------------------------------------------------------ 编码（纯物理，零参数）
    def encode(self, field: torch.Tensor) -> torch.Tensor:
        """场 [B, N, 2D] -> 物理潜 [B, N, L]。零参数、可微（无训练参数）。"""
        re = field[..., : self.d_model]
        im = field[..., self.d_model:]
        a = torch.sqrt(re * re + im * im)

        # ---- content：w=0 列的 (log1p(a), θ)，无损
        ci = self.cont_idx
        parts: List[torch.Tensor] = [
            torch.log1p(a[..., ci]),
            torch.atan2(im[..., ci], re[..., ci]),
        ]
        # ---- texture：每组 Kuramoto 序参量 (r, ψ) + log 能量
        for i in range(self.n_groups):
            chs = self._grp(i)
            r, psi, E = self._order_params(re[..., chs], im[..., chs])
            parts.append(torch.stack([r, psi, torch.log1p(E)], dim=-1))  # [B,N,3]
            if self.n_modes > 0:
                zr, zi = re[..., chs], im[..., chs]
                p = (zr * zr + zi * zi) / E.unsqueeze(-1)        # 能量份额 ≤1
                # 组内相位偏离模式 δ_c = wrap(θ_c − ψ) 的能量加权傅里叶系数
                delta = torch.remainder(torch.atan2(zi, zr) - psi.unsqueeze(-1)
                                        + math.pi, 2 * math.pi) - math.pi
                fbs = getattr(self, f"_fbs{i}")                    # [K, n_ch, 2]
                wd = p * delta                                     # [B,N,n_ch]
                Ak = torch.einsum("bnc,kc->bnk", wd, fbs[..., 0]) * 2.0
                Bk = torch.einsum("bnc,kc->bnk", wd, fbs[..., 1]) * 2.0
                parts.append(torch.cat([Ak, Bk], dim=-1))          # [B,N,2K]
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------ 解码（可学习）
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """物理潜 [B, N, L] -> 场 [B, N, 2D]。"""
        B, N, L = latent.shape
        assert L == self.latent_dim, f"latent L={L} != {self.latent_dim}"
        dev = latent.device
        re = torch.zeros(B, N, self.d_model, device=dev, dtype=latent.dtype)
        im = torch.zeros_like(re)

        # ---- content：无损放回 w=0 列
        loga_c = latent[..., : self.win]
        th_c = latent[..., self.win : 2 * self.win]
        a_c = torch.expm1(loga_c.clamp(max=12.0)).clamp(min=0.0)
        ci = self.cont_idx
        re[..., ci] = a_c * torch.cos(th_c)
        im[..., ci] = a_c * torch.sin(th_c)

        # ---- texture
        o = 2 * self.win
        for i in range(self.n_groups):
            chs = self._grp(i)
            r = latent[..., o]
            psi = latent[..., o + 1]
            logE = latent[..., o + 2]
            o += 3
            # N_MODES 傅里叶系数（如有）
            four = None
            if self.n_modes > 0:
                fb = getattr(self, f"_fbs{i}")                # [K, n_ch, 2]
                K = self.n_modes
                coef = latent[..., o : o + 2 * K]             # [B,N,2K]
                o += 2 * K
                four = (torch.einsum("bnk,kc->bnc", coef[..., :K], fb[..., 0])
                        + torch.einsum("bnk,kc->bnc", coef[..., K:], fb[..., 1]))
                # [B,N,n_ch]

            E = torch.expm1(logE.clamp(max=12.0)).clamp(min=0.0)
            amp_logit, eta = self.nets[i](r, torch.cos(psi), torch.sin(psi), logE)
            p = torch.softmax(amp_logit, dim=-1)              # 能量分配
            ac = torch.sqrt(E.unsqueeze(-1) * p + 1e-24)

            if self.phys_spread:
                # Kuramoto 均场相干度-展布关系：r=1 → s=0；r→r_min → s=s_max
                s = torch.sqrt((-2.0) * torch.log(r.clamp(min=self._r_min, max=1.0)))
                s = s.clamp(max=self.s_max)
            else:
                s = torch.ones_like(r)
            dth = s.unsqueeze(-1) * eta
            if four is not None:
                dth = dth + four
            th = torch.remainder(psi.unsqueeze(-1) + dth + math.pi,
                                 2 * math.pi) - math.pi
            re[..., chs] = ac * torch.cos(th)
            im[..., chs] = ac * torch.sin(th)
        return torch.cat([re, im], dim=-1)

    # ------------------------------------------------------------------ 物理正则
    def order_consistency(self, latent: torch.Tensor, rec_field: torch.Tensor) -> torch.Tensor:
        """重建场的序参量与 latent 的序参量一致性（物理正则项）。

        鼓励解码器待在「物理自洽」流形上：重建出的场重新计算序参量，
        应与潜变量携带的 (r, ψ) 一致。
        """
        re = rec_field[..., : self.d_model]
        im = rec_field[..., self.d_model:]
        o = 2 * self.win
        tot_r = latent.new_zeros(())
        tot_psi = latent.new_zeros(())
        for i in range(self.n_groups):
            chs = self._grp(i)
            r_l = latent[..., o]
            psi_l = latent[..., o + 1]
            o += 3 + (2 * self.n_modes if self.n_modes else 0)
            r_r, psi_r, _ = self._order_params(re[..., chs], im[..., chs])
            tot_r = tot_r + (r_l - r_r).abs().mean()
            tot_psi = tot_psi + (1.0 - torch.cos(psi_r - psi_l)).mean()
        n = float(self.n_groups)
        return tot_r / n + 0.5 * tot_psi / n

    # ------------------------------------------------------------------ 损失（v3600 §7 口径）
    @staticmethod
    def field_loss(a_rec: torch.Tensor, th_rec: torch.Tensor,
                   a_tar: torch.Tensor, th_tar: torch.Tensor,
                   amp_p: float = 0.5):
        """幅度 log1p 域 MSE + 相位 a² 加权 MSE（与 v3600 训练口径一致）。

        返回 (loss_amp, loss_ph, pe, wcos)。
        """
        la = torch.log1p(a_rec.clamp(min=1e-8))
        lt = torch.log1p(a_tar.clamp(min=1e-8))
        w = (a_tar / a_tar.max().clamp(min=1e-8)).pow(amp_p)
        loss_amp = (w * (la - lt).pow(2)).mean()
        pe = torch.remainder(th_rec - th_tar + math.pi, 2 * math.pi) - math.pi
        w_ph = (a_tar ** 2).clamp(min=1e-8)
        loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
        wcos = (w_ph * torch.cos(pe)).sum() / w_ph.sum()
        return loss_amp, loss_ph, pe, wcos

    # ------------------------------------------------------------------ 消融工具
    def zero_texture(self, latent: torch.Tensor) -> torch.Tensor:
        """克隆并把 texture 段置零（r=0,E=0,模式=0）——content-only 基线探针。"""
        out = latent.clone()
        out[..., 2 * self.win :] = 0.0
        return out

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
