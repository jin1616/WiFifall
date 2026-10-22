"""CSI 数据增强模块 (Data Augmentation)

针对自采 CSI 小样本 (如数十条~百条) 与跨人泛化问题设计的轻量增强流水线。
原理：
  1. Subcarrier Masking (载波掩码/频域Dropout): 随机遮盖几个子载波，强迫模型不依赖特定频段
  2. Jitter & Scaling (幅度抖动与尺度缩放): 模拟人体位置/距离微小漂移导致的整体幅值变化
  3. Time Shift (微小平移): 在时间轴上轻微左右平移，增强模型对跌倒发生位置的时间不变性
"""
from __future__ import annotations

import numpy as np
import torch


def augment_csi_numpy(
    x: np.ndarray,
    p_mask: float = 0.5,
    max_masked_subcarriers: int = 6,
    scale_range: tuple[float, float] = (0.9, 1.1),
    jitter_std: float = 0.02,
    p_shift: float = 0.5,
    max_shift: int = 4,
) -> np.ndarray:
    """对单个或批量 (..., T, S) 或 (..., S, T) 幅度矩阵进行数据增强。

    x: 形状为 (T, S) 或 (B, T, S)，float32
    """
    x_aug = x.copy()
    is_batch = x_aug.ndim == 3
    if not is_batch:
        x_aug = x_aug[None, ...]  # (1, T, S)

    B, T, S = x_aug.shape

    for b in range(B):
        # 1. 尺度缩放 (Scaling)
        if scale_range is not None:
            factor = np.random.uniform(scale_range[0], scale_range[1])
            x_aug[b] *= factor

        # 2. 高斯微扰 (Jitter)
        if jitter_std > 0:
            noise = np.random.normal(0, jitter_std, size=(T, S)).astype(np.float32)
            x_aug[b] += noise

        # 3. 随机子载波掩码 (Subcarrier Masking)
        if p_mask > 0 and np.random.rand() < p_mask:
            n_mask = np.random.randint(1, max_masked_subcarriers + 1)
            mask_idx = np.random.choice(S, size=n_mask, replace=False)
            x_aug[b, :, mask_idx] = 0.0

        # 4. 时间平移 (Time Shift)
        if p_shift > 0 and np.random.rand() < p_shift:
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift != 0:
                x_aug[b] = np.roll(x_aug[b], shift, axis=0)

    # 保证非负
    x_aug = np.clip(x_aug, 0.0, None)

    return x_aug if is_batch else x_aug[0]


class CSIAugmentBatch(torch.nn.Module):
    """PyTorch 批处理在线数据增强层，仅在 training 模式下激活。"""

    def __init__(
        self,
        p_mask: float = 0.5,
        max_subcarriers_mask: int = 5,
        scale_range: tuple[float, float] = (0.92, 1.08),
        jitter_std: float = 0.015,
    ) -> None:
        super().__init__()
        self.p_mask = p_mask
        self.max_subcarriers_mask = max_subcarriers_mask
        self.scale_range = scale_range
        self.jitter_std = jitter_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 x 形状 (B, S, T) 或 (B, T, S)。"""
        if not self.training:
            return x

        # 统一识别 S 维 (通常是 52)
        dim_s = 1 if x.shape[1] == 52 else (2 if x.shape[-1] == 52 else 1)

        # 1. 尺度缩放
        if self.scale_range:
            scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(*self.scale_range)
            x = x * scale

        # 2. 噪声扰动
        if self.jitter_std > 0:
            x = x + torch.randn_like(x) * self.jitter_std

        # 3. 随机载波 Mask
        if self.p_mask > 0 and torch.rand(1).item() < self.p_mask:
            B = x.size(0)
            S = x.size(dim_s)
            num_mask = torch.randint(1, self.max_subcarriers_mask + 1, (1,)).item()
            mask = torch.ones_like(x)
            for b in range(B):
                drop_idx = torch.randperm(S)[:num_mask]
                if dim_s == 1:
                    mask[b, drop_idx, :] = 0.0
                else:
                    mask[b, :, drop_idx] = 0.0
            x = x * mask

        return torch.clamp(x, min=0.0)
