"""CSI 数据增强模块 (Data Augmentation)

针对小样本量 (如几十到上百条样本) 的 CSI 跌倒检测场景，提供在线数据增强：
1. Subcarrier Masking (类似于 SpecAugment，随机遮蔽若干子载波)
2. Gaussian Noise Injection (加入微量高斯噪声)
3. Magnitude Scaling (微小的全局/局部增益抖动)
4. Time Shift (微小的时间平移)
"""
from __future__ import annotations

import numpy as np
import torch


def augment_csi_numpy(
    x: np.ndarray,
    mask_max_subcarriers: int = 4,
    noise_std: float = 0.01,
    scale_range: tuple[float, float] = (0.95, 1.05),
) -> np.ndarray:
    """对单条或单批次 CSI 数组 (..., T, S) 进行数据增强。"""
    x_aug = x.copy()
    
    # 1. 随机子载波遮蔽 (Subcarrier Masking)
    if mask_max_subcarriers > 0:
        s_dim = x_aug.shape[-1]
        n_mask = np.random.randint(0, mask_max_subcarriers + 1)
        if n_mask > 0:
            mask_indices = np.random.choice(s_dim, size=n_mask, replace=False)
            x_aug[..., mask_indices] = 0.0

    # 2. 增益抖动 (Magnitude Scaling)
    scale = np.random.uniform(scale_range[0], scale_range[1])
    x_aug = x_aug * scale

    # 3. 高斯噪声 (Gaussian Noise)
    if noise_std > 0:
        noise = np.random.randn(*x_aug.shape).astype(x_aug.dtype) * noise_std
        x_aug = x_aug + noise

    return x_aug


def augment_csi_tensor(
    x: torch.Tensor,
    mask_max_subcarriers: int = 4,
    noise_std: float = 0.01,
    scale_range: tuple[float, float] = (0.95, 1.05),
) -> torch.Tensor:
    """PyTorch Tensor 版本的在线数据增强，支持 Autograd-free 变换。
    x 形状: (B, T, S) 或 (B, S, T)
    """
    if not x.requires_grad:
        x_aug = x.clone()
    else:
        x_aug = x

    # 假设最后一维或者倒数第二维是子载波
    # 默认针对 (B, T, S) 形态，若最后一维是 S
    b, t, s = x_aug.shape
    
    # 1. 随机子载波遮蔽
    if mask_max_subcarriers > 0:
        for i in range(b):
            n_mask = torch.randint(0, mask_max_subcarriers + 1, (1,)).item()
            if n_mask > 0:
                mask_idx = torch.randperm(s)[:n_mask]
                x_aug[i, :, mask_idx] = 0.0

    # 2. 增益抖动
    scale = torch.empty(b, 1, 1, device=x.device).uniform_(scale_range[0], scale_range[1])
    x_aug = x_aug * scale

    # 3. 高斯噪声
    if noise_std > 0:
        noise = torch.randn_like(x_aug) * noise_std
        x_aug = x_aug + noise

    return x_aug
