"""卷积神经网络 (CNN) 模块。

跌倒检测场景中，CSI 形状为 (T=128 时间步, S=52 子载波) 的实值幅度矩阵，
CNN1D 把 S 当作 in_channels、T 当作时间维，对 1D 时间序列做局部时-频模式学习。

网络结构 (参见 :class:`CNN1D`):
    Input (B, S, T)   ──┐
                        │  Conv1d(S→32, k=5, pad=2) → BN → ReLU → MaxPool(2)
                        │  Conv1d(32→64, k=5, pad=2) → BN → ReLU → MaxPool(2)
                        │  Conv1d(64→128, k=5, pad=2) → BN → ReLU → MaxPool(2)
                        │
    AdaptiveAvgPool1d(8) │  →  Flatten  →  Linear(1024, 128)  →  ReLU → Dropout(0.3)
                        │  →  Linear(128, 2)
                        └─  输出 2 维 logits (CrossEntropyLoss, 无标签平滑)

正则化策略: 在仅约 1100 个训练窗的小数据集上，**过强正则反而欠拟合**——
实测 Dropout=0.5 + label_smoothing=0.05 让 CNN 窗 Acc 从 0.856 跌到 0.763，
故当前取 Dropout=0.3 (config.CNN_DROPOUT) + 不做标签平滑
(config.CNN_LABEL_SMOOTHING=0.0)，配合早停 (patience=8)。

前向传播由 :meth:`CNN1D.forward` 显式实现 (一句 ``forward``)，其余逻辑
(类别权重、Adadelta 余弦退火、早停、持久化) 在模块级函数中拆分实现。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .config import CNN_DROPOUT, CNN_LABEL_SMOOTHING


# ----------------------------------------------------------------------------
# 数据增强（在线，仅训练时作用于原始窗 batch）
# ----------------------------------------------------------------------------
def _augment_window_batch(
    xb: torch.Tensor,
    *,
    p_mask: float = 0.5,
    max_mask: int = 4,
    scale_range: tuple[float, float] = (0.9, 1.1),
    jitter_std: float = 0.02,
    max_shift: int = 4,
) -> torch.Tensor:
    """对一个 ``(B, T, S)`` 幅度 batch 做在线增强，强迫模型不背特定子载波/幅度/时延。

    * 全局增益抖动：模拟人体与天线距离的微小漂移（幅度是乘性的）
    * 高斯噪声：模拟采样抖动与热噪声
    * 子载波掩码：随机遮盖若干子载波，斩断"背某个固定频段"的过拟合
    * 时间平移：跌倒冲击在 1.54 s 窗内的位置随机平移，增强时间不变性

    仅作用于训练 batch；验证/测试不经过本函数。
    """
    B, T, S = xb.shape
    # 1. 全局增益抖动（乘性，符合 CSI 幅度物理）
    scale = torch.empty(B, 1, 1, device=xb.device).uniform_(*scale_range)
    x = xb * scale
    # 2. 高斯噪声
    if jitter_std > 0:
        x = x + torch.randn_like(x) * jitter_std
    # 3. 子载波掩码（按概率触发，随机选择 1..max_mask 个子载波整列置零）
    if p_mask > 0:
        mask = torch.ones(B, 1, S, device=xb.device)
        for b in range(B):
            if float(torch.rand(1).item()) < p_mask:
                n_mask = int(torch.randint(1, max_mask + 1, (1,)).item())
                drop_idx = torch.randperm(S)[:n_mask]
                mask[b, 0, drop_idx] = 0.0
        x = x * mask
    # 4. 时间平移（小幅 roll，最大约 48 ms；roll 会 wrap 边界，幅度数据下可接受）
    shift = torch.randint(-max_shift, max_shift + 1, (B,), device=xb.device)
    for b in range(B):
        if int(shift[b]) != 0:
            x[b] = torch.roll(x[b], int(shift[b]), dims=0)
    return torch.clamp(x, min=0.0)


# ----------------------------------------------------------------------------
# 通道注意力模块 (Squeeze-and-Excitation)
# ----------------------------------------------------------------------------
class SELayer(nn.Module):
    """Squeeze-and-Excitation 模块: 自适应调节各通道（特征图）重要性。"""

    def __init__(self, channel: int, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        mid = max(4, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


# ----------------------------------------------------------------------------
# 网络结构
# ----------------------------------------------------------------------------
class CNN1D(nn.Module):
    """1D-CNN + SE-Attention: 在子载波×时间步矩阵上学时-频局部模式。

    设计原则:
      * 三层 conv + BN + ReLU + SE-Attention + MaxPool 提取多尺度时频结构
      * SELayer 自适应抑制噪声子载波响应，强化动作关键特征
      * AdaptiveAvgPool 让 128 步时间序列变成固定 8 帧，避免依赖输入长度
      * 2 层 FC 头负责聚合到 2 类 (非跌倒 / 跌倒)
      * Dropout=0.3 抑制过拟合 (训练样本少)
    """

    def __init__(
        self,
        n_subcarriers: int = 52,
        n_classes: int = 2,
        hidden_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.3,
        pool_out: int = 8,
        use_se: bool = False,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        # 让 save_cnn 可以回填 dropout 值（无需重新训练即可知道原始正则强度）
        self._init_dropout = dropout
        self.use_se = use_se
        self.se_reduction = se_reduction
        self.n_subcarriers = n_subcarriers
        layers: list[nn.Module] = []
        in_c = n_subcarriers
        for h in hidden_channels:
            layers += [
                nn.Conv1d(in_c, h, kernel_size=5, padding=2),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ]
            if use_se:
                layers.append(SELayer(h, reduction=se_reduction))
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_c = h
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(pool_out)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_c * pool_out, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    # 显式前向传播: 一目了然输入 (B, T, S) 的张量走向
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` 形状:
            (B, T, S)  - 原始 ``(时间步, 子载波)``，自动转置为通道优先；或
            (B, S, T)  - 已是通道优先，直接走卷积。

        约定: ``S == n_subcarriers``。若第 3 维等于此值，就当作 (B, T, S)；
        否则当作 (B, S, T) 已对齐好的通道优先输入。
        """
        if x.dim() == 3 and x.shape[-1] == self.n_subcarriers and x.shape[1] != self.n_subcarriers:
            x = x.transpose(1, 2)  # (B, S, T)
        z = self.features(x)           # (B, last_ch, T/8)
        z = self.pool(z)               # (B, last_ch, pool_out)
        return self.classifier(z)      # (B, n_classes)


# ----------------------------------------------------------------------------
# sklearn 兼容包装器: 给上层 (run_demo_offline / run_train) 暴露一致的接口
# ----------------------------------------------------------------------------
class CNNClassifier:
    """sklearn 兼容包装器: ``predict_proba(X)`` 返回 ``(N, 2)`` 的 numpy 数组。

    输入 X 形状: (N, T, S) 浮点数组。底层是一个已训练好的 :class:`CNN1D`。
    """

    def __init__(self, model: CNN1D, device: str = "cpu") -> None:
        self.model = model
        self.device = device

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """对 ``X`` (N, T, S) 推断，返回 ``(N, 2)`` softmax 概率。"""
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                batch = torch.as_tensor(
                    X[i : i + 256], dtype=torch.float32
                ).to(self.device)
                logits = self.model(batch)
                proba = F.softmax(logits, dim=-1).cpu().numpy()
                out.append(proba)
        return np.concatenate(out, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回 0/1 类别预测。"""
        return self.predict_proba(X).argmax(axis=1)

    def state_dict_copy(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


# ----------------------------------------------------------------------------
# 训练循环
# ----------------------------------------------------------------------------
def _split_fit_val(
    y: np.ndarray, seed: int, groups: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """内部早停验证集切分，返回 (fit_idx, val_idx)。

    给了 ``groups``（每条窗所属的样本文件 id）就**按文件分组切**——同一文件
    的窗要么全在 fit、要么全在 val。窗级随机切分会让同一文件的窗同时出现在
    两边（相邻窗只差 0.39 s，几乎同源），val_loss 被低估、早停偏晚，等于
    变相过拟合。没给 groups 时退回窗级切分（历史行为）。

    本函数放在 cnn 模块（依赖链最底层），models.py 的 FC 模型训练也从这里复用。
    """
    if groups is not None:
        g = np.asarray(groups)
        uniq = np.unique(g)
        rng = np.random.RandomState(seed)
        rng.shuffle(uniq)
        n_val_g = max(1, int(round(0.1 * len(uniq))))
        val_g = set(uniq[:n_val_g].tolist())
        is_val = np.array([gi in val_g for gi in g])
        return np.where(~is_val)[0], np.where(is_val)[0]
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    n_val = max(1, int(round(0.1 * len(y))))
    return idx[n_val:], idx[:n_val]


def _make_class_weights(y: np.ndarray, device: str) -> torch.Tensor:
    """根据 ``y`` 计算类别权重 (等价于 ``class_weight='balanced'``)。"""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    n = pos + neg
    w0 = n / (2.0 * max(neg, 1.0))
    w1 = n / (2.0 * max(pos, 1.0))
    return torch.tensor([w0, w1], device=device, dtype=torch.float32)


def train_cnn(
    X_raw: np.ndarray,
    y: np.ndarray,
    *,
    X_raw_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    n_epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
    patience: int = 8,
    verbose: bool = False,
    n_subcarriers: int | None = None,
    val_X: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    augment: bool | None = None,
    use_se: bool | None = None,
) -> CNNClassifier:
    """训练 CNN。

    参数
    ----
    X_raw : (N, T, S) 浮点 ndarray
    y     : (N,) 0/1 标签
    val_X, val_y : 可选外部验证集（与 train_mlp 统一命名；旧名 X_raw_val/y_val
                   仍兼容，两边都给时以 val_X/val_y 为准）。若都不给，按
                   ``groups``（每条窗的样本文件 id）分组内部切 10% 做早停，
                   避免同一文件的窗同时落进 fit/val 的泄漏；groups 也没给时
                   退回窗级随机切分（历史行为）。
    n_subcarriers : 默认按 ``X_raw.shape[-1]`` 自动推断。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if augment is None:
        augment = bool(config.CNN_AUGMENT)
    if use_se is None:
        use_se = bool(config.CNN_USE_SE)

    if n_subcarriers is None:
        n_subcarriers = int(X_raw.shape[-1])

    if val_X is None and X_raw_val is not None:
        val_X, val_y = X_raw_val, y_val

    if val_X is None:
        fit_idx, val_idx = _split_fit_val(y, seed, groups)
        X_raw_val = X_raw[val_idx]
        y_val = y[val_idx]
        X_raw_fit = X_raw[fit_idx]
        y_fit = y[fit_idx]
    else:
        X_raw_val, y_val = val_X, val_y
        X_raw_fit, y_fit = X_raw, y

    model = CNN1D(
        n_subcarriers=n_subcarriers,
        dropout=CNN_DROPOUT,
        use_se=use_se,
        se_reduction=config.CNN_SE_REDUCTION,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)
    crit = nn.CrossEntropyLoss(
        weight=_make_class_weights(y_fit, device),
        label_smoothing=CNN_LABEL_SMOOTHING,
    )

    train_ds = TensorDataset(
        torch.as_tensor(X_raw_fit, dtype=torch.float32),
        torch.as_tensor(y_fit, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.as_tensor(X_raw_val, dtype=torch.float32),
        torch.as_tensor(y_val, dtype=torch.long),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    bad = 0

    for ep in range(n_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            # 在线数据增强 (抗过拟合，仅训练时；验证/测试不经过)
            if augment:
                xb = _augment_window_batch(
                    xb,
                    p_mask=config.CNN_AUG_P_MASK,
                    max_mask=config.CNN_AUG_MAX_MASK,
                    scale_range=config.CNN_AUG_SCALE,
                    jitter_std=config.CNN_AUG_JITTER,
                    max_shift=config.CNN_AUG_MAX_SHIFT,
                )
            optim.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            optim.step()
            train_loss += float(loss.item()) * xb.size(0)
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                val_loss += float(crit(out, yb).item()) * xb.size(0)
        val_loss /= max(len(y_val), 1)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if verbose:
            print(
                f"  [cnn] epoch {ep+1:3d}/{n_epochs}  "
                f"train_loss={train_loss/len(y_fit):.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"lr={sched.get_last_lr()[0]:.2e}"
            )

        if bad >= patience:
            if verbose:
                print(f"  [cnn] early stop at epoch {ep+1}, best_val_loss={best_val_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return CNNClassifier(model, device=device)


# ----------------------------------------------------------------------------
# 持久化
# ----------------------------------------------------------------------------
def save_cnn(classifier: CNNClassifier, path_dir: str | Path, tag: str = "cnn") -> Path:
    """保存 CNN 权重 + 架构信息。"""
    path_dir = Path(path_dir)
    path_dir.mkdir(parents=True, exist_ok=True)
    blob = {
        "state_dict": classifier.state_dict_copy(),
        "arch": {
            "n_subcarriers": getattr(classifier.model, "n_subcarriers", 52),
            "n_classes": 2,
            "hidden_channels": (32, 64, 128),
            "dropout": float(getattr(classifier.model, "_init_dropout", 0.3)),
            "pool_out": 8,
            "use_se": bool(getattr(classifier.model, "use_se", False)),
            "se_reduction": int(getattr(classifier.model, "se_reduction", 8)),
        },
    }
    p = path_dir / f"{tag}.pt"
    torch.save(blob, p)
    return p


def load_cnn(
    path_dir: str | Path,
    tag: str = "cnn",
    device: str = "cpu",
) -> CNNClassifier:
    """从 ``.pt`` 加载 CNN（不依赖特征名 JSON）。"""
    blob = torch.load(Path(path_dir) / f"{tag}.pt", map_location=device, weights_only=False)
    arch = blob.get("arch", {})
    model = CNN1D(
        n_subcarriers=arch.get("n_subcarriers", 52),
        n_classes=arch.get("n_classes", 2),
        hidden_channels=tuple(arch.get("hidden_channels", (32, 64, 128))),
        dropout=arch.get("dropout", 0.3),
        pool_out=arch.get("pool_out", 8),
        use_se=bool(arch.get("use_se", False)),
        se_reduction=int(arch.get("se_reduction", 8)),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return CNNClassifier(model, device=device)
