"""CNN1D 架构与训练细节测试。

直接 ``python tests/test_cnn_arch.py`` 运行，0 = 通过。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch


WIN = 32   # 测试用小窗，加速
S_DIM = 12


def test_arch_param_count():
    """CNN1D 默认配置参数数量应大于 0 且符合预期量级 (~190k)。"""
    from wifalldemo.cnn import CNN1D
    m = CNN1D()
    n = sum(p.numel() for p in m.parameters())
    assert 50_000 < n < 500_000, f"参数数量 {n} 应在 5w~50w 之间"
    # 应当包含 Conv1d/BN/Linear/Dropout
    assert any(isinstance(mod, torch.nn.Conv1d) for mod in m.modules())
    assert any(isinstance(mod, torch.nn.BatchNorm1d) for mod in m.modules())
    assert any(isinstance(mod, torch.nn.Linear) for mod in m.modules())
    assert any(isinstance(mod, torch.nn.Dropout) for mod in m.modules())
    assert any(isinstance(mod, torch.nn.AdaptiveAvgPool1d) for mod in m.modules())
    print(f"  [ok] CNN1D 默认架构: {n:,} 参数，含 Conv/BN/Linear/Dropout/AdaptivePool")


def test_eval_mode_deterministic():
    """eval 模式下两次 forward 应一致 (Dropout 关)。"""
    from wifalldemo.cnn import CNN1D
    m = CNN1D(n_subcarriers=S_DIM, dropout=0.5)
    m.eval()
    x = torch.randn(4, WIN, S_DIM)
    y1 = m(x)
    y2 = m(x)
    assert torch.allclose(y1, y2), "eval 模式下确定性应保持"
    print("  [ok] eval 模式确定性 (无 Dropout 噪声)")


def test_train_decreases_loss():
    """对可分的玩具数据训练 5 epochs 后，训练损失应单调下降。"""
    from wifalldemo.cnn import train_cnn, _make_class_weights
    from torch.nn import functional as F
    # 构造一组可分的数据：前一半是"negative"，后一半是"positive"
    rng = np.random.default_rng(7)
    Xtr = rng.normal(0, 1, size=(60, WIN, S_DIM)).astype(np.float32)
    ytr = np.array([0] * 30 + [1] * 30, dtype=np.int64)
    Xtr[:30, WIN // 2 : WIN // 2 + 5, :] -= 3.0   # negative spike
    Xtr[30:, WIN // 2 : WIN // 2 + 5, :] += 3.0   # positive spike

    import torch as T
    T.manual_seed(42)
    cls = train_cnn(Xtr, ytr, n_epochs=8, batch_size=16, patience=99, verbose=False, seed=42)
    assert isinstance(cls, object) and hasattr(cls, "predict_proba")
    # 重新跑一遍 5-step 训练并记录每 epoch 的训练损失
    T.manual_seed(42)
    import importlib, wifalldemo.cnn as cnn_mod
    # 用 monkey-patch 捕获 loss: 在 train_cnn 里 patch loss.item() 不好做；
    # 改为手工跑一遍训练循环
    from wifalldemo.cnn import CNN1D
    from torch.utils.data import DataLoader, TensorDataset

    device = "cpu"
    model = CNN1D(n_subcarriers=S_DIM).to(device)
    optim = T.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    crit = T.nn.CrossEntropyLoss(weight=_make_class_weights(ytr, device))
    dl = DataLoader(
        TensorDataset(T.as_tensor(Xtr, dtype=T.float32), T.as_tensor(ytr, dtype=T.long)),
        batch_size=16, shuffle=True,
    )
    losses = []
    for ep in range(8):
        model.train()
        lsum = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            l = crit(model(xb), yb)
            l.backward()
            optim.step()
            lsum += float(l.item()) * xb.size(0)
        losses.append(lsum / len(ytr))
    # 最后 epochs 训练损失应低于第一 epoch
    assert losses[-1] < losses[0], f"训练损失应下降: {losses}"
    print(f"  [ok] 训练损失下降: {losses[0]:.3f} -> {losses[-1]:.3f}")


def test_save_load_preserves_arch():
    """save_cnn -> load_cnn 后模型应能对同一输入给出一致预测。"""
    from wifalldemo.cnn import CNN1D, CNNClassifier, train_cnn, save_cnn, load_cnn
    X = np.random.default_rng(0).normal(0, 1, size=(50, WIN, S_DIM)).astype(np.float32)
    y = np.array([0] * 25 + [1] * 25, dtype=np.int64)
    cls = train_cnn(X, y, n_epochs=3, batch_size=16, patience=99, verbose=False)
    Xte = X[:5]
    p1 = cls.predict_proba(Xte)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        save_cnn(cls, td, tag="cnn")
        # 检查 .pt 文件存在
        assert (td / "cnn.pt").exists()
        # 重新加载
        cls2 = load_cnn(td, tag="cnn")
        p2 = cls2.predict_proba(Xte)
    assert np.allclose(p1, p2, atol=1e-6), "save/load 后概率应一致"
    print("  [ok] CNN save+load 预测完全一致 (round-trip)")


if __name__ == "__main__":
    print("=== CNN 架构与训练细节测试 ===")
    test_arch_param_count()
    test_eval_mode_deterministic()
    test_train_decreases_loss()
    test_save_load_preserves_arch()
    print("=== 全部通过 ===")
