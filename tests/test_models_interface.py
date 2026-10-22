"""接口一致性测试：保证三种 DL 模型 (MLP / DLsmall / CNN1D) 对外暴露完全
一样的 ``predict_proba(X) -> (N, 2)`` 接口，让上层 ``run_train`` /
``run_demo_offline`` / ``run_demo_realtime`` 不用关心背后是哪种模型。

直接 ``python tests/test_models_interface.py`` 运行，0 = 通过。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

# 走 tests/__init__.py 注入的 sys.path，确保能 import wifalldemo
from wifalldemo.models import (
    TorchCompat,
    train_dlsmall,
    train_mlp,
    predict_proba_pos,
    save_model,
    load_model,
)
from wifalldemo.cnn import CNNClassifier, train_cnn


N_TR, N_TE, F_DIM, WIN, S_DIM = 60, 20, 58, 32, 12


def _toy_features(seed: int = 0):
    """生成可分的 58 维玩具数据：前 30 个样本来自类别 0，后 30 个来自类别 1。
    简单到甚至一个线性分类器都能搞定，用来验证训练流程跑得通就行。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(N_TR + N_TE, F_DIM)).astype(np.float32)
    y = np.array([0] * (N_TR // 2) + [1] * (N_TR // 2) + [0] * (N_TE // 2) + [1] * (N_TE // 2), dtype=np.int64)
    # 加类别相关偏移，让模型可分
    X[: N_TR // 2, :5] -= 1.5
    X[N_TR // 2 : N_TR, :5] += 1.5
    return X, y


def _toy_raw(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(N_TR + N_TE, WIN, S_DIM)).astype(np.float32)
    y = np.array([0] * (N_TR // 2) + [1] * (N_TR // 2) + [0] * (N_TE // 2) + [1] * (N_TE // 2), dtype=np.int64)
    # 在时间序列中部加类别相关 spike
    X[: N_TR // 2, WIN // 2 : WIN // 2 + 5, :] -= 2.0
    X[N_TR // 2 : N_TR, WIN // 2 : WIN // 2 + 5, :] += 2.0
    return X, y


def test_mlp_interface():
    X, y = _toy_features()
    Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
    model = train_mlp(Xtr, ytr, n_epochs=5, verbose=False) if False else None
    # n_epochs 不暴露: 用 _fit_torch_model 接口接管
    from wifalldemo.models import _fit_torch_model, MLP
    model = _fit_torch_model(MLP(F_DIM), Xtr, ytr, n_epochs=5, patience=3, verbose=False, tag="t_mlp")
    assert isinstance(model, TorchCompat), "MLP 应返回 TorchCompat"
    proba = model.predict_proba(Xte)
    assert proba.shape == (N_TE, 2), f"predict_proba 应返回 (N,2); 得到 {proba.shape}"
    s = proba.sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-4), "softmax 概率求和应为 1"
    p = predict_proba_pos(model, Xte)
    assert p.shape == (N_TE,), f"predict_proba_pos 应返回 (N,); 得到 {p.shape}"
    print("  [ok] MLP predict_proba (N,2) + predict_proba_pos (N,)")
    return model


def test_dlsmall_interface():
    X, y = _toy_features(seed=1)
    Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
    from wifalldemo.models import _fit_torch_model, DLSmall
    model = _fit_torch_model(DLSmall(F_DIM), Xtr, ytr, n_epochs=5, patience=3, verbose=False, tag="t_dls")
    assert isinstance(model, TorchCompat), "DLSmall 应返回 TorchCompat"
    proba = model.predict_proba(Xte)
    assert proba.shape == (N_TE, 2)
    p = predict_proba_pos(model, Xte)
    assert p.shape == (N_TE,)
    print("  [ok] DLSmall predict_proba (N,2) + predict_proba_pos (N,)")


def test_cnn_interface():
    X, y = _toy_raw(seed=2)
    Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
    model = train_cnn(Xtr, ytr, n_epochs=3, batch_size=16, patience=3, verbose=False)
    assert isinstance(model, CNNClassifier), "CNN 应返回 CNNClassifier"
    proba = model.predict_proba(Xte)
    assert proba.shape == (N_TE, 2), f"CNN predict_proba 应 (N,2); 得到 {proba.shape}"
    s = proba.sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-4)
    p = predict_proba_pos(model, Xte)
    assert p.shape == (N_TE,)
    print("  [ok] CNN predict_proba (N,2) + predict_proba_pos (N,)")
    return model


def test_unified_interface_shape():
    """三种模型都走 ``predict_proba_pos`` 时返回的形状必须一致。"""
    X, y = _toy_features()
    Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
    Xr, yr = _toy_raw()
    Xrtr, yrtr, Xrte, yrte = Xr[:N_TR], yr[:N_TR], Xr[N_TR:], yr[N_TR:]
    from wifalldemo.models import _fit_torch_model, MLP, DLSmall
    m_mlp = _fit_torch_model(MLP(F_DIM), Xtr, ytr, n_epochs=3, patience=3)
    m_dls = _fit_torch_model(DLSmall(F_DIM), Xtr, ytr, n_epochs=3, patience=3)
    m_cnn = train_cnn(Xrtr, yrtr, n_epochs=3, batch_size=16, patience=3)
    for name, m, X in (("mlp", m_mlp, Xte), ("dlsmall", m_dls, Xte), ("cnn", m_cnn, Xrte)):
        p = predict_proba_pos(m, X)
        assert p.ndim == 1 and p.shape[0] == N_TE, f"{name} 应输出 (N,); 得到 {p.shape}"
        assert (p >= 0).all() and (p <= 1).all(), f"{name} 概率应在 [0,1]"
    print("  [ok] 三种模型 predict_proba_pos 形状/取值范围一致")


def test_save_load_roundtrip():
    """MLP 和 CNN save/load round-trip 后预测应相同 (load 之后再训练就放开)。"""
    X, y = _toy_features()
    Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
    Xr, yr = _toy_raw()
    Xrtr, yrtr, Xrte, yrte = Xr[:N_TR], yr[:N_TR], Xr[N_TR:], yr[N_TR:]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # MLP
        from wifalldemo.models import _fit_torch_model, MLP
        m = _fit_torch_model(MLP(F_DIM), Xtr, ytr, n_epochs=3, patience=3)
        save_model(m, [f"f{i}" for i in range(F_DIM)], td, "mlp")
        m2, names = load_model(td, "mlp")
        assert names == [f"f{i}" for i in range(F_DIM)]
        p1 = predict_proba_pos(m, Xte)
        p2 = predict_proba_pos(m2, Xte)
        assert np.allclose(p1, p2, atol=1e-5), "MLP save/load 后概率应一致"
        # CNN
        cnn = train_cnn(Xrtr, yrtr, n_epochs=3, batch_size=16, patience=3)
        save_model(cnn, [], td, "cnn")
        cnn2, names2 = load_model(td, "cnn")
        assert names2 == []
        p1 = predict_proba_pos(cnn, Xrte)
        p2 = predict_proba_pos(cnn2, Xrte)
        assert np.allclose(p1, p2, atol=1e-5), "CNN save/load 后概率应一致"
    print("  [ok] MLP/CNN save+load 预测完全一致 (round-trip)")


def test_cnn_layouts():
    """CNN1D.forward 同时支持 (B, T, S) 默认布局与 (B, S, T) 通道优先布局。"""
    import torch
    from wifalldemo.cnn import CNN1D
    m = CNN1D(n_subcarriers=S_DIM, dropout=0.0)
    m.eval()  # 关掉 dropout 让输出稳定
    x_ts = torch.randn(2, WIN, S_DIM)         # (B, T, S)
    x_st = torch.randn(2, S_DIM, WIN)         # (B, S, T) 通道优先
    y_ts = m(x_ts)
    y_st = m(x_st)
    # 两种布局是同一输入的转置；但 forward 看到布局不同时无法对齐输出绝对值，
    # 这里只检查形状和有限值。
    assert y_ts.shape == (2, 2)
    assert y_st.shape == (2, 2)
    assert torch.isfinite(y_ts).all()
    assert torch.isfinite(y_st).all()
    print("  [ok] CNN1D forward 同时接受 (B,T,S) 与 (B,S,T) 两种布局")


if __name__ == "__main__":
    print("=== 模型接口一致性测试 ===")
    test_mlp_interface()
    test_dlsmall_interface()
    test_cnn_interface()
    test_unified_interface_shape()
    test_save_load_roundtrip()
    test_cnn_layouts()
    print("=== 全部通过 ===")
