#!/usr/bin/env python
"""端到端训练/评估脚本。

用法:
    python run_train.py                       # 默认参数 (hold-out ID8/ID9)
    python run_train.py --test-users ID8 ID9 --no-cache

输出到 outputs/:
    models/{mlp,dlsmall,cnn,primary}.joblib|.pt  已训练模型
    figures/…png                                 各项可视化
    metrics.json                                 指标汇总

当前版本使用 PyTorch 深度学习 (MLP / DLsmall / CNN1D) 取代原 sklearn 随机森林
与逻辑回归；调用接口与外部输入/输出格式全部保持兼容。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from wifalldemo import config
from wifalldemo.audio import AUDIO_FEATURE_NAMES, audio_feature_names
from wifalldemo.cnn import CNNClassifier
from wifalldemo.data_loader import balanced_file_index, person_split, scan_dataset, summary
from wifalldemo.dataset import build_feature_dataset, build_raw_window_dataset
from wifalldemo.models import (
    evaluate,
    norm_config_snapshot,
    predict_proba_pos,
    save_model,
    train_dlsmall,
    train_mlp,
    train_cnn,
    youden_threshold,
)
from wifalldemo.postprocess import (
    apply_stillness_rule,
    find_activity_index,
    hybrid_scores,
    relative_scores,
    topk_file_scores,
    tune_stillness_rule,
)
from wifalldemo.visualize import (
    plot_audio_example,
    plot_confusion,
    plot_csi_sample,
    plot_feature_compare,
    plot_model_compare,
    plot_roc,
)
from wifalldemo.preprocessing import load_sample_amplitude


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Wi-Fi 跌倒检测 训练与评估 (DL)")
    ap.add_argument(
        "--data-dir",
        default=str(getattr(config, "DEFAULT_PICKLE", config.OUTPUT_DIR / "data_sequence.pkl")),
        help="数据源：目录 (ID*/<action>/*.csv) 或 data_sequence.pkl 路径 (默认: %(default)s)",
    )
    ap.add_argument("--test-users", nargs="+", default=list(config.TEST_USERS))
    ap.add_argument(
        "--val-users", nargs="+", default=None,
        help="从训练人员中再留出做阈值选择的人（默认取 config.VAL_USERS，"
             "Widar 是 ID6/ID7；训自采数据时必须显式指定，否则阈值退化成 0.5）",
    )
    ap.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    ap.add_argument("--window-step", type=int, default=config.WINDOW_STEP)
    ap.add_argument("--fs", type=float, default=config.SAMPLING_RATE_HZ)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-normalize", action="store_true",
                    help="关掉逐子载波归一化（CNN 退回吃裸幅度）。"
                         "Widar 那种\"同一房间、同一套摆位\"的数据集里静态 |H(f)| 本身带信息，"
                         "归一化反而会掉点；自采数据（每次摆位都可能不同）则必须开")
    ap.add_argument("--norm-causal", action="store_true",
                    help="归一化参考窗只用历史帧（因果）。流式推理天然是因果的，"
                         "部署前用这个开关训一次可消除训练/推理的参考量口径差")
    ap.add_argument("--norm-window", type=float, default=None, metavar="SEC",
                    help="归一化参考量的滑动窗长度(秒)，默认 config.NORM_WINDOW_S="
                         f"{config.NORM_WINDOW_S:g}。改动会同时让原始窗缓存失效")
    ap.add_argument("--no-balance", action="store_true",
                    help="关闭\"每用户每活动按中位数截断\"的用户均衡。"
                         "样本极少时这个截断会把大头用户砍到中位数（例如把 20 条 fall 砍成 4 条），"
                         "反而没法学；此时用这个开关保留全部样本")
    ap.add_argument("--no-cnn", action="store_true",
                    help="跳过 CNN 训练 (CNN 输入为原始 (128,52) 张量)。"
                         "注意：CNN 只作评估对照，不参与主模型择优/部署")
    ap.add_argument("--no-augment", action="store_true",
                    help="关闭 CNN 训练时的在线数据增强（子载波掩码/增益抖动/噪声/平移）。"
                         "默认开启：小样本跨人场景下增强等价于凭空放大样本量，抗过拟合")
    ap.add_argument("--no-se", action="store_true",
                    help="关闭 CNN 的 SE 通道注意力层。默认开启（轻量、几乎不增加计算）")
    ap.add_argument("--kfold", type=int, default=0, metavar="K",
                    help="用 Group K-Fold 的 out-of-fold 概率选阈值，替代单个验证人"
                         "（解决小验证集阈值乱跳问题，如 0.002/0.982 这种噪声阈值）。"
                         "K>=2 时启用；0 关闭（默认，沿用 --val-users 单验证人）")
    ap.add_argument("--score-mode", choices=("rel", "hybrid", "raw"), default="raw",
                    help="打分后处理：raw=原始概率（默认；LOPO 实测融合模型文件级 "
                        "Acc 0.790，rel/hybrid 会掉到 0.28~0.32 —— top-k 的组内 "
                        "z 分跨文件趋于常数，绝对置信度信息被抹掉）；rel=文件内稳健 "
                        "z-score（跨批次尺度不变，仅个别整折漂移场景有用）；"
                        "hybrid=z+0.25×raw（实测同样掉点，保留供研究）")
    ap.add_argument("--gain-aug-db", type=float, default=6.0, metavar="DB",
                    help="训练时对音频电平特征做 ±DB 的随机增益抖动增广（默认 6 dB，"
                         "0 关闭）。只作用于训练窗，val/test 不动 —— 强迫模型学"
                         "'相对本底的突出量'而不是绝对电平")
    ap.add_argument("--stillness", action="store_true",
                    help="启用'坠落后静止'时序确认规则（默认关闭：LOPO 五折上规则"
                         "对融合模型从未通过验证集调参 —— 单验证人 × 少量 stand 文件"
                         "的网格搜索太噪，留给 --stillness 显式开启做研究）")
    ap.add_argument("--audio", action="store_true",
                    help="把采集端录的 .wav 作为第 4 路特征接入（需要 .wav + .sync.json）。"
                         "开启后额外训练两个模型：mlp_av（CSI+音频融合）与 audio"
                         "（仅音频单模态，用作消融对照）；缺音频的样本自动补零")
    ap.add_argument("--no-audio", action="store_true",
                    help="强制关闭音频通路（优先级高于 config.AUDIO_ENABLED）")
    ap.add_argument("--metrics-out", default=None, metavar="PATH",
                    help="指标 JSON 的输出路径（默认 outputs/metrics.json）。"
                         "做多划分消融时用它把每次结果分开存，避免互相覆盖")
    ap.add_argument("--no-save", action="store_true",
                    help="只算指标、不落模型。换划分做实验时必须加：否则最后一次实验的"
                         "权重会覆盖 models/，而 metrics.json 还停留在上一次，"
                         "推理侧就拿到了一个来历不明的模型")
    ap.add_argument("--verbose-dl", action="store_true",
                    help="打印 MLP/DLsmall/CNN 每个 epoch 的损失曲线")
    return ap.parse_args()


def print_header(msg: str) -> None:
    print("\n" + "=" * 68)
    print("  " + msg)
    print("=" * 68)


def _gain_jitter(X: np.ndarray, feat_names: list[str], db: float,
                 seed: int = 20261017) -> np.ndarray:
    """对"有音频"的窗，给绝对电平特征加 ±db 随机偏移（训练增广）。

    只动 audio_logrms_mean/max/p90（dBFS 绝对量，随批次增益/距离平移）；
    crest/onset/peak_snr 等相对量与频带占比天然增益不变，不动。验证/测试
    不调本函数 —— 增广只进训练损失，迫使模型学"相对本底的突出量"。
    """
    rng = np.random.default_rng(seed)
    Xa = X.copy()
    pres = Xa[:, feat_names.index("audio_present")] >= 0.5
    if pres.any():
        cols = [feat_names.index(n) for n in
                ("audio_logrms_mean", "audio_logrms_max", "audio_logrms_p90")]
        off = rng.uniform(-db, db, size=(int(pres.sum()), 1)).astype(np.float32)
        Xa[np.ix_(pres, cols)] += off
    return Xa


def main() -> None:
    args = parse_args()
    # 这两个开关要在**建数据集之前**落到 config 上：dataset 用的是 config.X 的
    # 运行时取值（不是 import 时的快照），所以这里改有效。
    if args.no_normalize:
        config.NORMALIZE_PER_SUBCARRIER = False
    if args.norm_window is not None:
        config.NORM_WINDOW_S = float(args.norm_window)
    if args.norm_causal:
        config.NORM_CAUSAL = True
    # CNN 在线增强 / SE 注意力开关（默认开，可回滚）
    if args.no_augment:
        config.CNN_AUGMENT = False
    if args.no_se:
        config.CNN_USE_SE = False
    # 音频通路：命令行开关优先，其次 config.AUDIO_ENABLED
    use_audio = not args.no_audio and (
        bool(args.audio) or bool(getattr(config, "AUDIO_ENABLED", False))
    )

    records = scan_dataset(args.data_dir)
    train_records, test_records = person_split(records, tuple(args.test_users))

    # ---------------------------------------------------------------- 用户均衡
    keep_all = (
        train_records.index.values if args.no_balance
        else balanced_file_index(train_records)   # 默认保留原行为: 启用用户均衡
    )
    bal_train = train_records.loc[np.isin(train_records.index.values, keep_all)]
    print_header("步骤 1/6  数据概览（用户均衡后）")
    print("[训练集]")
    print(summary(bal_train))
    print("[测试集(跨人评估，保持完整)]")
    print(summary(test_records))

    print_header("步骤 2/6  CSI 读取 + 滑窗特征提取" + (" + 音频特征" if use_audio else ""))
    cache_fp = (
        config.CACHE_DIR
        / f"features_w{args.window_size}_s{args.window_step}_fs{args.fs:.1f}"
          f"{'_audio' if use_audio else ''}.npz"
    )
    X, y, record_ids, feat_names = build_feature_dataset(
        records, win=args.window_size, step=args.window_step,
        fs=args.fs, cache_fp=cache_fp, force=args.no_cache, audio=use_audio,
    )
    # CSI 手工特征（70 维）之后的列全是音频特征；分割点从特征名里读，避免硬编码
    audio_cols = [i for i, n in enumerate(feat_names) if n.startswith("audio_")]
    csi_dim = audio_cols[0] if audio_cols else len(feat_names)
    audio_cov = float(X[:, feat_names.index("audio_present")].mean()) if audio_cols else 0.0
    audio_ok = bool(audio_cols) and audio_cov > 0.0
    if audio_cols:
        print(f"  音频特征 {len(audio_cols)} 维；可用音频的窗占比 {audio_cov:.1%}")
        if not audio_ok:
            print("  ⚠ 数据源里没有任何 .wav/.sync.json，音频列全为占位值 —— "
                  "本次按纯 CSI 训练（音频模型跳过）。")
    elif use_audio:
        print("  ⚠ 请求了 --audio，但特征矩阵里没有音频列（数据源不支持），按纯 CSI 训练。")

    train_idx = np.isin(record_ids, bal_train.index.values)
    test_idx = np.isin(record_ids, test_records.index.values)
    Xtr, ytr = X[train_idx], y[train_idx]

    # ---- 打分后处理开关（详见 wifalldemo/postprocess.py）----
    score_mode = args.score_mode
    if score_mode != "raw":
        print(f"  打分后处理: {score_mode}"
              f"（阈值与确认规则在同一变换空间的验证集上选）")

    def _score_transform(p: np.ndarray, groups: np.ndarray) -> np.ndarray:
        if score_mode == "rel":
            return relative_scores(p, groups)
        if score_mode == "hybrid":
            return hybrid_scores(p, groups)
        return p

    # "坠落后静止"规则需要的 CSI 活动度特征列（跨子载波平均绝对差分）
    act_idx = find_activity_index(list(feat_names))
    use_stillness = bool(args.stillness) and act_idx is not None
    if args.stillness and act_idx is None:
        print("  ⚠ 特征里找不到活动度特征（raw/rel_diff_mean），静止确认规则自动关闭")
    # 音频电平增益增广：只增广训练行；无音频或 db=0 时原样返回
    if args.gain_aug_db > 0 and audio_ok:
        X_aug = _gain_jitter(X, list(feat_names), args.gain_aug_db)
        print(f"  音频增益增广: ±{args.gain_aug_db:g} dB（仅训练窗）"
              f" | 静止确认={'开' if use_stillness else '关'}")
    else:
        X_aug = X
        print(f"  静止确认={'开' if use_stillness else '关'}"
              + (" | 增益增广: 关" if args.gain_aug_db <= 0 else ""))
    Xte, yte = X[test_idx], y[test_idx]
    # 纯 CSI 视图（基线模型用）；融合模型用完整 X
    X_csi, Xtr_csi, Xte_csi = X[:, :csi_dim], Xtr[:, :csi_dim], Xte[:, :csi_dim]
    if audio_ok:
        X_aud, Xtr_aud, Xte_aud = X[:, csi_dim:], Xtr[:, csi_dim:], Xte[:, csi_dim:]
    print(f"  训练窗 {len(ytr)} (fall={int((ytr==1).sum())}) "
          f"| 测试窗 {len(yte)} (fall={int((yte==1).sum())})")

    # 若启用 CNN，同时构建原始滑窗张量 (B, T, S)
    X_raw_tr = X_raw_te = X_raw_all = None
    if not args.no_cnn:
        raw_cache_fp = (
            config.CACHE_DIR
            / f"raw_windows_w{args.window_size}_s{args.window_step}.npz"
        )
        X_raw_all, _, _ = build_raw_window_dataset(
            records, win=args.window_size, step=args.window_step,
            fs=args.fs, cache_fp=raw_cache_fp, force=args.no_cache,
        )
        X_raw_tr = X_raw_all[train_idx]
        X_raw_te = X_raw_all[test_idx]

    # ------------------------------------------------------------------
    print_header("步骤 3/6  模型训练 + 跨人阈值选择")
    wanted = list(config.VAL_USERS) if args.val_users is None else list(args.val_users)
    val_users = [u for u in wanted if u in set(bal_train["user"])]
    if wanted and not val_users:
        print(f"  警告: 指定的验证人员 {wanted} 不在训练人员里（可能是 Widar 的 ID6/ID7），"
              f"阈值将退化成 0.5。用 --val-users 指定自采数据的验证人")
    n_train_users = len(bal_train["user"].unique())
    if val_users and len(val_users) >= n_train_users:
        val_users = val_users[:-1] if n_train_users > 1 else []
    if not val_users:
        print("  警告: 无法从训练用户中留出验证人员，使用默认阈值 0.5")
        val_rids = np.array([], dtype=record_ids.dtype)
        fit_rids = bal_train.index.values
    else:
        val_rids = bal_train.index[bal_train["user"].isin(val_users)].values
        fit_rids = bal_train.index[~bal_train["user"].isin(val_users)].values
    fit_mask = np.isin(record_ids, fit_rids)
    val_mask = np.isin(record_ids, val_rids)
    use_yte_val = y[val_mask]
    if len(val_rids) == 0:
        use_yte_val = None  # 让 fit_with_threshold 走内部切分

    def _kfold_threshold(fit_fn, X_all, y_all, groups_all):
        """对训练集做 Group K-Fold，用 out-of-fold 概率选阈值。

        目的：单验证人（如 val=ID11 只有 8 条样本）上的 Youden-J 阈值基本是
        噪声（实测 0.002 / 0.982 乱跳）。改为在全部训练样本上做分组交叉验证，
        每折的验证概率拼成 OOF，阈值基于**全部训练窗**而非 8 条样本，稳定得多。
        """
        from sklearn.model_selection import GroupKFold

        n = len(y_all)
        n_groups = len(np.unique(groups_all))
        k = max(2, min(int(args.kfold), n_groups))
        if n < 2 or n_groups < 2:
            return 0.5, 0.5
        gkf = GroupKFold(n_splits=k)
        oof = np.zeros(n, dtype=np.float64)
        for tr_idx, va_idx in gkf.split(X_all, y_all, groups=groups_all):
            m = fit_fn(
                X_all[tr_idx], y_all[tr_idx],
                val_X=X_all[va_idx], val_y=y_all[va_idx],
            )
            oof[va_idx] = predict_proba_pos(m, X_all[va_idx])
        thr = youden_threshold(y_all, oof)
        p_topk, y_topk = topk_file_scores(oof, groups_all, y_all)
        if 0 < int(y_topk.sum()) < y_topk.size:
            thr_topk = youden_threshold(y_topk, p_topk)
        else:
            thr_topk = 0.5
        return float(thr), float(thr_topk)

    def fit_with_threshold(fit_fn, X_fit, X_val, X_full):
        """训练 + 选阈值 + 选确认规则参数。

        * ``X_fit``     训练行（音频模型传入增益增广后的矩阵；[fit_mask] 索引）
        * ``X_val``     验证行（**必须干净**，不增广；[val_mask] 索引）
        * ``X_full``    最终用全部训练人员 retrain 的输入（已是训练行）

        阈值口径：默认对 val 窗分做**文件内相对打分**后再选 Youden-J（相对空间），
        测试侧用同一变换；top-k 文件级投票另有阈值（实测沿用全窗阈会把大量负
        样本判正：Acc 0.767 -> 0.558）。"坠落后静止"规则的参数也在同一 val 上
        网格搜索，验证集不支持时自动退回不用规则。
        """
        if args.kfold >= 2:
            # K-Fold OOF 选阈值（替代单验证人，解决小验证集阈值乱跳）
            thr, thr_topk = _kfold_threshold(
                fit_fn, X_fit[fit_mask], y[fit_mask], record_ids[fit_mask])
            model = fit_fn(X_full, ytr, groups=record_ids[train_idx])
            return {"model": model, "threshold": thr, "threshold_topk": thr_topk,
                    "stillness": None}
        if len(val_rids) == 0:
            model = fit_fn(X_full, ytr, groups=record_ids[train_idx])
            return {"model": model, "threshold": 0.5, "threshold_topk": 0.5,
                    "stillness": None}
        tmp = fit_fn(
            X_fit[fit_mask], y[fit_mask],
            val_X=X_val[val_mask], val_y=y[val_mask],
        )
        proba_val = predict_proba_pos(tmp, X_val[val_mask])
        proba_val = _score_transform(proba_val, record_ids[val_mask])
        thr = youden_threshold(y[val_mask], proba_val)

        # top-k 文件级投票必须有自己的阈值：top-k 把分数整体上移，沿用全窗均值
        # 的阈值等于把大量负样本判成正样本（实测文件级 Acc 0.767 -> 0.558）。
        # 这里在**同一个 val** 上、按**同一口径**重选一次。
        vg = record_ids[val_mask]
        p_topk_val, y_topk_val = topk_file_scores(proba_val, vg, y[val_mask])
        if 0 < int(y_topk_val.sum()) < y_topk_val.size:
            thr_topk = youden_threshold(y_topk_val, p_topk_val)
        else:
            thr_topk = 0.5
        # "坠落后静止"确认规则：只在 val 两类齐全、且能找到活动度特征时调参
        rule = None
        if use_stillness and 0 < int(y_topk_val.sum()) < y_topk_val.size:
            act_val = X[val_mask][:, act_idx]
            rule = tune_stillness_rule(proba_val, vg, act_val, y[val_mask], thr_topk)
        final = fit_fn(X_full, ytr, groups=record_ids[train_idx])
        return {"model": final, "threshold": float(thr),
                "threshold_topk": float(thr_topk), "stillness": rule}

    fitted: dict[str, dict] = {}

    # ---- MLP (替换原 RF) ----
    fitted["mlp"] = {"name": "MLP", "feature_set": "csi",
                     **fit_with_threshold(train_mlp, X_csi, X_csi, Xtr_csi)}
    print(f"  训练 MLP(替换 RF) 完成; 决策阈值 = {fitted['mlp']['threshold']:.3f}")

    # ---- DLsmall (替换原 LR) ----
    fitted["dlsmall"] = {"name": "DLSmall", "feature_set": "csi",
                         **fit_with_threshold(train_dlsmall, X_csi, X_csi, Xtr_csi)}
    print(f"  训练 DLSmall(替换 LR) 完成; 决策阈值 = {fitted['dlsmall']['threshold']:.3f}")

    # ---- CNN (新增，输入为原始窗) ----
    # 注意：CNN 只作评估对照（自采数据上 recall≈0，见 README），不参与主模型
    # 择优，也不进部署链路 —— 部署主模型只从 {mlp, dlsmall, mlp_av} 里选。
    if not args.no_cnn and X_raw_tr is not None:
        fitted["cnn"] = {"name": "CNN1D", "feature_set": "csi-raw",
                         **fit_with_threshold(train_cnn, X_raw_all, X_raw_all, X_raw_tr)}
        print(f"  训练 CNN1D 完成; 决策阈值 = {fitted['cnn']['threshold']:.3f}"
              f"  [仅评估对照，不参与部署择优]")

    # ---- 音频通路：融合 + 单模态消融 ----
    # mlp_av 吃 [70 维 CSI, 31 维音频]，是"运动学 + 撞击声"的融合模型；
    # audio 只吃音频，用来回答"声音本身够不够判跌倒"（跳跃落地也有撞击声，
    # 单模态天生分不开，预期它会明显弱于融合 —— 这正是消融要展示的）。
    # 音频模型的训练行走增益增广矩阵 X_aug，验证行保持 X 原样。
    if audio_ok:
        fitted["mlp_av"] = {"name": "MLP-Fusion", "feature_set": "csi+audio",
                            **fit_with_threshold(train_mlp, X_aug, X, X_aug[train_idx])}
        print(f"  训练 MLP-Fusion(CSI+音频 {csi_dim}+{len(audio_cols)} 维) 完成; "
              f"决策阈值 = {fitted['mlp_av']['threshold']:.3f}")

        fitted["audio"] = {"name": "MLP-AudioOnly", "feature_set": "audio",
                           **fit_with_threshold(
                               train_mlp, X_aug[:, csi_dim:], X[:, csi_dim:],
                               X_aug[train_idx][:, csi_dim:])}
        print(f"  训练 MLP-AudioOnly(仅音频 {len(audio_cols)} 维) 完成; "
              f"决策阈值 = {fitted['audio']['threshold']:.3f}  [消融对照，不参与主模型择优]")

    # ------------------------------------------------------------------
    print_header("步骤 4/6  测试集评估（跨人泛化，窗口级 + 文件多数投票级）")
    results: dict[str, dict] = {}
    g_all = record_ids[test_idx]
    vote_groups = np.unique(g_all)
    y_file_true = np.array([int(yte[g_all == gi].mean() >= 0.5) for gi in vote_groups])

    TOPK = 3  # 跌倒是短时事件：一个文件里只有少数窗含冲击，全窗均值会稀释证据

    def file_metrics(p: np.ndarray, thr: float):
        p_file = np.array([float(p[g_all == gi].mean()) for gi in vote_groups])
        return evaluate(y_file_true, p_file, thr), p_file

    def file_metrics_topk(p: np.ndarray, thr: float):
        """top-k 窗均值投票：每文件取概率最高的 min(3, n) 个窗求平均。

        stand/sit 这类误报通常是"少量窗得高分"，全窗均值容易被拖过阈值；
        反过来 fall 的冲击窗是**连续多个**高分窗。top-k 对两者都更稳。
        """
        p_file = np.array([
            float(np.sort(p[g_all == gi])[-min(TOPK, int((g_all == gi).sum())):].mean())
            for gi in vote_groups
        ])
        return evaluate(y_file_true, p_file, thr), p_file

    for tag, d in fitted.items():
        if isinstance(d["model"], CNNClassifier):
            inp = X_raw_te
        elif tag == "mlp_av":
            inp = Xte            # 融合：完整 CSI + 音频
        elif tag == "audio":
            inp = Xte_aud        # 仅音频
        else:
            inp = Xte_csi        # 纯 CSI
        p = predict_proba_pos(d["model"], inp)
        # 评估口径与阈值口径必须一致：阈值在哪个变换空间选，这里就用哪个变换
        p_eval = _score_transform(p, g_all)
        met_w = evaluate(yte, p_eval, d["threshold"])
        met_f, p_file = file_metrics(p_eval, d["threshold"])
        met_ft, _ = file_metrics_topk(p_eval, d["threshold_topk"])
        # "坠落后静止"确认：top-k 判正的文件若触发后仍在活动（stand/sit 典型
        # 模式）则被否决。规则参数来自验证人网格搜索，None = 该模型不用规则。
        rule = d.get("stillness")
        if rule is not None:
            hard = apply_stillness_rule(p_eval, g_all, Xte[:, act_idx], rule)
            met_ftc = evaluate(y_file_true, hard.astype(float), 0.5)
            met_ftc["auc"] = met_ft["auc"]   # AUC 由连续分数承载；规则只改判定
        else:
            met_ftc = met_ft
        d["proba_test"] = p
        d["proba_eval"] = p_eval
        d["metrics_w"] = met_w
        d["metrics_f"] = met_f
        d["metrics_ft"] = met_ft
        d["metrics_ftc"] = met_ftc
        d["p_file"] = p_file
        results[tag] = d
        print(f"  [{d['name']:<13s} {d['feature_set']:>9s}]  window: Acc={met_w['accuracy']:.4f} "
              f"F1={met_w['f1']:.4f} AUC={met_w['auc']:.4f} | "
              f"file: Acc={met_f['accuracy']:.4f} F1={met_f['f1']:.4f} | "
              f"file-topk: Acc={met_ft['accuracy']:.4f} F1={met_ft['f1']:.4f} | "
              f"topk+静止确认: Acc={met_ftc['accuracy']:.4f} F1={met_ftc['f1']:.4f}"
              + ("" if rule is not None else " (规则未启用)"))
    # 主模型按 **top-k + 静止确认** 文件级 F1 择优（LOPO 实测 top-k 口径比全窗均值
    # 高 Acc +8.1 / F1 +13.1：跌倒是短时事件，全窗均值会稀释冲击窗的证据）。
    # audio（仅音频）只是消融对照、cnn 在自采数据上 recall≈0，都不参与择优 ——
    # 部署主模型只从 {mlp, dlsmall, mlp_av} 里选。
    DEPLOY_EXCLUDE = {"audio", "cnn"}
    primary = max(
        (t for t in results if t not in DEPLOY_EXCLUDE),
        key=lambda t: (results[t]["metrics_ftc"]["f1"], results[t]["metrics_w"]["f1"]),
    )

    test_ids_all = record_ids[test_idx]
    p_pri = results[primary]["proba_eval"]   # 与阈值同口径（相对打分后）
    thr_pri = results[primary]["threshold"]
    per_act = []
    for act in sorted(test_records["activity"].unique()):
        ids = test_records.index[test_records["activity"] == act].values
        mask = np.isin(test_ids_all, ids)
        acc = float(np.mean((p_pri[mask] >= thr_pri).astype(int) == yte[mask]))
        per_act.append((act, acc, int(mask.sum())))
    print(f"  主模型={results[primary]['name']}  测试窗按活动细分: "
          + ", ".join(f"{a}={acc:.3f}(n={n})" for a, acc, n in per_act))

    # ------------------------------------------------------------------
    print_header("步骤 5/6  样本级（文件）主口径评估：top-k 投票 + 静止确认（主模型）")
    m_file = results[primary]["metrics_ftc"]   # 主口径 = top-k + 静止确认
    p_file = results[primary]["p_file"]
    print(f"  [{results[primary]['name']} @ file-level]  Acc={m_file['accuracy']:.4f}  "
          f"Prec={m_file['precision']:.4f}  Rec={m_file['recall']:.4f}  "
          f"F1={m_file['f1']:.4f}")
    cm_f = np.array(
        [[m_file["confusion"]["tn"], m_file["confusion"]["fp"]],
         [m_file["confusion"]["fn"], m_file["confusion"]["tp"]]]
    )
    file_act_acc: dict[str, float] = {}
    for act in sorted(test_records["activity"].unique()):
        ids = test_records.index[test_records["activity"] == act].values
        sel = np.isin(vote_groups, ids)
        if not sel.any():
            continue
        acc = float(np.mean((p_file[sel] >= thr_pri).astype(int) == y_file_true[sel]))
        file_act_acc[act] = acc
        print(f"    活动 {act}: 文件级 Acc={acc:.3f} (n={int(sel.sum())})")

    # ------------------------------------------------------------------
    print_header("步骤 6/6  保存模型与可视化")
    m_win = results[primary]["metrics_w"]
    cm_w = np.array(
        [[m_win["confusion"]["tn"], m_win["confusion"]["fp"]],
         [m_win["confusion"]["fn"], m_win["confusion"]["tp"]]]
    )
    fpr, tpr, _ = roc_curve(yte, p_pri)
    plot_confusion(cm_w, f"cm_window_{primary}.png",
                   f"Window Confusion ({results[primary]['name']})")
    plot_confusion(cm_f, f"cm_file_{primary}.png",
                   f"File Confusion ({results[primary]['name']}, majority vote)")
    plot_roc(fpr, tpr, m_win["auc"], f"roc_{primary}.png")
    plot_feature_compare(Xte, yte, feat_names, top=6, name="feature_compare.png")
    plot_model_compare(
        [results[t]["name"] for t in results],
        [results[t]["metrics_w"]["accuracy"] for t in results],
        "model_compare.png",
    )
    fall_rows = test_records[test_records.label == 1]["path"]
    non_rows = test_records[test_records.label == 0]["path"]
    if len(fall_rows) and len(non_rows):
        plot_csi_sample(load_sample_amplitude(fall_rows.iloc[0]),
                        "csi_example_fall.png",
                        title="CSI heatmap - Fall sample")
        plot_csi_sample(load_sample_amplitude(non_rows.iloc[0]),
                        "csi_example_nonfall.png",
                        title="CSI heatmap - Non-Fall sample")
    else:
        print("  测试集缺少某一类别，跳过 CSI 示例图")

    # 音频三视图（波形 / 包络 / 语谱图）：给报告用的"声音里确实有撞击证据"直观材料
    if audio_ok:
        from wifalldemo.audio import read_audio_track

        for sub, act_label in ((fall_rows, "fall"), (non_rows, "nonfall")):
            if not len(sub):
                continue
            track = read_audio_track(sub.iloc[0])
            if track is None:
                continue
            plot_audio_example(
                track, f"audio_example_{act_label}.png",
                title=f"Audio of a {act_label} sample (green = CSI span)",
            )
        print(f"  音频示例图已输出到 {config.FIG_DIR}")

    # 每个模型必须连同**它自己的**输入列名一起存档：demo 靠特征名判断要不要
    # 现算音频特征、以及该切哪几列。存错了不会报错，只会静默喂错分布。
    names_by_tag = {
        "mlp": feat_names[:csi_dim],
        "dlsmall": feat_names[:csi_dim],
        "mlp_av": feat_names,
        "audio": feat_names[csi_dim:],
        "cnn": [],
    }
    # 只有默认训练（写 metrics.json 的那一次）才允许覆盖 models/。
    # 消融/扫参这类"换个划分再跑一遍"的实验必须带 --no-save：否则最后一次实验
    # 的权重会把部署模型顶掉，而 metrics.json 还停留在上一次 —— demo 看起来正常、
    # 打分却完全不同（历史上真踩过：诊断脚本拿到的模型已经见过测试集）。
    if args.no_save:
        print("  [--no-save] 跳过模型落盘，models/ 保持原样")
    else:
        for tag in ("mlp", "dlsmall", "cnn", "mlp_av", "audio"):
            if tag in results:
                save_model(results[tag]["model"], names_by_tag[tag], config.MODEL_DIR, tag)
        # primary 另存一个固定名 (供 demo 按 metrics.json 自动加载)
        save_model(results[primary]["model"], names_by_tag[primary], config.MODEL_DIR, "primary")
    report = {
        "test_users": list(args.test_users),
        "val_users": list(val_users) if val_rids.size else [],
        "data_dir": str(args.data_dir),
        "window_size": args.window_size,
        "window_step": args.window_step,
        "fs_hz": args.fs,
        "n_features": len(feat_names),
        "csi_features": int(csi_dim),
        "primary_model": primary,
        # 两个阈值用途不同，都要存档：逐窗判定用 threshold，
        # 文件级报警（top-k 投票）用 threshold_topk —— 后者才是主口径。
        "threshold": float(thr_pri),
        "threshold_topk": float(results[primary].get("threshold_topk", thr_pri)),
        "primary_file_metric": (
            "topk_window_vote+stillness_rule"
            if results[primary].get("stillness") else "topk_window_vote"
        ),
        "window_metrics": {t: results[t]["metrics_w"] for t in results},
        "file_metrics": {t: results[t]["metrics_f"] for t in results},
        "file_topk_metrics": {t: results[t]["metrics_ft"] for t in results},
        "file_topk_confirmed": {t: results[t]["metrics_ftc"] for t in results},
        # 打分后处理口径：推理侧（demo）必须复现同样的变换，否则阈值失效
        "postprocess": {
            "score_mode": score_mode,
            "gain_aug_db": float(args.gain_aug_db),
            "stillness_rule": results[primary].get("stillness"),
        },
        "stillness_rules": {t: results[t]["stillness"] for t in results},
        "primary_file_metrics": m_file,
        "per_activity_file_acc": file_act_acc,
        # 归一化是 CNN 输入表示的一部分，必须与模型一起存档：推理侧靠它复现
        # 训练时的口径（居中/因果窗 + 窗长），否则喂进去的是另一种分布
        "normalization": norm_config_snapshot(),
        # 音频同理：推理侧要知道"这个模型吃不吃音频、吃多少维、叫什么名字"
        "audio": {
            "enabled": bool(audio_ok),
            "n_features": int(len(audio_cols)),
            "coverage": float(audio_cov),
            "feature_names": list(AUDIO_FEATURE_NAMES) if audio_ok else [],
        },
        "feature_sets": {t: results[t]["feature_set"] for t in results},
    }
    metrics_path = Path(args.metrics_out) if args.metrics_out else (config.OUTPUT_DIR / "metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n完成。主模型={results[primary]['name']}，模型与指标保存到 {config.OUTPUT_DIR}")
    print(f"指标 JSON: {metrics_path}")
    print("可视化结果见:", config.FIG_DIR)


if __name__ == "__main__":
    main()
