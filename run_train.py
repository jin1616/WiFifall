#!/usr/bin/env python
"""端到端训练/评估脚本。

用法:
    python run_train.py                       # 默认参数 (hold-out ID8/ID9)
    python run_train.py --test-users ID8 ID9 --no-cache

输出到 outputs/:
    models/rf.joblib, models/lr.joblib  已训练模型
    figures/…png                         各项可视化
    metrics.json                         指标汇总
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import roc_curve

from wifalldemo import config
from wifalldemo.data_loader import balanced_file_index, person_split, scan_dataset, summary
from wifalldemo.dataset import build_feature_dataset
from wifalldemo.models import (
    evaluate,
    predict_proba_pos,
    save_model,
    train_lr,
    train_rf,
    youden_threshold,
)
from wifalldemo.visualize import (
    plot_confusion,
    plot_csi_sample,
    plot_feature_compare,
    plot_feature_importance,
    plot_model_compare,
    plot_roc,
)
from wifalldemo.preprocessing import load_amplitude_series


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Wi-Fi 跌倒检测 训练与评估")
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--test-users", nargs="+", default=list(config.TEST_USERS),
                    help="留出的人员 ID（跨人评估）")
    ap.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    ap.add_argument("--window-step", type=int, default=config.WINDOW_STEP)
    ap.add_argument("--fs", type=float, default=config.SAMPLING_RATE_HZ)
    ap.add_argument("--no-cache", action="store_true", help="忽略特征缓存并重建")
    ap.add_argument("--no-lr", action="store_true", help="不训练逻辑回归基线")
    ap.add_argument("--no-balance", action="store_true",
                    help="关闭用户均衡(默认按每用户每动作的中位样本数截断)")
    return ap.parse_args()


def print_header(msg: str) -> None:
    print("\n" + "=" * 68)
    print("  " + msg)
    print("=" * 68)


def main() -> None:
    args = parse_args()
    records = scan_dataset(args.data_dir)
    train_records, test_records = person_split(records, tuple(args.test_users))

    # ---------------------------------------------------------------- 用户均衡
    # 每个 (user, activity) 文件数若高于全用户中位数则截断，避免单用户(如 ID2
    # 有 20 倍轮次)主导模型，从而提高跨人泛化。
    # 注意：截断用的中位数只在训练人员上统计，测试人员分布不得混入训练流程。
    keep_all = (
        balanced_file_index(train_records)
        if not args.no_balance
        else train_records.index.values
    )
    bal_train = train_records.loc[np.isin(train_records.index.values, keep_all)]
    print_header("步骤 1/6  数据概览（用户均衡后）")
    print("[训练集]")
    print(summary(bal_train))
    print("[测试集(跨人评估，保持完整)]")
    print(summary(test_records))

    print_header("步骤 2/6  CSI 读取 + 滑窗特征提取")
    cache_fp = (
        config.CACHE_DIR
        / f"features_w{args.window_size}_s{args.window_step}_fs{args.fs:.1f}.npz"
    )
    X, y, record_ids, feat_names = build_feature_dataset(
        records, win=args.window_size, step=args.window_step,
        fs=args.fs, cache_fp=cache_fp, force=args.no_cache,
    )
    train_idx = np.isin(record_ids, bal_train.index.values)
    test_idx = np.isin(record_ids, test_records.index.values)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    print(f"  训练窗 {len(ytr)} (fall={int((ytr==1).sum())}) "
          f"| 测试窗 {len(yte)} (fall={int((yte==1).sum())})")

    # ------------------------------------------------------------------
    print_header("步骤 3/6  模型训练 + 跨人阈值选择")
    # 阈值不用训练窗自评（会过拟合概率），而从"训练用户"里再留出验证人员
    # 用 Youden-J 准则挑选；选定后再用全部训练人员重训最终模型。
    val_users = [u for u in config.VAL_USERS if u in set(bal_train["user"])]
    n_train_users = len(bal_train["user"].unique())
    # 阈值选择需保证"排除验证用户后仍有用户可训练临时模型"；
    # 验证人员覆盖全部训练用户时削掉一个；只剩 1 名用户时无法跨人验证，
    # 置空让下方回退到默认阈值 0.5 并打印警告。
    if val_users and len(val_users) >= n_train_users:
        val_users = val_users[:-1] if n_train_users > 1 else []
    if not val_users:
        print("  警告: 无法从训练用户中留出验证人员，使用默认阈值 0.5")

    def fit_with_threshold(fit_fn):
        """返回 (最终模型, 决策阈值)。"""
        if not val_users:
            return fit_fn(Xtr, ytr), 0.5
        val_rids = bal_train.index[bal_train["user"].isin(val_users)].values
        fit_rids = bal_train.index[~bal_train["user"].isin(val_users)].values
        fit_mask = np.isin(record_ids, fit_rids)
        val_mask = np.isin(record_ids, val_rids)
        tmp = fit_fn(X[fit_mask], y[fit_mask])
        thr = youden_threshold(y[val_mask], predict_proba_pos(tmp, X[val_mask]))
        final = fit_fn(Xtr, ytr)          # 用全部训练人员重训一次
        return final, thr

    fitted: dict[str, dict] = {}
    rf, rf_thr = fit_with_threshold(train_rf)
    fitted["rf"] = {"name": "RandomForest", "model": rf, "threshold": rf_thr}
    print(f"  训练 RandomForest 完成; 决策阈值 = {rf_thr:.3f}")
    if not args.no_lr:
        lr, lr_thr = fit_with_threshold(train_lr)
        fitted["lr"] = {"name": "LogisticReg", "model": lr, "threshold": lr_thr}
        print(f"  训练 LogisticReg   完成; 决策阈值 = {lr_thr:.3f}")

    # ------------------------------------------------------------------
    print_header("步骤 4/6  测试集评估（跨人泛化，窗口级 + 文件多数投票级）")
    results: dict[str, dict] = {}
    g_all = record_ids[test_idx]
    vote_groups = np.unique(g_all)
    # 文件真实标签 = 窗内多数（与文件动作标签一致）
    y_file_true = np.array([int(yte[g_all == gi].mean() >= 0.5) for gi in vote_groups])

    def file_metrics(p: np.ndarray, thr: float):
        """窗概率 -> 文件平均概率，再做阈值判定得到文件级指标。"""
        p_file = np.array([float(p[g_all == gi].mean()) for gi in vote_groups])
        return evaluate(y_file_true, p_file, thr), p_file

    for tag, d in fitted.items():
        p = predict_proba_pos(d["model"], Xte)
        met_w = evaluate(yte, p, d["threshold"])
        met_f, p_file = file_metrics(p, d["threshold"])
        d["proba_test"] = p
        d["metrics_w"] = met_w
        d["metrics_f"] = met_f
        d["p_file"] = p_file
        results[tag] = d
        print(f"  [{d['name']}]  window: Acc={met_w['accuracy']:.4f} F1={met_w['f1']:.4f} "
              f"AUC={met_w['auc']:.4f} | file: Acc={met_f['accuracy']:.4f} F1={met_f['f1']:.4f}")
    # 主模型 = 文件级 F1 更高者（离线/实时演示与混淆矩阵默认使用它）
    primary = max(
        results,
        key=lambda t: (results[t]["metrics_f"]["f1"], results[t]["metrics_w"]["f1"]),
    )

    # 各活动细分的窗级准确率（主模型）
    test_ids_all = record_ids[test_idx]
    p_pri = results[primary]["proba_test"]
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
    print_header("步骤 5/6  样本级（文件）多数投票评估（主模型）")
    m_file = results[primary]["metrics_f"]
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
    plot_feature_importance(rf.feature_importances_, feat_names,
                            "feature_importance_rf.png")
    plot_feature_compare(Xte, yte, feat_names, top=6, name="feature_compare.png")
    plot_model_compare(
        [results[t]["name"] for t in results],
        [results[t]["metrics_w"]["accuracy"] for t in results],
        "model_compare.png",
    )
    fall_rows = test_records[test_records.label == 1]["path"]
    non_rows = test_records[test_records.label == 0]["path"]
    if len(fall_rows) and len(non_rows):
        plot_csi_sample(load_amplitude_series(fall_rows.iloc[0]),
                        "csi_example_fall.png",
                        title="CSI heatmap - Fall sample")
        plot_csi_sample(load_amplitude_series(non_rows.iloc[0]),
                        "csi_example_nonfall.png",
                        title="CSI heatmap - Non-Fall sample")
    else:
        print("  测试集缺少某一类别，跳过 CSI 示例图")

    save_model(rf, feat_names, config.MODEL_DIR, "rf")
    if not args.no_lr:
        save_model(lr, feat_names, config.MODEL_DIR, "lr")
    # primary 另存一个固定名，供 demo 脚本按 metrics.json 自动加载
    save_model(results[primary]["model"], feat_names, config.MODEL_DIR, "primary")

    report = {
        "test_users": list(args.test_users),
        "val_users": list(val_users),
        "window_size": args.window_size,
        "window_step": args.window_step,
        "fs_hz": args.fs,
        "n_features": len(feat_names),
        "primary_model": primary,
        "threshold": float(thr_pri),
        "window_metrics": {t: results[t]["metrics_w"] for t in results},
        "file_metrics": {t: results[t]["metrics_f"] for t in results},
        "primary_file_metrics": m_file,
        "per_activity_file_acc": file_act_acc,
    }
    with (config.OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n完成。主模型={results[primary]['name']}，模型与指标保存到 {config.OUTPUT_DIR}")
    print("可视化结果见:", config.FIG_DIR)


if __name__ == "__main__":
    main()
