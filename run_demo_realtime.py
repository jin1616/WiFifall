#!/usr/bin/env python
"""实时（流式）检测演示。

模拟从 ESP32/Widar3.0 采集端逐包接收 CSI：每来一个包就往环形缓冲推入，
每积累 WINDOW_STEP 个新包就基于最近 WINDOW_SIZE 个包在线提取特征并打分。
当 P(fall) 穿越阈值时打印报警（含模拟时间）。

这样演示了真实系统中可以嵌入的流式推理循环——只需把下方 read_packet()
替换成 socket/串口的采集回调即可接入真实设备。

模型支持 (v0.3.0):
  * MLP / DLSmall — 走 70 维手工特征通路
  * CNN1D        — 走 (T=128, S=52) 原始滑窗通路

用法:
    python run_train.py              # 先训练
    python run_demo_realtime.py      # 默认读 metrics.json.primary_model
    python run_demo_realtime.py --model cnn
    python run_demo_realtime.py --file <某条测试样本路径>
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from wifalldemo import config
from wifalldemo.audio import AUDIO_DIM, read_audio_track
from wifalldemo.cnn import CNNClassifier
from wifalldemo.data_loader import person_split, scan_dataset
from wifalldemo.features import window_feature_vector
from wifalldemo.models import load_model, load_norm_config, predict_proba_pos
from wifalldemo.preprocessing import iter_sample_rows
from wifalldemo.visualize import plot_detection_timeline


def main() -> None:
    ap = argparse.ArgumentParser(description="实时流式跌倒检测演示 (DL)")
    ap.add_argument("--model", default=None,
                    help="mlp / dlsmall / cnn / primary (默认读 metrics.json 择优结果)")
    ap.add_argument(
        "--file",
        default=None,
        help="测试样本 CSV；缺省自动选取 ID9 下的首个 fall 样本",
    )
    args = ap.parse_args()

    thr_path = config.OUTPUT_DIR / "metrics.json"
    tag = "mlp"
    meta: dict = {}
    if thr_path.exists():
        meta = json.loads(thr_path.read_text(encoding="utf-8"))
        threshold = float(meta.get("threshold", 0.5))
        tag = str(meta.get("primary_model", tag))
    else:
        threshold = 0.5
    if args.model:
        tag = args.model
    model, feat_names = load_model(config.MODEL_DIR, tag)
    # 归一化口径跟模型走。流式**物理上**只能用因果窗（未来帧还没到），所以如果
    # 模型是用居中窗训的，这里必然不一致 —— 明确报警而不是静默跑出无意义结果。
    norm = load_norm_config()
    fs = config.SAMPLING_RATE_HZ
    is_cnn = isinstance(model, CNNClassifier)
    feat_names = list(feat_names or [])
    need_audio = (not is_cnn) and any(n.startswith("audio_") for n in feat_names)
    audio_only = need_audio and not any(not n.startswith("audio_") for n in feat_names)
    print(f"已加载模型 {tag} ({'CNN1D' if is_cnn else 'MLP-family'}); 阈值={threshold:.3f}")
    if need_audio:
        print(f"输入特征: {len(feat_names)} 维"
              + ("（仅音频）" if audio_only else "（CSI+音频融合）"))
    print(f"归一化口径(来自 {norm['source']}): "
          f"enabled={norm['enabled']}, 参考窗={norm['window_s']}s, "
          f"{'因果窗' if norm['causal'] else '居中窗'}")
    if is_cnn and norm["enabled"] and not norm["causal"]:
        print("  ⚠ 该模型用**居中窗**训练，而流式推理只能看历史帧（因果窗）——"
              "两者口径不一致，分数会偏低。部署前请用 --norm-causal 重训一次。")
    win, step = config.WINDOW_SIZE, config.WINDOW_STEP

    if args.file:
        sample = Path(args.file)
    else:
        # 默认跟训练同源：metrics.json 记了 data_dir / test_users（demo 数据必须
        # 和模型同源，否则喂的是另一种分布）；都没有再退回默认数据源 + ID9。
        data_src = meta.get("data_dir")
        test_users = tuple(meta.get("test_users") or ("ID9",))
        records = scan_dataset(data_src) if data_src else scan_dataset()
        _, test = person_split(records, test_users)
        fall = test[test.label == 1]
        assert not fall.empty, (f"测试用户 {list(test_users)} 没有 fall 样本，"
                                f"请先用 --file 指定")
        sample = Path(fall["path"].iloc[0])
    print(f"流式回放文件: {sample}")
    # 音频模型需要麦克风那一轨。这里用录音文件模拟"音频和 CSI 同时到达"：
    # 真实部署时把 read_audio_track 换成声卡回调/环形缓冲即可，窗口切法不变。
    audio_track = None
    if need_audio:
        if "::" in str(sample):
            raise SystemExit("该模型需要音频，但 pickle 样本没有 .wav 旁车文件；"
                             "请用 --file 指定 realdata 下的 CSV 样本")
        audio_track = read_audio_track(sample)
        if audio_track is None:
            raise SystemExit(
                f"该模型需要音频，但 {sample} 旁边没有可用的 .wav/.sync.json。\n"
                f"换一条带录音的样本，或改用纯 CSI 模型（--model mlp）。"
            )
        print(f"音频轨: {audio_track.path.name}  {audio_track.duration:.2f}s @ "
              f"{audio_track.fs} Hz（与本样本 CSI 按 sync.json 对齐）")
    print(f"窗={win} 包 / 推理间隔={step} 包 / 采样率≈{fs:.0f} Hz / "
          f"阈值={threshold:.3f}\n")

    ring = deque(maxlen=win)          # 最近 win 个幅度向量
    # 基线估计窗有界(覆盖最近约 4 个窗)：无界 history 会使内存与每次推理的
    # 中位数扫描随流长线性增长(O(n^2))，且基线被很久以前的动作拖住
    history = deque(maxlen=max(win * 4, 512))
    times_all, amps_all = [], []

    n_pkt = 0
    last_infer = -10**9
    infer_times, infer_probs = [], []
    alarm_state = False
    alarm_events: list[tuple[float, float]] = []  # (起始秒, 结束秒)
    alarm_start: float | None = None

    for t, amp_vec in iter_sample_rows(sample):
        ring.append(amp_vec)
        history.append(amp_vec)
        n_pkt += 1
        times_all.append(t)
        amps_all.append(amp_vec)
        if n_pkt < win:
            continue
        if n_pkt - last_infer < step:
            continue  # 还没攒够 step 个新包，先不推理
        last_infer = n_pkt

        buf = np.asarray(ring, dtype=np.float32)           # (win, S)
        if is_cnn:
            # CNN 吃 (1, win, S) 滑窗；训练时用的是归一化后的幅度，这里必须一致。
            # 离线训练用"居中"参考窗，流式只能用"过去 1.5 s"（因果）——都是去掉
            # 静态 |H(f)| 形状，差的是参考量里会不会带上跌倒的开头，实测可接受。
            if norm["enabled"]:
                ref_n = max(1, int(round(norm["window_s"] * fs)))
                hist = np.asarray(history, dtype=np.float32)
                ref = np.median(hist[-ref_n:], axis=0) if len(hist) else np.ones(
                    buf.shape[1], dtype=np.float32)
                buf = (buf / np.maximum(ref, 1e-6)).astype(np.float32)
            p = float(predict_proba_pos(model, buf.reshape(1, win, -1))[0])
        else:
            base = np.median(np.asarray(history), axis=0)   # 在线基线
            feat, _ = window_feature_vector(buf, baseline_s=base, fs=fs)
            if need_audio:
                # 窗覆盖第 n_pkt-win .. n_pkt-1 行，按行号取同一时间段的声学特征
                a = audio_track.features_for_rows(n_pkt - win, win)
                feat = a if audio_only else np.concatenate([feat, a]).astype(np.float32)
            p = float(predict_proba_pos(model, feat.reshape(1, -1))[0])

        center_t = times_all[-win + win // 2]
        infer_times.append(center_t)
        infer_probs.append(p)
        mark = "  <<< 跌倒报警 >>>" if p >= threshold else ""
        print(f"  t={center_t:7.3f}s  P(fall)={p:.3f}  包#{n_pkt}{mark}")

        # 沿边沿检测报警开始/结束
        if p >= threshold and not alarm_state:
            alarm_state, alarm_start = True, center_t
        elif p < threshold and alarm_state:
            alarm_state = False
            if alarm_start is not None:
                alarm_events.append((alarm_start, center_t))
                alarm_start = None
    if alarm_state and alarm_start is not None:
        alarm_events.append((alarm_start, times_all[-1]))

    # ---------------------------------------------------------------
    if not infer_times:
        print("样本过短，未形成完整窗口，无结果。")
        return
    probs = np.asarray(infer_probs)
    # 与文件级评估规则一致：窗平均概率 >= 决策阈值
    verdict = bool(float(probs.mean()) >= threshold)
    print("\n================ 流式检测小结 ================")
    print(f"  样本长度: {n_pkt} 包 ≈ {times_all[-1]:.2f}s | 推理次数: {len(infer_times)}")
    print(f"  触发报警的窗口比例: {(probs >= threshold).mean() * 100:.1f}%")
    if alarm_events:
        for a0, a1 in alarm_events:
            print(f"  报警事件: t=[{a0:.2f}, {a1:.2f}] s (持续 {a1 - a0:.2f}s)")
    else:
        print("  报警事件: 无")
    print(f"  结论: {'检出跌倒 (FALL DETECTED)' if verdict else '未检出跌倒'}")

    # 从所在目录名(活动)推断真值，避免非跌倒样本被误标成 Fall
    # 从所在目录名(活动)推断真值；对 pickle 引用需从 seq 中读 action_name。
    # 兼容 CSV (path 的 parent 是活动目录) 与 pickle::idx (seq[idx].action_name)。
    sample_str = str(sample)
    if "::" in sample_str:
        from wifalldemo.preprocessing import _parse_pickle_ref, _load_pickle_seq
        pkl_abs, idx = _parse_pickle_ref(sample_str)
        seq = _load_pickle_seq(pkl_abs)
        act_name = seq[idx].get("action_name", "")
        gt = 1 if "fall" in str(act_name).lower() else 0
        title_sample = f"{act_name}/idx={idx}"
    else:
        gt = 1 if "fall" in sample.parent.name.lower() else 0
        title_sample = f"{sample.parent.name}/{sample.name[:19]}..."
    plot_detection_timeline(
        np.asarray(infer_times),
        probs,
        gt_label=gt,
        threshold=threshold,
        fs=fs,
        name="realtime_timeline.png",
        title=f"Realtime streaming detection — {title_sample}",
    )
    print(f"时间线图已保存: {config.FIG_DIR / 'realtime_timeline.png'}")


if __name__ == "__main__":
    main()
