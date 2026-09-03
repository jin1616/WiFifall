#!/usr/bin/env python
"""实时（流式）检测演示。

模拟从 ESP32/Widar3.0 采集端逐包接收 CSI：每来一个包就往环形缓冲推入，
每积累 WINDOW_STEP 个新包就基于最近 WINDOW_SIZE 个包在线提取特征并打分。
当 P(fall) 穿越阈值时打印报警（含模拟时间）。

这样演示了真实系统中可以嵌入的流式推理循环——只需把下方 read_packet()
替换成 socket/串口的采集回调即可接入真实设备。

用法:
    python run_train.py            # 先训练
    python run_demo_realtime.py
    python run_demo_realtime.py --file <某条测试样本路径>
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np

from wifalldemo import config
from wifalldemo.features import window_feature_vector
from wifalldemo.models import load_model, predict_proba_pos
from wifalldemo.visualize import plot_detection_timeline
from wifalldemo.preprocessing import parse_data_field, complex_to_amplitude


def stream_rows(path: Path):
    """逐行读取样本 CSV，产出 (时刻秒, 幅度向量)。模拟一个采集源。"""
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        rd = csv.DictReader(f)
        t0 = None
        for row in rd:
            if row.get("type") != "CSI_DATA":
                continue
            csi = parse_data_field(row.get("data", ""))
            if csi is None:
                continue
            ts_us = row.get("local_timestamp")
            t = 0.0
            if ts_us is not None and ts_us.isdigit():
                ts = int(ts_us)
                t0 = ts if t0 is None else t0
                t = (ts - t0) / 1e6
            yield t, complex_to_amplitude(csi)


def main() -> None:
    ap = argparse.ArgumentParser(description="实时流式跌倒检测演示")
    ap.add_argument("--model", default=None,
                    help="rf / lr / primary(默认，训练择优结果)")
    ap.add_argument(
        "--file",
        default=None,
        help="测试样本 CSV；缺省自动选取 ID9 下的首个 fall 样本",
    )
    args = ap.parse_args()

    thr_path = config.OUTPUT_DIR / "metrics.json"
    tag = "rf"
    if thr_path.exists():
        meta = json.loads(thr_path.read_text(encoding="utf-8"))
        threshold = float(meta.get("threshold", 0.5))
        tag = str(meta.get("primary_model", tag))
    else:
        threshold = 0.5
    if args.model:
        tag = args.model
    model, _ = load_model(config.MODEL_DIR, tag)
    fs = config.SAMPLING_RATE_HZ
    print(f"已加载模型 {tag}")
    win, step = config.WINDOW_SIZE, config.WINDOW_STEP

    if args.file:
        sample = Path(args.file)
    else:
        cand = sorted((config.DATA_DIR / "ID9" / "fall").glob("*.csv"))
        assert cand, "未找到 ID9/fall 样本，请先用 --file 指定"
        sample = cand[0]
    print(f"流式回放文件: {sample}")
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

    for t, amp_vec in stream_rows(sample):
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
        base = np.median(np.asarray(history), axis=0)      # 在线基线
        feat, _ = window_feature_vector(buf, baseline_s=base, fs=fs)
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
    gt = 1 if "fall" in sample.parent.name.lower() else 0
    plot_detection_timeline(
        np.asarray(infer_times),
        probs,
        gt_label=gt,
        threshold=threshold,
        fs=fs,
        name="realtime_timeline.png",
        title=f"Realtime streaming detection — {sample.parent.name}/{sample.name[:19]}...",
    )
    print(f"时间线图已保存: {config.FIG_DIR / 'realtime_timeline.png'}")


if __name__ == "__main__":
    main()
