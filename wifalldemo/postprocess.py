"""打分后处理：文件内相对打分、top-k 文件投票、"坠落后静止"时序确认。

动机（LOPO 诊断的结论）
------------------------
自采数据每人只有一个采集批次，不同人/批次的**绝对**打分尺度差很多：
ID15 的撞击声偏轻，在 ID11 上选出的绝对阈值下整折 recall≈0；但把每个文件
的分**对自己做稳健归一化**后，文件级 AUC 仍有 0.95 —— 说明排序信号在、
失效的是跨文件的绝对阈值。本模块把两件事固化进评估/推理链路：

1. :func:`relative_scores` —— 窗分在文件内做 (median, MAD) 稳健 z-score，
   阈值在同样的相对空间里选，部署时对每个待判片段同样归一化。
2. :func:`tune_stillness_rule` / :func:`apply_stillness_rule` —— 跌倒是
   "剧烈冲击 → 长时间静止"的**转换**；stand/sit 误报常常也有冲击样窗，
   但触发后人体继续活动。规则在 top-k 触发的文件上检查触发点后若干窗的
   CSI 活动度：没有显著静下来（或触发能量相对文件基线不够暴）就否决。
   参数在**验证人**上网格搜索（含"不用规则"的候选），验证集不支持时自动退回。

训练与推理必须用同一套后处理（阈值是在后处理后的空间里选的），所以这些
函数同时被 run_train（评估）、run_audio_ablation（LOPO 汇总）和
run_demo_offline（部署回放）调用。
"""
from __future__ import annotations

import numpy as np

# 活动度特征：跨子载波平均绝对差分（运动速度代理）。在 70 维 CSI 特征里，
# 名字固定为 {raw,rel}_diff_mean；找不到时退到 rel，再不行规则自动停用。
ACTIVITY_FEATURE_CANDIDATES = ("raw_diff_mean", "rel_diff_mean")

DEFAULT_TOPK = 3          # top-k 投票窗数（与 run_train 主口径一致）
DEFAULT_POST_N = 4        # 触发点后检查 4 个窗 ≈ 1.5 s（窗步 0.39s）
STILL_RATIO_GRID = (0.4, 0.6, 0.8)     # 触发后活动度须降到触发窗的该比例以下
BURST_RATIO_GRID = (2.0, 3.0, 5.0)     # 触发窗活动度须达文件中位基线的该倍数


# ----------------------------------------------------------------------------
# 文件内相对打分
# ----------------------------------------------------------------------------
def relative_scores(p: np.ndarray, groups: np.ndarray,
                    eps: float = 1e-6) -> np.ndarray:
    """把窗分在文件内转成稳健 z-score：``(p - median) / (1.4826 * MAD)``。

    单窗文件 MAD=0 时退化为 ``p - median``（相对中位的偏移），避免出现
    除零产生的 inf。返回与 ``p`` 同形的 float64。
    """
    p = np.asarray(p, dtype=np.float64)
    out = np.empty_like(p)
    for g in np.unique(groups):
        m = groups == g
        pv = p[m]
        med = float(np.median(pv))
        mad = float(np.median(np.abs(pv - med)))
        scale = 1.4826 * mad
        out[m] = (pv - med) / (scale + eps) if scale > eps else (pv - med)
    return out


def hybrid_scores(p: np.ndarray, groups: np.ndarray, w: float = 0.25,
                  eps: float = 1e-6) -> np.ndarray:
    """相对 z 分 + w × 原始概率：兼顾"文件内对比"与"跨文件绝对置信度"。

    纯相对打分在**绝对校准已能跨人迁移**的模型上反而掉点（实测 mlp_av 文件级
    Acc 0.842→0.280：top-k 的组内 z 分跨文件趋于常数，绝对信息被抹掉）；
    纯原始分数又让 ID15 这类整体偏轻的折够不到阈值。混合分让两者互补：
    w=0.25 时 z 分（跨度通常 ±3~5）仍主导文件内对比，原始概率作为小幅
    跨文件修正。阈值仍在同一混合空间的验证集上选。
    """
    return relative_scores(p, groups, eps=eps) + w * np.asarray(p, dtype=np.float64)


# ----------------------------------------------------------------------------
# top-k 文件级投票
# ----------------------------------------------------------------------------
def topk_file_scores(p: np.ndarray, groups: np.ndarray, y: np.ndarray,
                     k: int = DEFAULT_TOPK) -> tuple[np.ndarray, np.ndarray]:
    """按文件聚合成 top-k 窗均值分数，返回 ``(file_prob, file_label)``。

    跌倒是短时事件：一个文件里只有少数窗含冲击，全窗均值会稀释证据；
    stand/sit 这类误报往往只有零星高分窗，top-k 对两者都更稳。
    """
    gids = np.unique(groups)
    p_f = np.array([
        float(np.sort(p[groups == g])[-min(k, int((groups == g).sum())):].mean())
        for g in gids
    ])
    y_f = np.array([int(round(float(y[groups == g].mean()))) for g in gids])
    return p_f, y_f


# ----------------------------------------------------------------------------
# "坠落后静止"确认规则
# ----------------------------------------------------------------------------
def _confirm_one(p: np.ndarray, act: np.ndarray, k: int, post_n: int,
                 still_ratio: float, burst_ratio: float) -> bool:
    """单文件：top-k 触发后，后置窗是否显著静止、触发是否足够"暴"。

    返回 True = 确认跌倒；False = 否决（改判非跌倒）。
    没有后置窗可检查时**不否决**（录音可能在落地后立刻停止，查无凭据）。
    """
    n = len(p)
    if n == 0:
        return True
    k_eff = min(k, n)
    top_idx = np.argsort(p)[-k_eff:]
    last_trig = int(top_idx.max())
    post = np.arange(last_trig + 1, min(last_trig + 1 + post_n, n))
    if post.size == 0:
        return True
    trig_act = float(np.mean(act[top_idx]))
    base_act = float(np.median(act)) + 1e-9
    post_act = float(np.mean(act[post]))
    if trig_act < base_act * burst_ratio:      # 触发能量不够"暴"（相对基线）
        return False
    return post_act <= trig_act * still_ratio  # 触发后没有静下来


def apply_stillness_rule(p: np.ndarray, groups: np.ndarray, act: np.ndarray,
                         params: dict | None, k: int = DEFAULT_TOPK) -> np.ndarray:
    """对 top-k 判正的文件依次跑确认规则，返回**调整后的文件预测标签**。

    窗口在每组内必须按时间有序（dataset 构建时按起点追加，天然有序）。
    调用前规则参数已由 :func:`tune_stillness_rule` 在验证人上选好（或为
    None = 规则关闭，此时本函数不应被调用）。
    """
    if params is None:
        raise ValueError("规则参数为 None（规则关闭）时不应调用本函数")
    gids = np.unique(groups)
    p_f, _ = topk_file_scores(p, groups, np.zeros(len(p), dtype=int), k=k)
    thr = float(params["threshold_topk"])
    pred = (p_f >= thr).astype(int)
    for i, g in enumerate(gids):
        if not pred[i]:
            continue
        m = groups == g
        if not _confirm_one(p[m], act[m], k, int(params.get("post_n", DEFAULT_POST_N)),
                            float(params["still_ratio"]),
                            float(params["burst_ratio"])):
            pred[i] = 0
    return pred


def tune_stillness_rule(p_val: np.ndarray, groups_val: np.ndarray,
                        act_val: np.ndarray, y_val: np.ndarray,
                        thr_topk_val: float, k: int = DEFAULT_TOPK) -> dict | None:
    """在验证人上为确认规则选参数（含"关闭规则"的 None 候选）。

    以文件级 F1 为目标做小规模网格搜索；规则在验证集上都不能提升 F1 时
    返回 ``None``（关闭），避免在分布不支持的数据上硬套规则伤召回。
    """
    from .models import evaluate

    p_f, y_f = topk_file_scores(p_val, groups_val, y_val, k=k)
    gids = np.unique(groups_val)
    base = evaluate(y_f, p_f, thr_topk_val)["f1"]

    best_f1, best = base, None
    for still in STILL_RATIO_GRID:
        for burst in BURST_RATIO_GRID:
            pred = (p_f >= thr_topk_val).astype(int)
            for i, g in enumerate(gids):
                if not pred[i]:
                    continue
                m = groups_val == g
                if not _confirm_one(p_val[m], act_val[m], k, DEFAULT_POST_N,
                                    still, burst):
                    pred[i] = 0
            f1 = evaluate(y_f, pred.astype(float), 0.5)["f1"]
            if f1 > best_f1 + 1e-6:
                best_f1, best = f1, {"still_ratio": float(still),
                                     "burst_ratio": float(burst),
                                     "post_n": DEFAULT_POST_N,
                                     "threshold_topk": float(thr_topk_val)}
    return best


def find_activity_index(feat_names: list[str]) -> int | None:
    """在特征名列表里找活动度特征的列号，找不到返回 None。"""
    for n in ACTIVITY_FEATURE_CANDIDATES:
        if n in feat_names:
            return feat_names.index(n)
    return None


__all__ = [
    "ACTIVITY_FEATURE_CANDIDATES",
    "DEFAULT_POST_N",
    "DEFAULT_TOPK",
    "apply_stillness_rule",
    "find_activity_index",
    "hybrid_scores",
    "relative_scores",
    "topk_file_scores",
    "tune_stillness_rule",
]
