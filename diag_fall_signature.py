# 对比自采 fall 与 Widar fall 的"跌倒签名"：冲击尖峰 + 倒地静止段
# 用法: E:/Miniconde/envs/big_create/python.exe diag_fall_signature.py
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from wifalldemo.preprocessing import load_amplitude_series, normalize_per_subcarrier
from wifalldemo import config

FS = config.SAMPLING_RATE_HZ

def signature(amp):
    """归一化后逐帧均值轨迹 -> (运动强度, 峰值强度, 峰后静止度)"""
    rel = normalize_per_subcarrier(amp, fs=FS, window_s=config.NORM_WINDOW_S, causal=False)
    x = rel.mean(axis=1)
    d = np.abs(np.diff(x))
    motion = float(d.mean())
    peak = float(d.max())
    # 峰后静止度：峰值时刻之后 1.5s 的平均变化 / 峰前 1.5s 平均变化
    ip = int(np.argmax(d))
    w = int(1.5 * FS)
    pre = d[max(0, ip - w):ip].mean() if ip > 10 else np.nan
    post = d[ip:ip + w].mean() if ip + 10 < len(d) else np.nan
    still = float(post / (pre + 1e-9)) if not (np.isnan(pre) or np.isnan(post)) else np.nan
    return motion, peak, still, x

def scan(root, users, acts=('fall',)):
    rows = []
    for u in users:
        for act in acts:
            p = os.path.join(root, u, act)
            if not os.path.isdir(p):
                continue
            for f in sorted(os.listdir(p)):
                if not f.endswith('.csv') or '_audio_env' in f:
                    continue
                fp = os.path.join(p, f)
                try:
                    amp = load_amplitude_series(fp)
                    m, pk, st, x = signature(amp)
                    rows.append(dict(user=u, act=act, file=f, dur=round(len(amp)/FS, 2),
                                     motion=round(m, 3), peak=round(pk, 3), still=round(st, 3) if st == st else None))
                except Exception as e:
                    print('ERR', fp, e)
    return rows

print('=== 自采 fall 逐条（realdata）===')
self_rows = scan(os.path.join(ROOT, 'realdata'), ['ID11', 'ID12', 'ID13', 'ID14'])
for r in self_rows:
    flag = 'OK ' if r['peak'] >= 1.5 else '弱!'
    print(f"  {flag} {r['user']}/{r['file']:16s} 时长{r['dur']:4.1f}s  运动{r['motion']:.3f}  峰值{r['peak']:.2f}  峰后/峰前={r['still']}")
ms = [r['motion'] for r in self_rows]; pk = [r['peak'] for r in self_rows]
print(f'  汇总 n={len(self_rows)} 运动中位 {np.median(ms):.3f} 峰值中位 {np.median(pk):.2f} 峰值>=1.5 的 {sum(1 for v in pk if v>=1.5)}/{len(pk)}')

print()
print('=== Widar fall 抽样（ID0/ID1/ID5 全部 + ID2 抽 20 条）===')
w_rows = []
for u, cap in (('ID0', 99), ('ID1', 99), ('ID5', 99), ('ID2', 20)):
    p = os.path.join(ROOT, 'DATA', u, 'fall')
    files = sorted(os.listdir(p))[:cap]
    for f in files:
        if not f.endswith('.csv'):
            continue
        fp = os.path.join(p, f)
        try:
            amp = load_amplitude_series(fp)
            m, pk, st, x = signature(amp)
            w_rows.append(dict(user=u, file=f, motion=round(m, 3), peak=round(pk, 3),
                               still=round(st, 3) if st == st else None))
        except Exception as e:
            print('ERR', fp, e)
for r in w_rows[:12]:
    print(f"      {r['user']}/{r['file']:16s} 运动{r['motion']:.3f}  峰值{r['peak']:.2f}  峰后/峰前={r['still']}")
ms = [r['motion'] for r in w_rows]; pk = [r['peak'] for r in w_rows]
st = [r['still'] for r in w_rows if r['still'] is not None]
print(f'  汇总 n={len(w_rows)} 运动中位 {np.median(ms):.3f} 峰值中位 {np.median(pk):.2f} 峰值>=1.5 的 {sum(1 for v in pk if v>=1.5)}/{len(pk)} 峰后/峰前中位 {np.median(st):.2f}')

# 导出两条代表轨迹画图：自采最差/最好 + Widar 典型
def trace_of(fp):
    amp = load_amplitude_series(fp)
    rel = normalize_per_subcarrier(amp, fs=FS, window_s=config.NORM_WINDOW_S, causal=False)
    x = rel.mean(axis=1)
    return (np.arange(len(x)) / FS).tolist(), x.tolist()

worst = min(self_rows, key=lambda r: r['peak'])
best = max(self_rows, key=lambda r: r['peak'])
wt, wx = trace_of(os.path.join(ROOT, 'DATA', 'ID0', 'fall', sorted(os.listdir(os.path.join(ROOT, 'DATA', 'ID0', 'fall')))[0]))
bt, bx = trace_of(os.path.join(ROOT, 'realdata', best['user'], 'fall', best['file']))
dt, dx = trace_of(os.path.join(ROOT, 'realdata', worst['user'], 'fall', worst['file']))
json.dump({'widar': [wt, wx], 'self_best': [bt, bx], 'self_worst': [dt, dx],
           'best_meta': best, 'worst_meta': worst},
          open(os.path.join(ROOT, 'outputs', 'fall_sig.json'), 'w'))
print()
print('轨迹已存 outputs/fall_sig.json  best:', best['user'], best['file'], 'worst:', worst['user'], worst['file'])
