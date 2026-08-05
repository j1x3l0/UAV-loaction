import csv, glob
import numpy as np

def load(pattern):
    data = {}
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                key = (row['axis_name'], row['level'], row['unit'])
                data.setdefault(key, []).append(float(row['success_rate']))
    return data

clean = load('/tmp/sv1007_deg/clean/*.csv')
cur = load('/tmp/sv1007_deg/curriculum/*.csv')

axes_order = ['高斯球稀疏化','渲染分辨率','深度噪声','光照偏移','视角不确定性','深度大面积失效','相机遮挡']
print(f"{'axis':<12}{'level':>6} {'clean':>7} {'curriculum':>10} {'diff':>6}  (clean-cur)")
print('-'*55)
for ax in axes_order:
    rows = sorted([k for k in clean if k[0]==ax], key=lambda k: float(k[1]) if k[1].replace('.','').isdigit() else 0)
    for k in rows:
        c = np.mean(clean[k]); v = np.mean(cur.get(k,[np.nan]*5))
        if np.isnan(v): continue
        d = c - v
        mark = ' *' if abs(d)>=10 else ''
        print(f"{ax:<12}{k[1]+k[2]:>8} {c:6.1f}% {v:9.1f}% {d:+5.1f}{mark}")
