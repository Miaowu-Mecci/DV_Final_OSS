#!/usr/bin/env python3
"""
重新生成正确的密度统计分布数据
- 线性直方图: 密度值原始分布
- 对数直方图: log10(密度) — 宇宙学标准表示
- 过密度直方图: log10(1+δ) 其中 δ = (ρ - ρ̄)/ρ̄
"""

import numpy as np
import json
from pathlib import Path

DATA_DIR = Path('/home/zhb/DV_Final/data/Nyx')
OUT_DIR = Path('/home/zhb/DV_Final/web/static/data')
N_STEPS = 100
GRID = 128

def load_timestep(idx):
    path = DATA_DIR / f"{idx:04d}.dat"
    return np.fromfile(path, dtype=np.float32)

def compute_stats(data):
    flat = data.flatten()
    return {
        'min': float(data.min()),
        'max': float(data.max()),
        'mean': float(data.mean()),
        'std': float(data.std()),
        'median': float(np.median(flat)),
        'p1': float(np.percentile(flat, 1)),
        'p5': float(np.percentile(flat, 5)),
        'p25': float(np.percentile(flat, 25)),
        'p75': float(np.percentile(flat, 75)),
        'p95': float(np.percentile(flat, 95)),
        'p99': float(np.percentile(flat, 99)),
        'p999': float(np.percentile(flat, 99.9)),
    }

# 全局统计：用于计算过密度
print("Computing global mean density...")
all_data = []
for i in range(N_STEPS):
    all_data.append(load_timestep(i))
    if i % 20 == 0:
        print(f"  loaded {i}/100")

# 每个时间步的均值
timestep_means = [d.mean() for d in all_data]
# 全局平均密度（所有时间步的平均）
global_mean = np.mean(timestep_means)
print(f"Global mean density: {global_mean:.4f}")

# 全局范围，用于统一 bin 边界
global_min = min(d.min() for d in all_data)
global_max = max(d.max() for d in all_data)
global_log_min = min(np.log10(d.min()) for d in all_data)
global_log_max = max(np.log10(d.max()) for d in all_data)

print(f"Density range: [{global_min:.3f}, {global_max:.3f}]")
print(f"Log10 density range: [{global_log_min:.3f}, {global_log_max:.3f}]")

# 统一 bin 边界
BINS = 200
lin_edges = np.linspace(global_min, global_max, BINS + 1)
log_edges = np.linspace(global_log_min, global_log_max, BINS + 1)

# 过密度: log10(1 + δ)，δ = (ρ - ρ̄_t)/ρ̄_t，每个时间步用自己的均值
# 先找出所有时间步的 log10(1+δ) 范围
all_logdelta = []
for d, mean in zip(all_data, timestep_means):
    delta = (d - mean) / mean
    logdelta = np.log10(1 + delta)
    all_logdelta.extend(logdelta)
all_logdelta = np.array(all_logdelta)
logdelta_min = all_logdelta.min()
logdelta_max = all_logdelta.max()
print(f"Log10(1+δ) range: [{logdelta_min:.3f}, {logdelta_max:.3f}]")
logdelta_edges = np.linspace(logdelta_min, logdelta_max, BINS + 1)

merged = []
for i, data in enumerate(all_data):
    stats = compute_stats(data)
    stats['timestep'] = i

    # 1. 线性直方图（统一 bins）
    hist, _ = np.histogram(data, bins=lin_edges)

    # 2. log10(密度) 直方图
    log_data = np.log10(data)
    log_hist, _ = np.histogram(log_data, bins=log_edges)

    # 3. 过密度 log10(1+δ) 直方图
    delta = (data - timestep_means[i]) / timestep_means[i]
    logdelta = np.log10(1 + delta)
    logdelta_hist, _ = np.histogram(logdelta, bins=logdelta_edges)

    merged.append({
        'timestep': i,
        'hist': hist.tolist(),
        'edges': lin_edges.tolist(),
        'log_hist': log_hist.tolist(),
        'log_edges': log_edges.tolist(),
        'logdelta_hist': logdelta_hist.tolist(),
        'logdelta_edges': logdelta_edges.tolist(),
        'stats': stats,
        'mean_density': float(timestep_means[i])
    })

    if i % 20 == 0:
        print(f"Processed {i}/100")

with open(OUT_DIR / 'histograms.json', 'w') as f:
    json.dump(merged, f)

print(f"\nSaved histograms.json with {len(merged)} entries")
print("Keys:", list(merged[0].keys()))
print(f"lin_edges: {lin_edges[0]:.3f} ~ {lin_edges[-1]:.3f}")
print(f"log_edges: {log_edges[0]:.3f} ~ {log_edges[-1]:.3f}")
print(f"logdelta_edges: {logdelta_edges[0]:.3f} ~ {logdelta_edges[-1]:.3f}")
