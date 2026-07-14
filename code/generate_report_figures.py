#!/usr/bin/env python3
"""Generate additional figures for the LaTeX report."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path('/home/zhb/DV_Final/data/Nyx')
OUT_DIR = Path('/home/zhb/DV_Final/data/processed/report_figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID = 128

def load_timestep(idx):
    path = DATA_DIR / f"{idx:04d}.dat"
    return np.fromfile(path, dtype=np.float32).reshape((GRID, GRID, GRID))

# 1. Log histogram evolution for key timesteps
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
steps = [0, 10, 25, 50, 75, 99]
for ax, ts in zip(axes, steps):
    data = load_timestep(ts)
    log_data = np.log10(data - data.min() + 1e-8)
    ax.hist(log_data.flatten(), bins=80, color='steelblue', edgecolor='white', alpha=0.8, density=True)
    ax.set_xlabel('log₁₀(ρ - ρ_min + ε)')
    ax.set_ylabel('概率密度')
    ax.set_title(f't = {ts}')
    ax.set_yscale('log')
plt.suptitle('宇宙密度对数直方图演化序列', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / 'log_histogram_evolution.png', dpi=200, bbox_inches='tight')
plt.close()

# 2. Comparison: early vs late density field (slice)
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for row, ts in enumerate([0, 99]):
    data = load_timestep(ts)
    for col, (axis, pos, title_suffix) in enumerate([
        (0, 64, 'X=64切片'),
        (1, 64, 'Y=64切片'),
        (2, 64, 'Z=64切片')
    ]):
        ax = axes[row, col]
        if axis == 0:
            sl = data[pos, :, :]
        elif axis == 1:
            sl = data[:, pos, :]
        else:
            sl = data[:, :, pos]
        im = ax.imshow(sl, cmap='hot', origin='lower')
        ax.set_title(f't={ts} {title_suffix}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle('早期(t=0)与晚期(t=99)密度场切片对比', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / 'slice_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

# 3. CDF evolution showing polarization
fig, ax = plt.subplots(figsize=(10, 6))
steps_cdf = [0, 20, 40, 60, 80, 99]
colors = plt.cm.plasma(np.linspace(0, 1, len(steps_cdf)))
for ts, c in zip(steps_cdf, colors):
    data = load_timestep(ts).flatten()
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    ax.plot(sorted_data, cdf, color=c, label=f't={ts}', linewidth=2)
ax.set_xlabel('密度值')
ax.set_ylabel('累积概率')
ax.set_title('密度累积分布函数(CDF)时序演化')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'cdf_evolution.png', dpi=200, bbox_inches='tight')
plt.close()

# 4. Power spectrum proxy: variance at different scales
fig, ax = plt.subplots(figsize=(10, 6))
from scipy import ndimage
steps_all = range(0, 100, 5)
variances = []
for ts in steps_all:
    data = load_timestep(ts)
    # Compute standard deviation as proxy for structure formation
    variances.append(data.std())
ax.plot(list(steps_all), variances, 'o-', color='#e94560', linewidth=2, markersize=6)
ax.set_xlabel('时间步长')
ax.set_ylabel('密度标准差 σ')
ax.set_title('密度涨落幅度演化（结构形成强度指标）')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'std_evolution.png', dpi=200, bbox_inches='tight')
plt.close()

# 5. Volume rendering style composite: isosurface + MIP concept
fig = plt.figure(figsize=(14, 6))
for idx, ts in enumerate([0, 50, 99]):
    ax = fig.add_subplot(1, 3, idx+1)
    data = load_timestep(ts)
    mip = np.max(data, axis=2)
    im = ax.imshow(mip, cmap='magma', origin='lower', vmin=data.min(), vmax=data.max())
    ax.set_title(f't = {ts}', fontsize=13)
    ax.axis('off')
    # Add contour lines for high density
    levels = [np.percentile(data, 95), np.percentile(data, 99)]
    ax.contour(mip, levels=levels, colors='cyan', linewidths=0.8, alpha=0.7)
plt.suptitle('最大强度投影与高密度等值线叠加（青色=95%/99%分位数）', fontsize=13, y=0.95)
plt.tight_layout()
plt.savefig(OUT_DIR / 'mip_contours.png', dpi=200, bbox_inches='tight')
plt.close()

print("Report figures generated in", OUT_DIR)
