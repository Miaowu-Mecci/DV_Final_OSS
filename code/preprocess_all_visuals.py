#!/usr/bin/env python3
"""
预渲染所有可视化数据：
1. 投影图 PNG（已完成，这里补充缺失的）
2. 统计图 PNG（直方图、对数直方图、CDF）
3. 体数据二进制文件（供前端直接加载到 GPU）
"""

import numpy as np
import json
import os
from pathlib import Path
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 路径配置
DATA_DIR = Path("/home/zhb/DV_Final/data/Nyx")
PROCESSED_DIR = Path("/home/zhb/DV_Final/data/processed")
WEB_DATA_DIR = Path("/home/zhb/DV_Final/web/static/data")

STAT_PNG_DIR = WEB_DATA_DIR / "stats"
VOLUME_BIN_DIR = WEB_DATA_DIR / "volumes"
PROJECTIONS_DIR = WEB_DATA_DIR / "projections"

for d in [STAT_PNG_DIR, VOLUME_BIN_DIR, PROJECTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NX, NY, NZ = 128, 128, 128
NTOTAL = 100


def read_raw_dat(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    n_floats = len(raw) // 4
    data = struct.unpack('<' + 'f' * n_floats, raw)
    return np.array(data, dtype=np.float32).reshape((NX, NY, NZ))


def compute_projection(field, axis, proj_type='mip'):
    if proj_type == 'mip':
        return np.max(field, axis=axis)
    elif proj_type == 'aip':
        return np.mean(field, axis=axis)
    else:
        raise ValueError(f"Unknown projection type: {proj_type}")


def save_projection_png(data, filepath):
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=64)
    im = ax.imshow(data, cmap='hot', origin='lower', aspect='auto')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(filepath, dpi=64, bbox_inches='tight', pad_inches=0,
                facecolor='black', edgecolor='none')
    plt.close(fig)


def save_histogram_png(hist_data, ts):
    """保存三张统计图为 PNG"""
    edges = np.array(hist_data['edges'])
    hist = np.array(hist_data['hist'])
    centers = (edges[:-1] + edges[1:]) / 2

    # 计算 log histogram
    log_edges = np.log10(edges - edges.min() + 1e-10)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    log_hist = hist.copy()

    # 计算 CDF
    cdf = np.cumsum(hist)
    cdf_norm = cdf / cdf[-1] if cdf[-1] > 0 else cdf

    stats = hist_data.get('stats', {})

    # 1. 密度直方图
    fig, ax = plt.subplots(figsize=(3.2, 1.8), dpi=72)
    ax.bar(centers, hist, width=(edges[1]-edges[0])*0.9,
           color='#4facfe', edgecolor='none', alpha=0.7)
    ax.set_yscale('log')
    ax.set_title(f'密度分布直方图 (t={ts})', color='white', fontsize=10)
    ax.set_xlabel('密度值', color='#8892b0', fontsize=8)
    ax.set_ylabel('频数', color='#8892b0', fontsize=8)
    ax.tick_params(colors='#8892b0', labelsize=7)
    ax.spines['bottom'].set_color('#2a2a40')
    ax.spines['left'].set_color('#2a2a40')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    fig.savefig(STAT_PNG_DIR / f"hist_{ts:04d}.png", dpi=72,
                bbox_inches='tight', pad_inches=0.1, facecolor='none', edgecolor='none')
    plt.close(fig)

    # 2. 对数直方图
    fig, ax = plt.subplots(figsize=(3.2, 1.8), dpi=72)
    ax.bar(log_centers, log_hist, width=(log_edges[1]-log_edges[0])*0.9,
           color='#00f2fe', edgecolor='none', alpha=0.6)
    ax.set_title(f'密度对数直方图 (t={ts})', color='white', fontsize=10)
    ax.set_xlabel('log₁₀(ρ)', color='#8892b0', fontsize=8)
    ax.set_ylabel('频数', color='#8892b0', fontsize=8)
    ax.tick_params(colors='#8892b0', labelsize=7)
    ax.spines['bottom'].set_color('#2a2a40')
    ax.spines['left'].set_color('#2a2a40')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    fig.savefig(STAT_PNG_DIR / f"loghist_{ts:04d}.png", dpi=72,
                bbox_inches='tight', pad_inches=0.1, facecolor='none', edgecolor='none')
    plt.close(fig)

    # 3. CDF
    fig, ax = plt.subplots(figsize=(3.2, 1.8), dpi=72)
    ax.fill_between(centers, cdf_norm, alpha=0.12, color='#a18cd1')
    ax.plot(centers, cdf_norm, color='#a18cd1', linewidth=1.5)

    # 分位数线
    for p, color in [(stats.get('p1'), '#51cf66'), (stats.get('p5'), '#51cf66'),
                      (stats.get('p95'), '#ff6b6b'), (stats.get('p99'), '#ff6b6b')]:
        if p is not None:
            ax.axvline(p, color=color, linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_title(f'密度累积分布函数 (t={ts})', color='white', fontsize=10)
    ax.set_xlabel('密度值', color='#8892b0', fontsize=8)
    ax.set_ylabel('累积概率', color='#8892b0', fontsize=8)
    ax.tick_params(colors='#8892b0', labelsize=7)
    ax.spines['bottom'].set_color('#2a2a40')
    ax.spines['left'].set_color('#2a2a40')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    fig.savefig(STAT_PNG_DIR / f"cdf_{ts:04d}.png", dpi=72,
                bbox_inches='tight', pad_inches=0.1, facecolor='none', edgecolor='none')
    plt.close(fig)


def save_volume_binary(ts):
    """将体数据保存为纯二进制文件"""
    npz_path = PROCESSED_DIR / "volumes" / f"vol_{ts:04d}.npz"
    if not npz_path.exists():
        return False

    data = np.load(npz_path)
    vol = data['data']  # 应该是 uint8, 64x64x64

    bin_path = VOLUME_BIN_DIR / f"vol_{ts:04d}.bin"
    with open(bin_path, 'wb') as f:
        f.write(vol.astype(np.uint8).tobytes())

    return True


def main():
    print("开始预渲染所有可视化数据...")

    # 加载 histograms
    with open(PROCESSED_DIR / "histograms.json") as f:
        histograms = json.load(f)

    # 加载 stats
    with open(PROCESSED_DIR / "stats.json") as f:
        stats = json.load(f)

    # 合并 stats 到 histograms
    for i, h in enumerate(histograms):
        h['stats'] = stats[i]

    for ts in range(NTOTAL):
        # 1. 投影图
        dat_path = DATA_DIR / f"{ts:04d}.dat"
        if dat_path.exists():
            field = read_raw_dat(dat_path)
            for axis in range(3):
                axis_name = ['x', 'y', 'z'][axis]
                for ptype in ['mip', 'aip']:
                    proj = compute_projection(field, axis, ptype)
                    filepath = PROJECTIONS_DIR / f"{ptype}_{axis_name}_{ts:04d}.png"
                    if not filepath.exists():
                        save_projection_png(proj, filepath)

        # 2. 统计图
        if ts < len(histograms):
            save_histogram_png(histograms[ts], ts)

        # 3. 体数据二进制
        save_volume_binary(ts)

        if ts % 10 == 0:
            print(f"  已处理 t={ts}")

    print(f"\n完成！")
    print(f"  投影图: {PROJECTIONS_DIR}")
    print(f"  统计图: {STAT_PNG_DIR}")
    print(f"  体数据: {VOLUME_BIN_DIR}")


if __name__ == '__main__':
    main()
