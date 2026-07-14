#!/usr/bin/env python3
"""
生成任务一图像：MIP序列 + 等值线叠加
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import struct

plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = Path('/home/zhb/DV_Final/data/Nyx')
OUT_DIR = Path('/home/zhb/DV_Final/Document/figures')

NX, NY, NZ = 128, 128, 128

def read_raw_dat(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    n_floats = len(raw) // 4
    data = struct.unpack('<' + 'f' * n_floats, raw)
    return np.array(data, dtype=np.float32).reshape((NX, NY, NZ))

def generate_task1_figures():
    # ===== 图1: MIP序列 =====
    steps = [0, 25, 50, 75, 99]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), facecolor='white')
    
    for ax, ts in zip(axes, steps):
        data = read_raw_dat(DATA_DIR / f"{ts:04d}.dat")
        mip = np.max(data, axis=2)
        im = ax.imshow(mip, cmap='hot', origin='lower', vmin=data.min(), vmax=data.max())
        ax.set_title(f't = {ts}', fontsize=12, color='white')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.patch.set_facecolor('#0a0a12')
    for ax in axes:
        ax.set_facecolor('#0a0a12')
    
    plt.suptitle('宇宙密度场最大强度投影演化序列（Z轴MIP）', color='white', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'task1_mip_sequence.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a12', edgecolor='none')
    plt.close()
    print("✓ task1_mip_sequence.png 已生成")
    
    # ===== 图2: 等值线叠加 =====
    data = read_raw_dat(DATA_DIR / '0099.dat')
    mip = np.max(data, axis=2)
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # 显示 MIP
    im = ax.imshow(mip, cmap='hot', origin='lower', vmin=mip.min(), vmax=mip.max())
    
    # 计算分位数阈值
    th95 = np.percentile(mip, 95)
    th99 = np.percentile(mip, 99)
    
    # 绘制等值线
    ax.contour(mip, levels=[th95, th99], colors='cyan', linewidths=1.5, alpha=0.8)
    
    # 标注晕核区域（黑色虚线框）
    rect1 = Rectangle((40, 35), 30, 30, fill=False, edgecolor='black',
                       linestyle='--', linewidth=2)
    ax.add_patch(rect1)
    ax.text(55, 33, '晕核', color='black', fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 标注纤维区域（白色虚线框）
    rect2 = Rectangle((75, 70), 35, 25, fill=False, edgecolor='white',
                       linestyle='--', linewidth=2)
    ax.add_patch(rect2)
    ax.text(92, 68, '纤维', color='white', fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_title('t=99 MIP 投影 + 95%/99% 分位数等值线', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6, label='密度值')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'task1_contours.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ task1_contours.png 已生成")

if __name__ == '__main__':
    generate_task1_figures()
