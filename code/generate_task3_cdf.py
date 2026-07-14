#!/usr/bin/env python3
"""
生成CDF时序演化序列图 - 替代热图，展示累积分布随时间的偏移
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def generate_cdf_sequence():
    steps = [0, 10, 25, 50, 75, 99]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor='white')
    fig.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.10, hspace=0.40, wspace=0.30)
    axes_flat = axes.flatten()
    
    colors = ['#3498DB', '#2ECC71', '#F1C40F', '#E67E22', '#E74C3C', '#9B59B6']
    
    # 统一x轴范围
    all_data = []
    for ts in steps:
        data = read_raw_dat(DATA_DIR / f"{ts:04d}.dat")
        all_data.append(data.flatten())
    
    global_xlim = (min(d.min() for d in all_data) - 0.1, max(d.max() for d in all_data) + 0.1)
    
    for i, (ts, flat) in enumerate(zip(steps, all_data)):
        ax = axes_flat[i]
        ax.set_facecolor('white')
        
        # 计算CDF
        sorted_data = np.sort(flat)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 绘制CDF曲线
        ax.plot(sorted_data, cdf, color=colors[i], lw=2.5, alpha=0.85)
        ax.fill_between(sorted_data, cdf, alpha=0.15, color=colors[i])
        
        # 标注关键分位数
        for p, color_p in [(1, '#51cf66'), (5, '#51cf66'), (50, '#f1c40f'), (95, '#ff6b6b'), (99, '#ff6b6b')]:
            val = np.percentile(flat, p)
            ax.axvline(val, color=color_p, linestyle='--', alpha=0.5, lw=1)
            ax.text(val, 0.05 if p < 50 else 0.95, f'P{p}', 
                   ha='center', fontsize=7, color=color_p, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor=color_p, alpha=0.8))
        
        ax.set_xlim(global_xlim)
        ax.set_ylim(0, 1.05)
        ax.set_title(f't = {ts}', fontsize=14, fontweight='bold', color='#2C3E50', pad=10)
        ax.set_xlabel('密度值 ρ', fontsize=10, color='#555555')
        if i % 3 == 0:
            ax.set_ylabel('累积概率', fontsize=10, color='#555555')
        
        ax.tick_params(axis='both', labelsize=9, colors='#555555')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, color='#AAAAAA', linestyle='-', linewidth=0.5)
        
        # 统计信息
        mean_val = np.mean(flat)
        std_val = np.std(flat)
        ax.text(0.5, -0.22, f'μ={mean_val:.2f}  σ={std_val:.3f}', 
                transform=ax.transAxes, fontsize=9, ha='center', va='top', color='#555555',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', 
                         edgecolor='#CCCCCC', alpha=0.95))
    
    fig.suptitle('密度累积分布函数（CDF）演化序列', fontsize=17, fontweight='bold', 
                 color='#2C3E50', y=0.97)
    
    plt.savefig(OUT_DIR / 'task3_cdf_sequence.png', 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ task3_cdf_sequence.png 已生成")

if __name__ == '__main__':
    generate_cdf_sequence()
