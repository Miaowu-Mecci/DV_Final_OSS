#!/usr/bin/env python3
"""
重新生成图6：对数直方图演化序列 - 变化更明显，叠加对比
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

def generate_enhanced_log_histogram():
    steps = [0, 10, 25, 50, 75, 99]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
    fig.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.10, hspace=0.40, wspace=0.30)
    axes_flat = axes.flatten()
    
    # 先读取所有数据
    all_log_data = []
    for ts in steps:
        data = read_raw_dat(DATA_DIR / f"{ts:04d}.dat")
        flat = data.flatten()
        log_flat = np.log10(flat)
        all_log_data.append(log_flat)
    
    # 统一x轴
    log_min = min(np.min(d) for d in all_log_data)
    log_max = max(np.max(d) for d in all_log_data)
    log_xlim = (log_min - 0.02, log_max + 0.02)
    
    # 计算全局统一的y轴上限，保证比例一致
    global_ymax = 0
    for log_flat in all_log_data:
        n, _ = np.histogram(log_flat, bins=60, range=log_xlim, density=True)
        global_ymax = max(global_ymax, max(n))
    
    colors = ['#3498DB', '#2ECC71', '#F1C40F', '#E67E22', '#E74C3C', '#9B59B6']
    
    for i, (ts, log_flat) in enumerate(zip(steps, all_log_data)):
        ax = axes_flat[i]
        ax.set_facecolor('white')
        
        # 绘制直方图
        n, bins, patches = ax.hist(log_flat, bins=60, range=log_xlim,
                                    color=colors[i], edgecolor='white', 
                                    alpha=0.85, density=True, linewidth=0.3)
        
        # 统一坐标轴
        ax.set_xlim(log_xlim)
        ax.set_ylim(0, global_ymax * 1.15)
        
        ax.set_title(f't = {ts}', fontsize=14, fontweight='bold', 
                     color='#2C3E50', pad=10)
        ax.set_xlabel('log₁₀(ρ)', fontsize=11, color='#555555')
        if i % 3 == 0:
            ax.set_ylabel('概率密度', fontsize=11, color='#555555')
        
        # x轴精确到0.0001
        from matplotlib.ticker import FormatStrFormatter
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        ax.tick_params(axis='both', labelsize=9, colors='#555555')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, color='#AAAAAA', linestyle='-', linewidth=0.5)
        
        # 统计信息
        flat = np.power(10, log_flat)  # 还原原始密度
        mean_val = np.mean(flat)
        std_val = np.std(flat)
        ax.text(0.5, -0.22, f'μ={mean_val:.2f}  σ={std_val:.3f}', 
                transform=ax.transAxes, fontsize=9, ha='center', va='top', 
                color='#555555',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', 
                         edgecolor='#CCCCCC', alpha=0.95))
        
        # 标注峰值位置
        peak_idx = np.argmax(n)
        peak_x = (bins[peak_idx] + bins[peak_idx + 1]) / 2
        peak_y = n[peak_idx]
        ax.plot(peak_x, peak_y, 'o', color='#E74C3C', markersize=6, 
                markeredgecolor='white', markeredgewidth=1, zorder=5)
        ax.text(peak_x, peak_y + global_ymax * 0.08, f'峰:{peak_x:.3f}', 
                ha='center', fontsize=8, color='#E74C3C', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='#E74C3C', alpha=0.9))
    
    fig.suptitle('密度对数直方图演化序列', fontsize=17, fontweight='bold', 
                 color='#2C3E50', y=0.97)
    
    plt.savefig(OUT_DIR / 'task3_log_histogram_sequence.png', 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # ============ 叠加对比图 ============
    fig2, ax2 = plt.subplots(figsize=(12, 7), facecolor='white')
    ax2.set_facecolor('white')
    
    for i, (ts, log_flat) in enumerate(zip(steps, all_log_data)):
        n, bins = np.histogram(log_flat, bins=80, range=log_xlim, density=True)
        centers = (bins[:-1] + bins[1:]) / 2
        ax2.plot(centers, n, color=colors[i], lw=2.5, label=f't={ts}', alpha=0.8)
        ax2.fill_between(centers, n, alpha=0.15, color=colors[i])
    
    ax2.set_xlim(log_xlim)
    ax2.set_ylim(0, global_ymax * 1.25)
    ax2.set_xlabel('log₁₀(ρ)', fontsize=12, color='#333333')
    ax2.set_ylabel('概率密度', fontsize=12, color='#333333')
    ax2.set_title('对数密度分布演化叠加对比', fontsize=15, fontweight='bold', color='#2C3E50')
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2.tick_params(axis='both', labelsize=10, colors='#555555')
    ax2.spines['bottom'].set_color('#CCCCCC')
    ax2.spines['left'].set_color('#CCCCCC')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.2, color='#AAAAAA', linestyle='-', linewidth=0.5)
    
    plt.savefig(OUT_DIR / 'task3_log_histogram_overlay.png', 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ task3_log_histogram_sequence.png 已生成（增强版）")
    print(f"✓ task3_log_histogram_overlay.png 已生成（叠加对比）")

if __name__ == '__main__':
    generate_enhanced_log_histogram()
