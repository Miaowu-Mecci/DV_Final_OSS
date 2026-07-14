#!/usr/bin/env python3
"""
生成任务二图像：统计四面板 + 分位数剪刀差
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

STATS_PATH = Path('/home/zhb/DV_Final/data/processed/stats.json')
OUT_DIR = Path('/home/zhb/DV_Final/Document/figures')

def generate_task2_figures():
    with open(STATS_PATH) as f:
        STATS = json.load(f)
    
    steps = np.arange(100)
    means = np.array([s['mean'] for s in STATS])
    stds = np.array([s['std'] for s in STATS])
    maxs = np.array([s['max'] for s in STATS])
    mins = np.array([s['min'] for s in STATS])
    ranges = maxs - mins
    cvs = np.array([s['std']/s['mean']*100 for s in STATS])
    
    # ===== 图3: 统计四面板 =====
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 面板1: 均值
    ax = axes[0, 0]
    ax.plot(steps, means, color='#E74C3C', lw=2.5)
    ax.fill_between(steps, means, alpha=0.15, color='#E74C3C')
    ax.set_ylim(means.min()-0.02, means.max()+0.02)
    ax.set_title('密度均值：缓慢下降', fontsize=12, fontweight='bold')
    ax.set_xlabel('时间步'); ax.set_ylabel('密度值')
    for x in [10, 40, 70]:
        ax.axvline(x, color='gray', linestyle='--', alpha=0.4)
    
    # 面板2: 标准差
    ax = axes[0, 1]
    ax.plot(steps, stds, color='#3498DB', lw=2.5)
    ax.fill_between(steps, stds, alpha=0.15, color='#3498DB')
    ax.set_ylim(stds.min()-0.02, stds.max()+0.02)
    ax.set_title('标准差：持续增长', fontsize=12, fontweight='bold')
    ax.set_xlabel('时间步'); ax.set_ylabel('标准差')
    for x in [10, 40, 70]:
        ax.axvline(x, color='gray', linestyle='--', alpha=0.4)
    
    # 面板3: 极差
    ax = axes[1, 0]
    ax.plot(steps, ranges, color='#9B59B6', lw=2.5)
    ax.fill_between(steps, ranges, alpha=0.15, color='#9B59B6')
    ax.set_ylim(ranges.min()-0.3, ranges.max()+0.3)
    ax.set_title('极差：双向分离', fontsize=12, fontweight='bold')
    ax.set_xlabel('时间步'); ax.set_ylabel('极差')
    for x in [10, 40, 70]:
        ax.axvline(x, color='gray', linestyle='--', alpha=0.4)
    
    # 面板4: 变异系数
    ax = axes[1, 1]
    ax.plot(steps, cvs, color='#F39C12', lw=2.5)
    ax.fill_between(steps, cvs, alpha=0.15, color='#F39C12')
    ax.set_ylim(cvs.min()-0.2, cvs.max()+0.2)
    ax.set_title('变异系数（CV）', fontsize=12, fontweight='bold')
    ax.set_xlabel('时间步'); ax.set_ylabel('CV (%)')
    for x in [10, 40, 70]:
        ax.axvline(x, color='gray', linestyle='--', alpha=0.4)
    
    fig.suptitle('统计量时序演化：四阶段结构形成过程', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUT_DIR / 'task2_evolution_stats.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ task2_evolution_stats.png 已生成")
    
    # ===== 图4: 分位数剪刀差 =====
    p1s = np.array([s['p1'] for s in STATS])
    p5s = np.array([s['p5'] for s in STATS])
    p50s = np.array([s['median'] for s in STATS])
    p95s = np.array([s['p95'] for s in STATS])
    p99s = np.array([s['p99'] for s in STATS])
    p999s = np.array([s['p999'] for s in STATS])
    
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
    
    # 半透明填充带
    ax.fill_between(steps, p1s, p999s, alpha=0.08, color='gray', label='P999--P1')
    ax.fill_between(steps, p5s, p99s, alpha=0.15, color='steelblue', label='P99--P5')
    ax.fill_between(steps, p5s, p95s, alpha=0.25, color='#3498DB', label='P95--P5')
    
    # 曲线
    ax.plot(steps, p999s, color='#9B59B6', lw=2, label='P999')
    ax.plot(steps, p99s, color='#E74C3C', lw=2, label='P99')
    ax.plot(steps, p95s, color='#E67E22', lw=2, label='P95')
    ax.plot(steps, p50s, color='#2ECC71', lw=2.5, linestyle='-', label='P50（中位数）')
    ax.plot(steps, means, color='#1ABC9C', lw=2.5, linestyle='--', label='均值')
    ax.plot(steps, p5s, color='#3498DB', lw=2, label='P5')
    ax.plot(steps, p1s, color='#2980B9', lw=2, label='P1')
    
    for x in [10, 40, 70]:
        ax.axvline(x, color='gray', linestyle='--', alpha=0.4)
    
    ax.set_title('分位数偏移演化：P1/P5 下降、P95/P99/P999 上升', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间步', fontsize=12); ax.set_ylabel('密度值', fontsize=12)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2, color='#AAAAAA')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'task2_quantile_shifts.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ task2_quantile_shifts.png 已生成")

if __name__ == '__main__':
    generate_task2_figures()
