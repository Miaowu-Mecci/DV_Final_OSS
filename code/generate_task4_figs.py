#!/usr/bin/env python3
"""
生成任务四图像：交互式刷选可视化
1. 直方图框选示意图
2. 高密度筛选3D点云
3. 低密度筛选3D点云
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import struct
from matplotlib.patches import Rectangle

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

def generate_task4_figures():
    # 读取t=50数据
    data = read_raw_dat(DATA_DIR / '0050.dat')
    flat = data.flatten()
    
    # ========== 图1：直方图框选示意图 ==========
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ax.set_facecolor('white')
    
    # 绘制直方图
    n, bins, patches = ax.hist(flat, bins=80, color='#4facfe', edgecolor='white', alpha=0.7)
    
    # 高亮Top 1%区域 (p99以上的区域)
    p99 = np.percentile(flat, 99)
    for i, (patch, left_edge, right_edge) in enumerate(zip(patches, bins[:-1], bins[1:])):
        if left_edge >= p99:
            patch.set_facecolor('#ff6b6b')
            patch.set_alpha(0.9)
    
    # 框选区域标注
    ax.axvline(p99, color='#E74C3C', linestyle='--', lw=2, label=f'P99 = {p99:.2f}')
    ax.text(p99 + 0.1, max(n) * 0.9, f'Top 1%\n阈值: {p99:.2f}', 
            fontsize=10, color='#E74C3C', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE5E5', edgecolor='#E74C3C'))
    
    # 框选矩形
    rect = Rectangle((p99, 0), bins[-1]-p99, max(n)*1.05, 
                     linewidth=2, edgecolor='#E74C3C', facecolor='#FFE5E5', alpha=0.3)
    ax.add_patch(rect)
    
    ax.set_xlabel('密度值 ρ', fontsize=12, color='#333')
    ax.set_ylabel('频数', fontsize=12, color='#333')
    ax.set_title('直方图框选：Top 1% 高密度区间', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.tick_params(axis='both', labelsize=10, colors='#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='#AAA')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'task4_histogram_brush.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ task4_histogram_brush.png 已生成")
    
    # ========== 图2：Top 1%高密度3D点云 ==========
    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # 筛选Top 1%
    threshold = p99
    mask = data >= threshold
    x, y, z = np.where(mask)
    
    # 采样避免过多点
    if len(x) > 5000:
        idx = np.random.choice(len(x), 5000, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
    
    # 根据密度值着色
    vals = data[mask]
    if len(vals) > 5000:
        vals = vals[idx]
    
    scatter = ax.scatter(x, y, z, c=vals, cmap='hot', s=3, alpha=0.8)
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 128)
    ax.set_title('Top 1% 高密度体素三维分布 (t=50)', fontsize=14, fontweight='bold', color='white', pad=20)
    ax.set_xlabel('X', fontsize=10, color='white')
    ax.set_ylabel('Y', fontsize=10, color='white')
    ax.set_zlabel('Z', fontsize=10, color='white')
    ax.tick_params(axis='both', labelsize=8, colors='white')
    
    # 隐藏坐标轴背景
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('密度值', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'task4_top1_3d.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print("✓ task4_top1_3d.png 已生成")
    
    # ========== 图3：Bottom 10%低密度3D点云 ==========
    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    p10 = np.percentile(flat, 10)
    mask_low = data <= p10
    x_l, y_l, z_l = np.where(mask_low)
    
    if len(x_l) > 5000:
        idx = np.random.choice(len(x_l), 5000, replace=False)
        x_l, y_l, z_l = x_l[idx], y_l[idx], z_l[idx]
    
    vals_l = data[mask_low]
    if len(vals_l) > 5000:
        vals_l = vals_l[idx]
    
    scatter = ax.scatter(x_l, y_l, z_l, c=vals_l, cmap='cool', s=3, alpha=0.8)
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 128)
    ax.set_title('Bottom 10% 低密度体素三维分布 (t=50)', fontsize=14, fontweight='bold', color='white', pad=20)
    ax.set_xlabel('X', fontsize=10, color='white')
    ax.set_ylabel('Y', fontsize=10, color='white')
    ax.set_zlabel('Z', fontsize=10, color='white')
    ax.tick_params(axis='both', labelsize=8, colors='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('密度值', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'task4_bottom10_3d.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print("✓ task4_bottom10_3d.png 已生成")
    
    # ========== 图4：联动对比面板 ==========
    fig = plt.figure(figsize=(16, 7), facecolor='white')
    
    # 左：直方图框选
    ax1 = fig.add_subplot(131)
    ax1.set_facecolor('white')
    n, bins, patches = ax1.hist(flat, bins=80, color='#4facfe', edgecolor='white', alpha=0.7)
    p99 = np.percentile(flat, 99)
    for i, (patch, left_edge, right_edge) in enumerate(zip(patches, bins[:-1], bins[1:])):
        if left_edge >= p99:
            patch.set_facecolor('#ff6b6b')
            patch.set_alpha(0.9)
    ax1.axvline(p99, color='#E74C3C', linestyle='--', lw=2)
    rect = Rectangle((p99, 0), bins[-1]-p99, max(n)*1.05, 
                     linewidth=2, edgecolor='#E74C3C', facecolor='#FFE5E5', alpha=0.3)
    ax1.add_patch(rect)
    ax1.set_xlabel('密度值 ρ', fontsize=11)
    ax1.set_ylabel('频数', fontsize=11)
    ax1.set_title('(a) 直方图框选 Top 1%', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.2)
    
    # 中：Top 1% 3D
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_facecolor('black')
    threshold = p99
    mask = data >= threshold
    x, y, z = np.where(mask)
    if len(x) > 3000:
        idx = np.random.choice(len(x), 3000, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
    vals = data[mask]
    if len(vals) > 3000:
        vals = vals[idx]
    ax2.scatter(x, y, z, c=vals, cmap='hot', s=5, alpha=0.9)
    ax2.set_xlim(0, 128)
    ax2.set_ylim(0, 128)
    ax2.set_zlim(0, 128)
    ax2.set_title('(b) 三维高密度节点', fontsize=12, fontweight='bold', color='white', pad=10)
    ax2.set_xlabel('X', fontsize=9, color='white')
    ax2.set_ylabel('Y', fontsize=9, color='white')
    ax2.set_zlabel('Z', fontsize=9, color='white')
    ax2.tick_params(axis='both', labelsize=7, colors='white')
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.xaxis.pane.set_edgecolor('none')
    ax2.yaxis.pane.set_edgecolor('none')
    ax2.zaxis.pane.set_edgecolor('none')
    ax2.grid(False)
    
    # 右：切片验证
    ax3 = fig.add_subplot(133)
    ax3.set_facecolor('black')
    slice_z = data[:, :, 64]
    im = ax3.imshow(slice_z, cmap='hot', origin='lower', vmin=flat.min(), vmax=flat.max())
    # 叠加Top 1%区域轮廓
    mask_slice = slice_z >= threshold
    from skimage import measure
    try:
        contours = measure.find_contours(mask_slice, 0.5)
        for contour in contours:
            ax3.plot(contour[:, 1], contour[:, 0], 'cyan', linewidth=1.5, alpha=0.8)
    except:
        pass
    ax3.set_title('(c) Z=64切片 + 高密度轮廓', fontsize=12, fontweight='bold', color='white')
    ax3.set_xlabel('X', fontsize=10, color='white')
    ax3.set_ylabel('Y', fontsize=10, color='white')
    ax3.tick_params(axis='both', labelsize=8, colors='white')
    plt.colorbar(im, ax=ax3, shrink=0.6, label='密度')
    
    fig.suptitle('统计↔空间双向关联验证：Top 1% 高密度筛选', fontsize=15, fontweight='bold', color='#2C3E50', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_DIR / 'task4_linked_validation.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ task4_linked_validation.png 已生成")

if __name__ == '__main__':
    generate_task4_figures()
