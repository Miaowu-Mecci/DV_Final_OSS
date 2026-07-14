#!/usr/bin/env python3
"""
预处理所有时间步的投影图为 PNG 图片，供前端直接加载。
避免拖动时实时计算投影，实现即时响应。
"""

import numpy as np
import json
import os
from pathlib import Path
import struct

# 路径配置
DATA_DIR = Path("/home/zhb/DV_Final/data/Nyx")
PROCESSED_DIR = Path("/home/zhb/DV_Final/data/processed")
WEB_DATA_DIR = Path("/home/zhb/DV_Final/web/static/data")

# 确保输出目录存在
PROJECTIONS_DIR = WEB_DATA_DIR / "projections"
PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)
WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 元数据
NX, NY, NZ = 128, 128, 128
NTOTAL = 100


def read_raw_dat(filepath):
    """读取 Nyx 原始密度场 .dat 文件"""
    with open(filepath, 'rb') as f:
        raw = f.read()
    n_floats = len(raw) // 4
    data = struct.unpack('<' + 'f' * n_floats, raw)
    return np.array(data, dtype=np.float32).reshape((NX, NY, NZ))


def compute_projection(field, axis, proj_type='mip'):
    """计算投影图"""
    if proj_type == 'mip':
        return np.max(field, axis=axis)
    elif proj_type == 'aip':
        return np.mean(field, axis=axis)
    else:
        raise ValueError(f"Unknown projection type: {proj_type}")


def save_projection_png(data, filepath):
    """将投影数据保存为 PNG 图片（使用 matplotlib）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=64)
    im = ax.imshow(data, cmap='hot', origin='lower', aspect='auto')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(filepath, dpi=64, bbox_inches='tight', pad_inches=0, 
                facecolor='black', edgecolor='none')
    plt.close(fig)


def main():
    print(f"开始预处理 {NTOTAL} 个时间步的投影图...")
    
    for ts in range(NTOTAL):
        dat_path = DATA_DIR / f"{ts:04d}.dat"
        if not dat_path.exists():
            print(f"  跳过 t={ts} (文件不存在)")
            continue
        
        # 读取密度场
        field = read_raw_dat(dat_path)
        
        # 生成 6 个投影图
        for axis in range(3):
            axis_name = ['x', 'y', 'z'][axis]
            
            # MIP
            mip = compute_projection(field, axis, 'mip')
            mip_path = PROJECTIONS_DIR / f"mip_{axis_name}_{ts:04d}.png"
            save_projection_png(mip, mip_path)
            
            # AIP
            aip = compute_projection(field, axis, 'aip')
            aip_path = PROJECTIONS_DIR / f"aip_{axis_name}_{ts:04d}.png"
            save_projection_png(aip, aip_path)
        
        if ts % 10 == 0:
            print(f"  已处理 t={ts}")
    
    # 复制统计数据到 web 目录
    import shutil
    for fname in ['stats.json', 'histograms.json']:
        src = PROCESSED_DIR / fname
        dst = WEB_DATA_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"已复制 {fname} -> {dst}")
    
    print(f"\n完成！投影图保存在: {PROJECTIONS_DIR}")
    print(f"共生成 {NTOTAL * 6} 个 PNG 文件")


if __name__ == '__main__':
    main()
