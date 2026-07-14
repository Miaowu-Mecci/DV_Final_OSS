#!/usr/bin/env python3
"""
用 yt 工具包生成 Nyx 宇宙学模拟数据的高质量体渲染图片
参考 yt 的 Volume Rendering API:
  - yt.create_scene() 创建场景
  - ColorTransferFunction 定义传递函数（对数空间）
  - 相机控制、光源、采样步长
"""

import os
import sys
import glob
import numpy as np
import yt

DATA_DIR = "/home/zhb/DV_Final/data/Nyx"
OUTPUT_DIR = "/home/zhb/DV_Final/Document/yt_renders"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# yt 4.x volume rendering
# 加载单个时间步并渲染
def render_timestep(idx, cmap='hot', n_layers=5):
    pattern = os.path.join(DATA_DIR, f"{idx:04d}.dat")
    files = glob.glob(pattern)
    if not files:
        print(f"[!] No data file found for timestep {idx}")
        return None
    path = files[0]
    
    # 读取 128^3 float32 密度场
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != 128**3:
        print(f"[!] Unexpected data size: {raw.size} (expected {128**3})")
        return None
    data = raw.reshape(128, 128, 128)
    
    # 创建 yt Uniform Grid Dataset
    # bbox: [xmin, xmax], [ymin, ymax], [zmin, zmax]
    bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    data_dict = {"Density": (data, "g/cm**3")}
    ds = yt.load_uniform_grid(data_dict, data.shape, length_unit="Mpc", bbox=bbox)
    
    # 创建场景
    sc = yt.create_scene(ds, field="Density")
    source = sc.sources["source_00"]
    
    # 传递函数：在对数空间工作（Nyx 密度值域极大）
    dmin, dmax = float(data.min()), float(data.max())
    log_bounds = (np.log10(max(dmin, 1e-10)), np.log10(dmax))
    
    tf = yt.ColorTransferFunction(log_bounds)
    # add_layers: 自动分层，每层的颜色和透明度均匀分布
    tf.add_layers(n_layers, colormap=cmap)
    
    source.set_transfer_function(tf)
    source.tfh.set_log(True)
    source.tfh.bounds = log_bounds
    
    # 相机设置：等轴测视角
    cam = sc.camera
    cam.position = np.array([0.5, 0.5, 2.5])  # 前方
    cam.focus = np.array([0.5, 0.5, 0.5])
    cam.resolution = (1024, 1024)
    
    # 渲染并保存
    sc.render()
    out_path = os.path.join(OUTPUT_DIR, f"yt_render_t{idx:04d}_{cmap}.png")
    sc.save(out_path, sigma_clip=2.0)
    print(f"[OK] Saved: {out_path}")
    return out_path


def render_all_key_timesteps():
    """渲染关键时间步，多种 colormap"""
    key_indices = [0, 25, 50, 75, 99]
    colormaps = ['hot', 'cool', 'algae', 'kamae']
    
    for idx in key_indices:
        for cmap in colormaps:
            try:
                render_timestep(idx, cmap=cmap, n_layers=5)
            except Exception as e:
                print(f"[!] Failed t={idx} cmap={cmap}: {e}")


def render_comparison_transfer_functions():
    """生成传递函数对比图（不同分层数 + 对数/线性）"""
    idx = 50
    for n_layers in [3, 5, 8]:
        for log_scale in [True, False]:
            try:
                pattern = os.path.join(DATA_DIR, f"{idx:04d}.dat")
                files = glob.glob(pattern)
                if not files:
                    continue
                raw = np.fromfile(files[0], dtype=np.float32)
                data = raw.reshape(128, 128, 128)
                
                bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
                ds = yt.load_uniform_grid({"Density": (data, "g/cm**3")}, data.shape, length_unit="Mpc", bbox=bbox)
                sc = yt.create_scene(ds, field="Density")
                source = sc.sources["source_00"]
                
                dmin, dmax = float(data.min()), float(data.max())
                if log_scale:
                    bounds = (np.log10(max(dmin, 1e-10)), np.log10(dmax))
                else:
                    bounds = (dmin, dmax)
                
                tf = yt.ColorTransferFunction(bounds)
                tf.add_layers(n_layers, colormap='hot')
                source.set_transfer_function(tf)
                source.tfh.set_log(log_scale)
                source.tfh.bounds = bounds
                
                sc.camera.position = np.array([0.5, 0.5, 2.5])
                sc.camera.focus = np.array([0.5, 0.5, 0.5])
                sc.camera.resolution = (1024, 1024)
                
                sc.render()
                suffix = f"log_{n_layers}layers" if log_scale else f"lin_{n_layers}layers"
                out = os.path.join(OUTPUT_DIR, f"yt_tf_compare_t50_{suffix}.png")
                sc.save(out, sigma_clip=2.0)
                print(f"[OK] {out}")
            except Exception as e:
                print(f"[!] Compare failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("yt Volume Rendering for Nyx Cosmological Simulation")
    print("=" * 60)
    
    # 先测试单个时间步
    print("\n[1] Test render t=50 (hot colormap)...")
    render_timestep(50, cmap='hot', n_layers=5)
    
    # 渲染所有关键时间步
    print("\n[2] Render key timesteps with multiple colormaps...")
    render_all_key_timesteps()
    
    # 传递函数对比
    print("\n[3] Transfer function comparison...")
    render_comparison_transfer_functions()
    
    print("\n[Done] All renders saved to:", OUTPUT_DIR)
