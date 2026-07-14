#!/usr/bin/env python3
"""
Nyx Dataset Preprocessing
Reads 100 timesteps of 128^3 float32 density data and generates:
1. Statistics per timestep (JSON)
2. Histograms per timestep (JSON)
3. Volume-rendered projections (PNG)
4. Downsampled volumes for web rendering (NPZ)
"""

import numpy as np
import os
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure

DATA_DIR = Path('/home/zhb/DV_Final/data/Nyx')
OUT_DIR = Path('/home/zhb/DV_Final/data/processed')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STEPS = 100
GRID = 128
TOTAL_VOXELS = GRID ** 3

def load_timestep(idx):
    path = DATA_DIR / f"{idx:04d}.dat"
    return np.fromfile(path, dtype=np.float32).reshape((GRID, GRID, GRID))

def compute_histogram(data, bins=200):
    # Use log10 density histogram since data seems to be log-scale
    # But let's just histogram the raw values
    hist, edges = np.histogram(data.flatten(), bins=bins, range=(data.min(), data.max()))
    return hist.tolist(), edges.tolist()

def compute_histogram_log(data, bins=200):
    # Log-space histogram
    log_data = np.log10(data - data.min() + 1e-6)
    hist, edges = np.histogram(log_data.flatten(), bins=bins)
    return hist.tolist(), edges.tolist()

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
        'p95': float(np.percentile(flat, 95)),
        'p99': float(np.percentile(flat, 99)),
        'p999': float(np.percentile(flat, 99.9)),
    }

def generate_mip_projection(data, axis=0, cmap='hot'):
    """Maximum Intensity Projection"""
    mip = np.max(data, axis=axis)
    return mip

def generate_aip_projection(data, axis=0):
    """Average Intensity Projection"""
    return np.mean(data, axis=axis)

def save_projection_image(data, out_path, cmap='hot', title=''):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(data, cmap=cmap, origin='lower')
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_volume_raycast_views(data, out_dir, idx):
    """Generate MIP projections along 3 axes + a 3D isosurface snapshot"""
    # MIP along Z (axis 2)
    mip_z = np.max(data, axis=2)
    save_projection_image(mip_z, out_dir / f"mip_z_{idx:04d}.png", title=f"Timestep {idx} - MIP Z")
    
    # MIP along X (axis 0) 
    mip_x = np.max(data, axis=0)
    save_projection_image(mip_x, out_dir / f"mip_x_{idx:04d}.png", title=f"Timestep {idx} - MIP X")
    
    # MIP along Y (axis 1)
    mip_y = np.max(data, axis=1)
    save_projection_image(mip_y, out_dir / f"mip_y_{idx:04d}.png", title=f"Timestep {idx} - MIP Y")
    
    # Average projection along Z
    aip_z = np.mean(data, axis=2)
    save_projection_image(aip_z, out_dir / f"aip_z_{idx:04d}.png", cmap='viridis', title=f"Timestep {idx} - AIP Z")

def downsample_for_web(data, target=64):
    factor = GRID // target
    if factor == 1:
        return data
    return ndimage.zoom(data, 1/factor, order=1)

def generate_composite_timeline(out_dir, selected_steps):
    """Create a composite figure showing evolution"""
    n = len(selected_steps)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    
    for col, idx in enumerate(selected_steps):
        data = load_timestep(idx)
        
        # Top row: MIP Z
        mip = np.max(data, axis=2)
        ax = axes[0, col]
        im = ax.imshow(mip, cmap='hot', origin='lower')
        ax.set_title(f"t = {idx}")
        ax.axis('off')
        
        # Bottom row: histogram
        ax = axes[1, col]
        ax.hist(data.flatten(), bins=100, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_xlabel('Density')
        ax.set_ylabel('Count')
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'timeline_composite.png', dpi=200, bbox_inches='tight')
    plt.close()

def main():
    print("Starting preprocessing...")
    
    all_stats = []
    all_histograms = []
    all_log_histograms = []
    
    # Selected steps for detailed projections (start, early, mid, late, end)
    selected_steps = [0, 10, 25, 50, 75, 99]
    proj_dir = OUT_DIR / 'projections'
    proj_dir.mkdir(exist_ok=True)
    
    vol_dir = OUT_DIR / 'volumes'
    vol_dir.mkdir(exist_ok=True)
    
    for i in range(N_STEPS):
        print(f"Processing timestep {i:04d}...")
        data = load_timestep(i)
        
        # Statistics
        stats = compute_stats(data)
        stats['timestep'] = i
        all_stats.append(stats)
        
        # Histograms
        hist, edges = compute_histogram(data, bins=200)
        all_histograms.append({'timestep': i, 'hist': hist, 'edges': edges})
        
        log_hist, log_edges = compute_histogram_log(data, bins=200)
        all_log_histograms.append({'timestep': i, 'hist': log_hist, 'edges': log_edges})
        
        # Projections for selected steps
        if i in selected_steps:
            generate_volume_raycast_views(data, proj_dir, i)
        
        # Save downsampled volume for web (64^3, uint8 normalized)
        ds = downsample_for_web(data, target=64)
        # Normalize to 0-255 for texture
        norm = (ds - ds.min()) / (ds.max() - ds.min() + 1e-8)
        uint8_vol = (norm * 255).astype(np.uint8)
        np.savez_compressed(vol_dir / f"vol_{i:04d}.npz", data=uint8_vol, 
                           raw_min=float(data.min()), raw_max=float(data.max()),
                           raw_mean=float(data.mean()))
    
    # Save aggregated stats
    with open(OUT_DIR / 'stats.json', 'w') as f:
        json.dump(all_stats, f)
    
    with open(OUT_DIR / 'histograms.json', 'w') as f:
        json.dump(all_histograms, f)
    
    with open(OUT_DIR / 'log_histograms.json', 'w') as f:
        json.dump(all_log_histograms, f)
    
    # Generate composite timeline
    generate_composite_timeline(OUT_DIR, [0, 25, 50, 75, 99])
    
    # Generate evolution plots
    print("Generating evolution plots...")
    steps = np.arange(N_STEPS)
    means = [s['mean'] for s in all_stats]
    stds = [s['std'] for s in all_stats]
    mins = [s['min'] for s in all_stats]
    maxs = [s['max'] for s in all_stats]
    p99s = [s['p99'] for s in all_stats]
    p1s = [s['p1'] for s in all_stats]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.plot(steps, means, label='Mean', color='blue')
    ax.fill_between(steps, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.3)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Density')
    ax.set_title('Mean Density Evolution')
    ax.legend()
    
    ax = axes[0, 1]
    ax.plot(steps, stds, label='Std Dev', color='red')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Density Fluctuation Evolution')
    ax.legend()
    
    ax = axes[1, 0]
    ax.plot(steps, mins, label='Min', color='green')
    ax.plot(steps, maxs, label='Max', color='purple')
    ax.plot(steps, p99s, label='P99', color='orange')
    ax.plot(steps, p1s, label='P1', color='cyan')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Density')
    ax.set_title('Extrema Evolution')
    ax.legend()
    ax.set_yscale('log')
    
    ax = axes[1, 1]
    # Log histogram evolution heatmap
    hist_matrix = np.array([h['hist'] for h in all_histograms])
    # Use common bins from first timestep
    bin_centers = np.array(all_histograms[0]['edges'][:-1])
    im = ax.imshow(hist_matrix.T, aspect='auto', origin='lower', cmap='magma',
                   extent=[0, 99, bin_centers[0], bin_centers[-1]])
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Density')
    ax.set_title('Histogram Evolution (Heatmap)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'evolution_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Preprocessing complete!")

if __name__ == '__main__':
    main()
