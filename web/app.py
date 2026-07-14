#!/usr/bin/env python3
"""
Nyx Cosmological Visualization Dashboard
Flask backend (non-separated frontend/backend)
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
import json
import base64
from pathlib import Path
import os

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = Path('/home/zhb/DV_Final/data')
PROCESSED_DIR = DATA_DIR / 'processed'
NYX_DIR = DATA_DIR / 'Nyx'

GRID = 128
N_STEPS = 100

# Load precomputed stats
with open(PROCESSED_DIR / 'stats.json') as f:
    STATS = json.load(f)

with open(PROCESSED_DIR / 'histograms.json') as f:
    HISTOGRAMS = json.load(f)

# Cache for raw data to avoid repeated disk reads
_data_cache = {}

def load_timestep(idx):
    if idx in _data_cache:
        return _data_cache[idx]
    path = NYX_DIR / f"{idx:04d}.dat"
    data = np.fromfile(path, dtype=np.float32).reshape((GRID, GRID, GRID))
    _data_cache[idx] = data
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def api_stats():
    return jsonify(STATS)

@app.route('/api/histograms')
def api_histograms():
    return jsonify(HISTOGRAMS)

@app.route('/api/timestep/<int:idx>/volume')
def api_volume(idx):
    """Return 64^3 downsampled volume data for WebGL texture"""
    if not 0 <= idx < N_STEPS:
        return jsonify({'error': 'Invalid timestep'}), 400
    vol_path = PROCESSED_DIR / 'volumes' / f"vol_{idx:04d}.npz"
    if vol_path.exists():
        vol = np.load(vol_path)
        return jsonify({
            'data': vol['data'].tolist(),
            'shape': vol['data'].shape,
            'raw_min': float(vol['raw_min']),
            'raw_max': float(vol['raw_max']),
            'raw_mean': float(vol['raw_mean'])
        })
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/timestep/<int:idx>/volume_b64')
def api_volume_b64(idx):
    """Return 64^3 downsampled volume as base64 binary (efficient for WebGL)"""
    if not 0 <= idx < N_STEPS:
        return jsonify({'error': 'Invalid timestep'}), 400
    vol_path = PROCESSED_DIR / 'volumes' / f"vol_{idx:04d}.npz"
    if vol_path.exists():
        vol = np.load(vol_path)
        raw = vol['data'].tobytes()
        return jsonify({
            'timestep': idx,
            'data_b64': base64.b64encode(raw).decode('ascii'),
            'shape': [int(x) for x in vol['data'].shape],
            'raw_min': float(vol['raw_min']),
            'raw_max': float(vol['raw_max']),
            'raw_mean': float(vol['raw_mean'])
        })
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/timestep/<int:idx>/filter')
def api_filter(idx):
    """Filter voxels by density range and return sparse point cloud"""
    if not 0 <= idx < N_STEPS:
        return jsonify({'error': 'Invalid timestep'}), 400
    
    dmin = request.args.get('min', type=float)
    dmax = request.args.get('max', type=float)
    max_points = request.args.get('max_points', 50000, type=int)
    
    data = load_timestep(idx)
    
    if dmin is None:
        dmin = data.min()
    if dmax is None:
        dmax = data.max()
    
    mask = (data >= dmin) & (data <= dmax)
    coords = np.argwhere(mask)
    values = data[mask]
    
    total = len(coords)
    
    # Subsample if too many points
    if len(coords) > max_points:
        indices = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[indices]
        values = values[indices]
    
    # Normalize coordinates to [-1, 1]
    coords_norm = (coords / (GRID - 1)) * 2 - 1
    
    # Normalize values to [0, 1] for color mapping
    v_norm = (values - data.min()) / (data.max() - data.min() + 1e-8)
    
    return jsonify({
        'points': coords_norm.tolist(),
        'values': v_norm.tolist(),
        'raw_values': values.tolist(),
        'total_matching': int(total),
        'returned': len(coords),
        'density_range': [float(dmin), float(dmax)]
    })

@app.route('/api/timestep/<int:idx>/percentile_range')
def api_percentile_range(idx):
    """Get density range for a given percentile range, e.g. top 1%"""
    if not 0 <= idx < N_STEPS:
        return jsonify({'error': 'Invalid timestep'}), 400
    
    low_p = request.args.get('low_p', 99.0, type=float)
    high_p = request.args.get('high_p', 100.0, type=float)
    
    data = load_timestep(idx)
    dmin = np.percentile(data, low_p)
    dmax = np.percentile(data, high_p)
    
    return jsonify({
        'timestep': idx,
        'percentile_range': [low_p, high_p],
        'density_range': [float(dmin), float(dmax)],
        'count_estimate': int(np.sum((data >= dmin) & (data <= dmax)))
    })

@app.route('/api/timestep/<int:idx>/slice')
def api_slice(idx):
    """Return a 2D slice for quick preview"""
    if not 0 <= idx < N_STEPS:
        return jsonify({'error': 'Invalid timestep'}), 400
    axis = request.args.get('axis', 0, type=int)
    pos = request.args.get('pos', 64, type=int)
    
    data = load_timestep(idx)
    if axis == 0:
        sl = data[pos, :, :]
    elif axis == 1:
        sl = data[:, pos, :]
    else:
        sl = data[:, :, pos]
    
    return jsonify({
        'slice': sl.tolist(),
        'shape': sl.shape,
        'min': float(sl.min()),
        'max': float(sl.max())
    })

@app.route('/api/timestep/<int:idx>/projection')
def api_projection(idx):
    """Return MIP or AIP projection as 2D array for interactive Plotly heatmap"""
    if not 0 <= idx < N_STEPS:
        return jsonify({'error': 'Invalid timestep'}), 400
    axis = request.args.get('axis', 2, type=int)  # 0=x, 1=y, 2=z
    ptype = request.args.get('type', 'mip')       # 'mip' or 'aip'
    
    data = load_timestep(idx)
    if ptype == 'mip':
        proj = np.max(data, axis=axis)
    else:
        proj = np.mean(data, axis=axis)
    
    return jsonify({
        'projection': proj.tolist(),
        'shape': proj.shape,
        'axis': axis,
        'type': ptype,
        'min': float(proj.min()),
        'max': float(proj.max())
    })

@app.route('/api/timestep/<int:idx>/histogram')
def api_histogram(idx):
    """Return histogram data for a single timestep"""
    if not 0 <= idx < N_STEPS:
        return jsonify({'error': 'Invalid timestep'}), 400
    bins = request.args.get('bins', 200, type=int)
    
    data = load_timestep(idx)
    hist, edges = np.histogram(data.flatten(), bins=bins)
    log_hist, log_edges = np.histogram(np.log10(data - data.min() + 1e-8), bins=bins)
    
    return jsonify({
        'timestep': idx,
        'hist': hist.tolist(),
        'edges': edges.tolist(),
        'log_hist': log_hist.tolist(),
        'log_edges': log_edges.tolist(),
        'stats': {
            'min': float(data.min()),
            'max': float(data.max()),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'median': float(np.median(data)),
            'p1': float(np.percentile(data, 1)),
            'p5': float(np.percentile(data, 5)),
            'p95': float(np.percentile(data, 95)),
            'p99': float(np.percentile(data, 99)),
        }
    })

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(PROCESSED_DIR / 'projections', filename)

@app.route('/summary/<path:filename>')
def serve_summary(filename):
    return send_from_directory(PROCESSED_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
