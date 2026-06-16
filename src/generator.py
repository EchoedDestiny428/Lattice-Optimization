# src/generator.py
import numpy as np
from scipy.ndimage import gaussian_filter
import config

def generate_tpms_voxels(equation_type='gyroid', t_threshold=0.0, sigma=1.0, cutoff=0.75):
    res = config.GRID_SIZE
    x = np.linspace(-np.pi, np.pi, res)
    y = np.linspace(-np.pi, np.pi, res)
    z = np.linspace(-np.pi, np.pi, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    if equation_type == 'gyroid':
        equation_matrix = np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X)

    elif equation_type == 'schwarz_p':
        equation_matrix = np.cos(X) + np.cos(Y) + np.cos(Z)

    else:
        raise ValueError(f"Unknown equation type: {equation_type}")
        
    raw_binary = (np.abs(equation_matrix) <= t_threshold).astype(np.float32)
    smoothed = gaussian_filter(raw_binary, sigma=sigma)
    final_voxels = (smoothed >= cutoff).astype(np.float32)
    return final_voxels


def generate_lattice(equation_type='gyroid', target_percent=30.0, tolerance=0.5):
    """
    Guess and check (15 times)
    
    """
    target_fraction = target_percent / 100.0
    
    low, high = 0.0, 1.3 # 1.3 because anything above 1.3 only has infinitely small mathematical points.
    
    for attempt in range(15):
        mid_t = (low + high) / 2.0
        voxels = generate_tpms_voxels(equation_type=equation_type, t_threshold=mid_t)
        
        current_fraction = np.sum(voxels == 1.0) / voxels.size
        
        if abs(current_fraction - target_fraction) <= (tolerance / 100.0):
            return voxels, current_fraction * 100.0
            
        if current_fraction < target_fraction:
            low = mid_t
        else:
            high = mid_t
            
    final_fraction = np.sum(voxels == 1.0) / voxels.size
    return voxels, final_fraction * 100.0