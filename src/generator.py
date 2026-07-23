import numpy as np
from scipy.ndimage import gaussian_filter
from config import RESOLUTION

from scipy.ndimage import gaussian_filter, distance_transform_edt

def generate_harmonic_field(complexity=3):
    """Generates a generic lattice field using a sum of random harmonic functions."""
    x = np.linspace(0, 2 * np.pi * 1.0, RESOLUTION, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    
    field = np.zeros_like(X)
    
    for _ in range(complexity):
        f = np.random.uniform(0.8, 1.5)
        term = (np.sin(X * f) * np.cos(Y * f) + 
                np.sin(Y * f) * np.cos(Z * f) + 
                np.sin(Z * f) * np.cos(X * f))
        
        coeff = np.random.uniform(-1, 1)
        field += coeff * term
        
    field = gaussian_filter(field, sigma=0.5)
    return (field - field.mean()) / (field.std() + 1e-8)

def generate_voronoi_field(complexity=3):
    """Generates an open-cell Voronoi foam."""
    num_seeds = complexity * 15 # e.g. 45 to 75 seeds
    grid = np.ones((RESOLUTION, RESOLUTION, RESOLUTION), dtype=bool)
    for _ in range(num_seeds):
        x, y, z = np.random.randint(0, RESOLUTION, size=3)
        grid[x, y, z] = False
    
    dist = distance_transform_edt(grid)
    return (dist - dist.mean()) / (dist.std() + 1e-8)

def generate_strut_field(complexity=3):
    """Generates rigid intersecting struts (Cubic/Grid style)."""
    freq = complexity
    x = np.linspace(0, freq * np.pi, RESOLUTION, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    
    dx = np.abs(np.sin(X))
    dy = np.abs(np.sin(Y))
    dz = np.abs(np.sin(Z))
    
    dist_z_strut = dx**2 + dy**2
    dist_y_strut = dx**2 + dz**2
    dist_x_strut = dy**2 + dz**2
    
    field = np.minimum(np.minimum(dist_x_strut, dist_y_strut), dist_z_strut)
    field = -field # Invert so higher values are solid (center of strut)
    return (field - field.mean()) / (field.std() + 1e-8)

def generate_noise_field(complexity=3):
    """Generates random continuous organic blobs."""
    noise = np.random.randn(RESOLUTION, RESOLUTION, RESOLUTION)
    sigma = max(1.0, 4.0 - complexity * 0.5) # Higher complexity = smaller blobs
    field = gaussian_filter(noise, sigma=sigma)
    return (field - field.mean()) / (field.std() + 1e-8)

def find_threshold(field, target_density):
    """
    Binary search for the isosurface threshold.
    Since field is normalized to mean 0, std 1, the range [-3, 3] is safe.
    """
    lo, hi = -3.0, 3.0
    best_threshold = 0.0
    
    for _ in range(25): # 25 iterations = high precision
        mid = (lo + hi) / 2.0
        # Calculate density: field < threshold means solid
        current_density = np.mean(field < mid)
        
        if abs(current_density - target_density) < 0.001:
            best_threshold = mid
            break
            
        if current_density < target_density:
            lo = mid
        else:
            hi = mid
        best_threshold = mid
        
    return best_threshold