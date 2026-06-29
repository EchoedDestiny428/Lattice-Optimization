import numpy as np
from scipy.ndimage import gaussian_filter
from config import RESOLUTION

def generate_random_lattice(freq_range=(0.8, 1.5), complexity=3):
    """
    Generates a generic lattice field using a sum of random harmonic functions.
    """
    x = np.linspace(0, 2 * np.pi * np.mean(freq_range), RESOLUTION, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    
    field = np.zeros_like(X)
    
    for _ in range(complexity):
        f = np.random.uniform(*freq_range)
        term = (np.sin(X * f) * np.cos(Y * f) + 
                np.sin(Y * f) * np.cos(Z * f) + 
                np.sin(Z * f) * np.cos(X * f))
        
        coeff = np.random.uniform(-1, 1)
        field += coeff * term
        
    field = gaussian_filter(field, sigma=0.5)
    field = (field - field.mean()) / (field.std() + 1e-8)
    return field

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