import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
import matplotlib.pyplot as plt

RESOLUTION = 48

def generate_harmonic_field(complexity=3):
    x = np.linspace(0, 2 * np.pi * 1.0, RESOLUTION, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    field = np.zeros_like(X)
    for _ in range(complexity):
        f = np.random.uniform(0.8, 1.5)
        term = (np.sin(X * f) * np.cos(Y * f) + np.sin(Y * f) * np.cos(Z * f) + np.sin(Z * f) * np.cos(X * f))
        coeff = np.random.uniform(-1, 1)
        field += coeff * term
    field = gaussian_filter(field, sigma=0.5)
    return (field - field.mean()) / (field.std() + 1e-8)

def generate_voronoi_field(num_seeds=50):
    grid = np.ones((RESOLUTION, RESOLUTION, RESOLUTION), dtype=bool)
    # Place random seeds
    for _ in range(num_seeds):
        x, y, z = np.random.randint(0, RESOLUTION, size=3)
        grid[x, y, z] = False
    
    # Distance to nearest seed
    # The walls of the voronoi cells are where distance is MAXIMAL
    dist = distance_transform_edt(grid)
    # We want a field where high values = solid material (or vice versa)
    # By making field = dist, higher threshold means keeping only the boundaries between cells
    return (dist - dist.mean()) / (dist.std() + 1e-8)

def generate_strut_field(freq=2):
    # Generates a field where high values are near the center of struts
    x = np.linspace(0, freq * np.pi, RESOLUTION, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    
    # Distance to nearest grid planes
    dx = np.abs(np.sin(X))
    dy = np.abs(np.sin(Y))
    dz = np.abs(np.sin(Z))
    
    # Struts exist where two planes intersect (e.g. dx and dy are both small)
    dist_z_strut = dx**2 + dy**2
    dist_y_strut = dx**2 + dz**2
    dist_x_strut = dy**2 + dz**2
    
    # Field is the minimum distance to ANY strut
    field = np.minimum(np.minimum(dist_x_strut, dist_y_strut), dist_z_strut)
    # Invert so higher values are solid (center of strut)
    field = -field
    return (field - field.mean()) / (field.std() + 1e-8)

def generate_noise_field():
    # Pure random noise blurred out
    noise = np.random.randn(RESOLUTION, RESOLUTION, RESOLUTION)
    field = gaussian_filter(noise, sigma=np.random.uniform(1.5, 3.0))
    return (field - field.mean()) / (field.std() + 1e-8)

print("Testing generators...")
print("Harmonic:", generate_harmonic_field().shape)
print("Voronoi:", generate_voronoi_field().shape)
print("Struts:", generate_strut_field().shape)
print("Noise:", generate_noise_field().shape)
print("All continuous fields passed.")
