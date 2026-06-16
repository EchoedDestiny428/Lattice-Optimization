# src/physics.py
import numpy as np

def evaluate_physical_stats(voxel_grid, material_profile, ai_efficiency_score=1.0):
    """
    Hypothetical physical properties
    Uses Gibson-Ashby scaling laws for cellular solids for est.
    """
    total_voxels = voxel_grid.size
    solid_voxels = np.sum(voxel_grid == 1.0)
    relative_density = solid_voxels / total_voxels
    
    E_solid_mpa = material_profile['youngs_modulus_gpa'] * 1000  
    sigma_solid_mpa = material_profile['yield_strength_mpa']     
    
    C_stiffness, n_stiffness = 0.35, 2.0
    C_strength, m_strength = 0.25, 1.5
    
    E_baseline = E_solid_mpa * C_stiffness * (relative_density ** n_stiffness)
    sigma_baseline = sigma_solid_mpa * C_strength * (relative_density ** m_strength)
    
    # Apply AI Shape Efficiency Multiplier
    E_lattice_actual = E_baseline * ai_efficiency_score
    sigma_lattice_actual = sigma_baseline * ai_efficiency_score
    
    # Lock Bounding Envelope to Exactly 1 cm (10.0 mm)
    # Cross-sectional Area = 10mm * 10mm = 100 mm^2
    area_mm2 = 100.0 
    radius_meters = 0.005 
    
    peak_force_newtons = sigma_lattice_actual * area_mm2
    peak_torque_nm = peak_force_newtons * radius_meters
    
    return {
        "Relative Density": relative_density,
        "AI Efficiency Factor": ai_efficiency_score,
        "Effective Stiffness (MPa)": E_lattice_actual,
        "Peak Hypothetical Force Capacity (Newtons)": peak_force_newtons,
        "Peak Hypothetical Torque Capacity (Newton-Meters)": peak_torque_nm
    }