# main.py
import src.config as config
from src.generator import generate_lattice_by_percentage
from src.physics import evaluate_physical_stats

def main():
    chosen_lattice_type = 'gyroid'
    desired_fill_percent = 50.0

    selected_material_key = 'test_material' 
    custom_material = config.MATERIAL_PROFILES[selected_material_key]
    
    print(f"\n[GEOMETRY] Running binary search to hit {desired_fill_percent}% fill...")
    voxels, actual_percent = generate_lattice_by_percentage(
        equation_type=chosen_lattice_type,
        target_percent=desired_fill_percent
    )
    print(f"[GEOMETRY] Target matched! Final Lattice Fill: {actual_percent:.2f}%")
    
    simulated_ai_score = 1.0 # (placeholder)
    
    print(f"[PHYSICS] Evaluating properties using profile: {custom_material['name']}...")
    stats = evaluate_physical_stats(voxels, custom_material, ai_efficiency_score=simulated_ai_score)
    
    print("\n--------------------------------------------------")
    print(f"hypothetical calculated vals for {chosen_lattice_type.upper()}")
    print("--------------------------------------------------")
    print(f"Material Profile       : {custom_material['name']}")
    print(f"True Relative Density  : {stats['Relative Density']:.4f}")
    print(f"AI Efficiency Factor   : {stats['AI Efficiency Factor']:.2f}")
    print(f"Effective Stiffness    : {stats['Effective Stiffness (MPa)']:.2f} MPa")
    print(f"Peak Force Capacity    : {stats['Peak Hypothetical Force Capacity (Newtons)']:.2f} Newtons")
    print(f"Peak Torque Capacity   : {stats['Peak Hypothetical Torque Capacity (Newton-Meters)']:.4f} N-m")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()