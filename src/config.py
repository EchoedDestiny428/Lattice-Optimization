# config.py

# --- FIXED HARDWARE & ACCURACY CONSTANTS ---
GRID_SIZE = 64

# --- DEFAULT PHYSICAL FRAME PROFILE ---
# 10 mm default bounding box
DEFAULT_BOUNDING_BOX = {
    'x': 10.0,  # Total bounding box envelope width
    'y': 10.0,  # Total bounding box envelope depth
    'z': 10.0   # Total bounding box envelope height
}

MATERIAL_PROFILES = {
    "test_material": {
        "name": "Test Material",
        "youngs_modulus_gpa": 2.0,
        "yield_strength_mpa": 50.0
    },
}