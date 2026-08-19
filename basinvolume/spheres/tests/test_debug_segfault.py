#!/usr/bin/env python
"""Debug script to isolate the segfault issue"""

import numpy as np
import sys

print("Python version:", sys.version)
print("NumPy version:", np.__version__)

# Try importing the module
try:
    from basinvolume.monte_carlo import CheckOverlapPeriodicCellLists
    print("Successfully imported CheckOverlapPeriodicCellLists")
except Exception as e:
    print("Failed to import:", e)
    sys.exit(1)

# Test with simple parameters
radii = np.array([1.0, 1.0], dtype=np.float64)
boxvec = np.array([5.0, 5.0, 5.0], dtype=np.float64)

print("Creating CheckOverlapPeriodicCellLists...")
print("radii:", radii)
print("boxvec:", boxvec)

try:
    checker = CheckOverlapPeriodicCellLists(radii, boxvec)
    print("Successfully created CheckOverlapPeriodicCellLists")
except Exception as e:
    print("Failed to create CheckOverlapPeriodicCellLists:", e)
    import traceback
    traceback.print_exc()