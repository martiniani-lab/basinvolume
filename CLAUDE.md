# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BasinVolume is a scientific computing library implementing the mean basin volume method for computing thermodynamic properties of complex systems, particularly hard sphere packings and jammed materials. The project uses a hybrid Python/C++ architecture with Cython bridging for performance-critical computations.

## Architecture

### Core Components
- `basinvolume/monte_carlo/` - Monte Carlo sampling algorithms
- `basinvolume/spheres/` - Hard sphere specific implementations  
- `basinvolume/utils/` - Utility functions and statistical analysis
- `basinvolume/post_processing/` - Analysis and visualization tools
- `source/` - High-performance C++ backend implementations

### Cython Integration
Six main Cython modules bridge Python and C++ (`.pxd` files are automatically imported to corresponding `.pyx` files):
- `_action_cpp` - Monte Carlo action handling and data recording
- `_takestep_cpp` - Step generation algorithms
- `_independence_sampling` - Replica exchange and advanced sampling
- `_conf_test_cpp` - Configuration testing and overlap detection
- `_utils_cpp` - Core utilities and statistical analysis
- `_cross_validation_cost_cpp` - Statistical validation

### External Dependencies
- **PELE** - Energy landscape exploration library (../pele)
- **MCPELE** - Monte Carlo extensions for PELE
- **PyCG_DESCENT** - Conjugate gradient optimization
- **SUNDIALS** - Differential equation solvers
- **Eigen** - C++ linear algebra templates

## Build System

The project uses a complex multi-tool build system:

### Development Commands
```bash
# Build with parallel compilation
python setup_with_cmake.py build_ext --inplace -j4

# Rebuild only Cython modules
python cythonize.py basinvolume

# Clean build
rm -rf build/ && python setup_with_cmake.py build_ext --inplace

# Alternative with specific compiler
python setup_with_cmake.py build_ext --inplace -c gcc -j8
```

### Build Process
1. `cythonize.py` compiles `.pyx` → `.cxx` 
2. `setup_with_cmake.py` generates CMakeLists.txt from template
3. CMake configures C++ build with dependency detection
4. Make builds C++ libraries with parallel compilation
5. Python setup copies libraries to package

## Testing

### C++ Tests
```bash
cd cpp_tests && mkdir build && cd build
cmake ../source && make && ctest
```

### Python Tests
```bash
python -m pytest basinvolume/*/tests/
```

## Code Quality

- **Formatting**: Black with 99 character line length
- **Import sorting**: isort with black profile
- **C++ Standard**: C++20 with extensive optimizations
- **Memory management**: Smart pointers and RAII patterns

## Template Architecture

The codebase uses template specialization for different dimensions with a Cython workaround:
```cython
# Integer template parameters via fake types
ctypedef int INT1 "1"  # becomes template<1>
ctypedef int INT2 "2"  # becomes template<2>
ctypedef int INT3 "3"  # becomes template<3>
```

This enables dimension-specific optimizations for 1D, 2D, and 3D systems.

## Key Algorithms

### Monte Carlo Sampling
- Independence sampling for replica exchange
- Parallel tempering with MPI support
- Checkpointing for long-running simulations

### Hard Sphere Systems
- Overlap detection with periodic, Cartesian, and Lees-Edwards boundary conditions
- Cell list optimization for O(N) neighbor finding
- Polydisperse system support

### Basin Volume Calculations
- Minimum identification through quenching
- Basin boundary detection via configuration tests
- Volume estimation through Monte Carlo integration

## Performance Optimizations

- **Cell Lists**: Efficient neighbor finding algorithms
- **Template Specialization**: Compile-time dimension optimization
- **Array Wrapping**: Zero-copy NumPy ↔ C++ array conversion
- **Frozen Particles**: Partial system freezing for computational efficiency

## Scientific Workflow

```bash
# Generate hard sphere packings
python -m basinvolume.spheres.generate_jammed_packing

# Run basin volume calculations
python -m basinvolume.spheres.bv_parallel_tempering

# Analyze results
python -m basinvolume.post_processing.compute_volumes
```