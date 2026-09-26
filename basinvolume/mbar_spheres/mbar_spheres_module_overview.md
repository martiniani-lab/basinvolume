# MBAR Spheres Module Overview

## Introduction

The `mbar_spheres` module provides tools for computing basin volumes of jammed sphere packings using the **Multistate Bennett Acceptance Ratio (MBAR)** method. This is the primary post-processing module that analyzes parallel tempering simulation data to extract basin volumes.

### What is a Basin Volume?

For a jammed packing of hard spheres, the **basin of attraction** is the region of configuration space from which energy minimization returns to the same local minimum. The **basin volume** quantifies the size of this region:

```
V_basin = ∫ dr (1 if minimize(r) = origin, 0 otherwise)
```

The basin volume is related to the configurational entropy and plays a crucial role in understanding the thermodynamics of disordered solids.

---

## Module Architecture

### File Structure

```
mbar_spheres/
├── __init__.py                          # Module exports
├── mbar_compute_volume.py               # Main MBAR volume computation
├── fastmbar_compute_volume.py           # GPU-accelerated FastMBAR version
├── wham_compute_volume.py               # Legacy WHAM implementation
├── innersphere_mcrunner.py              # Inner sphere MC runner class
├── _config_innersphere_mcrunner.py      # Inner sphere configuration wrapper
├── bv_innersphere_dos.py                # Inner sphere entry point script
└── Documentation:
    ├── mbar_spheres_module_overview.md      # This file
    ├── mbar_compute_volume_documentation.md # Detailed MBAR docs
    └── innersphere_mcrunner_documentation.md # Inner sphere docs
```

### Class Hierarchy

```
Post-Processing Classes:
├── mbar_compute_dos       # Main MBAR implementation (mbar_compute_volume.py)
├── fastmbar_compute_dos   # GPU FastMBAR version (fastmbar_compute_volume.py)
└── wham_compute_dos       # Legacy WHAM version (wham_compute_volume.py)

Inner Sphere Classes:
├── ConfigMCRunner (from spheres module)
│   └── ConfigInnerSphereMCRunner (_config_innersphere_mcrunner.py)
│
└── SpheresMCRunner (from spheres module)
    └── BVInnerSphereMCrunner (innersphere_mcrunner.py)
```

---

## Volume Computation Pipeline

### Complete Workflow

The mbar_spheres module is the final step in the basin volume computation pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BASIN VOLUME PIPELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Generate Jammed Packing                                          │
│     └── Create equilibrated packing, quench to T=0                   │
│                                                                      │
│  2. Find kmax (bv_find_kmax.py)                                      │
│     └── Determine maximum spring constant for 90% acceptance         │
│                                                                      │
│  3. Find kmin (bv_find_kmin.py)                                      │
│     └── Sample basin at k=0, measure displacement scale              │
│                                                                      │
│  4. Parallel Tempering (bv_parallel_tempering.py)                    │
│     └── Run replica exchange at multiple k values                    │
│                                                                      │
│  5. Inner Sphere Sampling (bv_innersphere_dos.py)        ◄─┐         │
│     └── Sample innermost region for reference volume       │         │
│                                                             │         │
│  6. MBAR Volume Computation (mbar_compute_volume.py)     ◄─┘         │
│     └── Combine all data to compute basin volume          mbar_spheres│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
INPUT FILES:
├── jammed_packings/jammed_packing0.config    # Packing parameters
├── explore_bv_jammed_packing0/
│   ├── findk_jammed_packing0.config          # kmax, prob, dtol
│   ├── kmin_jammed_packing0.config           # Verification file
│   ├── explore_jammed_packing0.config        # PT parameters
│   ├── biases                                # k values for each replica
│   ├── 0/TimeSeries.*                        # PT time series, replica 0
│   ├── 1/TimeSeries.*                        # PT time series, replica 1
│   ├── ...
│   ├── inner_sphere.timeseries               # Inner sphere samples
│   └── innersphere_jammed_packing0.config    # Inner sphere parameters

OUTPUT FILES:
└── explore_bv_jammed_packing0/analysis/
    ├── mbar_volume_data                      # Main volume output
    ├── dos.pdf                               # Density of states plot
    ├── log_dos.pdf                           # Log DOS plot
    ├── histograms.pdf                        # Per-replica histograms
    ├── time_series.pdf                       # Time series plot
    └── *.csv                                 # Data files
```

---

## Key Quantities Computed

### Primary Output

| Quantity | Symbol | Description |
|----------|--------|-------------|
| **Basin Volume (log)** | `F0` | `-ln(V_basin / V_cavity)` |
| **Standard Error** | `sigF0` | Error estimate on F0 |
| **Unit Box Volume** | `unit_box_F0` | Volume in unit box coordinates |
| **Uncorrected Volume** | `F0unc` | `-ln(V_basin)` without cavity correction |

### Intermediate Quantities

| Quantity | Symbol | Description |
|----------|--------|-------------|
| **Free Energies** | `f_k` | MBAR free energy at each k |
| **Weights** | `w_i` | Optimal sample weights |
| **Reference Volume** | `V_min` | Inner sphere reference volume |
| **Acceptance** | `acc_frac` | Inner sphere acceptance fraction |

### Diagnostic Outputs

| Quantity | Description |
|----------|-------------|
| **Density of States** | `g(r)` - probability distribution of displacements |
| **Shape Factor** | `g(r)/r^(N-1)` - deviation from spherical |
| **Histograms** | Per-replica displacement distributions |
| **Time Series** | Raw displacement vs MC step |

---

## Mathematical Framework

### MBAR Equations

MBAR solves the self-consistent equations:

```
exp(-f_i) = Σ_n W_n exp(-u_i(x_n))

W_n = 1 / Σ_k N_k exp(f_k - u_k(x_n))
```

where:
- `f_i` = free energy at state i
- `u_i(x)` = reduced potential at state i
- `N_k` = number of samples from state k
- `W_n` = weight for sample n

### Reduced Potentials

For harmonic bias:

```
u_k(r) = (k/2) * r²
```

For inner sphere (includes Jacobian):

```
u_innersphere(r) = (N-1) * ln(r) + (k/2) * r²
```

### Volume Formula

```
F0 = Fmin - ΔF(inner→k=0) - ln(V_cavity)

where:
Fmin = -ln(V_ball(rmin)) - ln(acceptance)
ΔF = Free energy difference from MBAR perturbed states
V_cavity = Product of single-particle accessible volumes
```

---

## Implementation Variants

### 1. Standard MBAR (`mbar_compute_dos`)

- Uses pymbar library
- CPU-based computation
- Supports subsampling for correlated data
- Full error estimation via MBAR covariance

```python
from basinvolume.mbar_spheres import mbar_compute_dos

sim = mbar_compute_dos(nbins=1000, kde=True)
sim(fname="jammed_packing0")
```

### 2. FastMBAR (`fastmbar_compute_dos`)

- Uses FastMBAR library
- GPU-accelerated (CUDA optional)
- Faster for large datasets
- Bootstrap error estimation

```python
from basinvolume.mbar_spheres.fastmbar_compute_volume import fastmbar_compute_dos

sim = fastmbar_compute_dos(cuda=True, bootstrap=True)
sim(fname="jammed_packing0")
```

### 3. WHAM (`wham_compute_dos`)

- Legacy implementation using WHAM
- Minimizes chi-squared functional
- For testing/comparison only
- No rigorous error estimates

```python
from basinvolume.mbar_spheres.wham_compute_volume import wham_compute_dos

sim = wham_compute_dos()
sim(fname="jammed_packing0", nbins=300)
```

---

## Bias Potential Types

### Harmonic Bias (Default)

```
U_bias(r) = (k/2) * r²
```

Standard umbrella sampling with spring constant k.

### Radial Gaussian Bias

```
U_bias(r) = (k/2) * (r - l0)² + correction(r > r_cutoff)
```

Gaussian centered at `l0` with soft boundary at `r_cutoff`. Used for more uniform coverage of large basins.

---

## Error Estimation

### Statistical Uncertainty

MBAR provides error estimates via the covariance matrix:

```python
result_dict = mbar.compute_free_energy_differences(return_theta=True)
dDeltaf_ij = result_dict["dDelta_f"]  # Standard errors
Theta_ij = result_dict["Theta"]        # Covariance matrix
```

### Bootstrap (Optional)

For more robust error estimates:

```python
sim = mbar_compute_dos(bootstrap=True)
```

This resamples the time series and recomputes MBAR multiple times.

---

## Performance Considerations

### Memory Usage

- Time series can be large: `nreplicas × nsamples × 8 bytes`
- Reduce with subsampling or max_series_size parameter
- Parallel loading uses multiple cores

### Computation Time

| Component | Typical Time |
|-----------|--------------|
| Load time series | 10-60 seconds |
| Equilibration detection | 5-30 seconds |
| Build u_kn matrix | 1-10 seconds |
| MBAR optimization | 30-300 seconds |
| Histogram/DOS | 10-60 seconds |

### Parallelization

```python
sim = mbar_compute_dos(ncores=7)  # Use 7 cores for parallel operations
```

Parallelization applies to:
- Time series loading
- Equilibration detection
- Histogram construction

---

## Validation and Diagnostics

### Sanity Checks

The module includes `VolumeSanityCheck` which computes:

```python
volume_sanity_check = VolumeSanityCheck(packing_configpath)
F0_acc = volume_sanity_check.F0_acc  # Hard sphere fluid reference
```

This provides a reference volume for comparison.

### Key Diagnostics

1. **Time Series Stability**: Check for equilibration artifacts
2. **Histogram Overlap**: Adjacent replicas should overlap significantly
3. **DOS Continuity**: `g(r)` should be smooth across r
4. **Shape Factor**: `g(r)/r^(N-1)` reveals basin geometry

---

## Common Issues and Solutions

### Issue: MBAR fails to converge

**Cause**: Poor overlap between adjacent replicas or extreme k values.

**Solution**:
- Check histogram overlap in `histograms.pdf`
- Increase number of PT replicas
- Adjust k spacing to improve overlap

### Issue: Large error bars

**Cause**: Insufficient sampling or high correlation.

**Solution**:
- Run longer PT simulations
- Increase inner sphere iterations
- Check statistical inefficiency values

### Issue: Missing config files

**Cause**: Pipeline not completed or wrong directory.

**Solution**:
- Verify all prerequisite steps completed
- Check file paths in error messages
- Use absolute paths for packings_dir and explore_dir

---

## References

1. **MBAR**: Shirts & Chodera, J. Chem. Phys. 129, 124105 (2008)
2. **Basin Volume**: Martiniani et al., Phys. Rev. E 93, 012906 (2016)
3. **pymbar**: https://github.com/choderalab/pymbar
4. **FastMBAR**: https://github.com/BrooksResearchGroup-UM/FastMBAR

---

## Quick Start

```bash
# After completing PT simulation, run inner sphere
python bv_innersphere_dos.py jammed_packing0.xydr \
    -p jammed_packings \
    --explore-dir explore_bv_jammed_packing \
    --niter 1e5

# Then compute volume with MBAR
python mbar_compute_volume.py \
    -f jammed_packing0 \
    -d explore_bv_ \
    --kde

# Results in:
# explore_bv_jammed_packing0/analysis/mbar_volume_data
```

---

## See Also

- [mbar_compute_volume_documentation.md](mbar_compute_volume_documentation.md) - Detailed MBAR documentation
- [innersphere_mcrunner_documentation.md](innersphere_mcrunner_documentation.md) - Inner sphere sampling details
- [find_kmax_documentation.md](../find_kmax_documentation.md) - Finding maximum spring constant
- [find_kmin_documentation.md](../find_kmin_documentation.md) - Finding minimum displacement scale
- [PT_MASTER_WORKER.md](../PT_MASTER_WORKER.md) - Parallel tempering implementation
