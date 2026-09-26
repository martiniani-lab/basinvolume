# MBAR Volume Computation Documentation

## Overview

The `mbar_compute_volume.py` module implements basin volume computation using the **Multistate Bennett Acceptance Ratio (MBAR)** method. This is the primary and recommended approach for computing basin volumes from parallel tempering simulation data.

### Purpose

Given time series data from parallel tempering (PT) simulations at different spring constants `k`, MBAR optimally combines all the data to compute:
1. **Free energy differences** between replicas
2. **Basin volume** (expressed as negative log volume, `F0`)
3. **Density of states** `g(r)` as a function of radial displacement

### Key Advantage over Thermodynamic Integration

MBAR directly estimates free energy differences without explicit numerical integration, making it:
- More statistically efficient (uses all data optimally)
- Less sensitive to the choice of intermediate k values
- Able to provide rigorous error estimates

---

## Physical Background

### The Basin Volume Problem

For a jammed packing of particles, the **basin volume** is the hypervolume of configuration space that flows to this minimum under energy minimization. The basin volume `V_basin` is related to the configurational entropy:

```
S = ln(V_basin)
```

### Biased Sampling Strategy

Direct sampling of the basin is intractable because:
- The basin spans a huge range of displacements (from 0 to several particle diameters)
- The probability distribution `P(r) ~ r^(N-1)` concentrates at large r

The solution uses **umbrella sampling** with harmonic bias potentials:

```
U_bias(r) = (k/2) * r^2
```

where `r = |x - x_origin|` is the radial displacement from the jammed configuration.

### MBAR Reweighting

MBAR combines samples from K different bias potentials (spring constants k_0, k_1, ..., k_{K-1}) to estimate:

```
exp(-f_i) = sum_n W_n * exp(-u_i(r_n))
```

where:
- `f_i` is the free energy at state i
- `W_n` are optimal weights computed self-consistently
- `u_i(r_n)` is the reduced potential at state i

---

## Algorithm Description

### Input Data Requirements

The algorithm requires data from a completed parallel tempering simulation:

1. **PT Time Series**: Radial displacement `r(t)` for each replica at different k values
2. **Inner Sphere Time Series**: Samples from the innermost sphere (at very high k)
3. **Configuration Files**:
   - `findk_*.config`: Contains `kmax` and probability at kmax
   - `kmin_*.config`: Verification file (existence check only)
   - `innersphere_*.config`: Inner sphere parameters (k, radius, acceptance)
   - `explore_*.config`: PT simulation parameters

### Main Workflow

```
1. IMPORT CONFIGURATION
   ├── Read jammed packing parameters (N, bdim, boxv, etc.)
   ├── Read kmax from findk config
   └── Read inner sphere parameters (k, radius, acceptance)

2. IMPORT TIME SERIES DATA
   ├── Load k values from "biases" file
   ├── Import PT time series for each replica
   ├── Detect and remove equilibration region
   └── Import inner sphere time series

3. BUILD FLAT TIMESERIES
   ├── Compute statistical inefficiency g[k] for each replica
   ├── Subsample correlated data
   └── Concatenate into single flat array

4. BUILD MBAR
   ├── Construct reduced potential matrix u_kn
   ├── Initialize MBAR with BAR initialization
   └── Iterate to convergence

5. COMPUTE VOLUME
   ├── Get free energy differences from MBAR
   ├── Compute perturbed free energies for volume
   └── Apply reference volume correction

6. (Optional) COMPUTE DENSITY OF STATES
   ├── Build histograms for each replica
   ├── Unbias histograms
   └── Combine with MBAR weights to get g(r)
```

---

## Volume Computation Formula

### The Core Calculation

The volume computation (`_mbar_compute_volume()`) follows these steps:

#### Step 1: Reference Volume

The smallest sampled region provides a reference volume:

```python
rmin = ref_radii[0]  # Ball-pick stepsize from inner sphere
logvmin = log_volume_nball(rmin, ndof)
Fmin = -logvmin - log(ref_acceptance)
```

where:
- `rmin` is the inner sphere sampling radius
- `ref_acceptance` is the acceptance fraction in the inner sphere
- `ndof = (N-1) * bdim` is the number of degrees of freedom (COM fixed)

#### Step 2: MBAR Perturbed Free Energy

To connect the inner sphere to k=0 (unbiased):

```python
# Create perturbed states
u_lk[0] = u_kn[k0_index] where r < rmin, else LARGE
u_lk[1] = u_kn[k0_index]  # Full k=0 potential

# Compute free energy difference
Deltaf_ij = mbar.compute_perturbed_free_energies(u_lk)
```

#### Step 3: Final Volume

```python
F0 = (Fmin - Deltaf_ij[1,0]) - log(vcavity)
```

where `vcavity` is the cavity volume (total accessible volume per particle).

### Physical Interpretation

```
F0 = -ln(V_basin / V_cavity)

where:
V_basin = basin volume in configuration space
V_cavity = product of single-particle accessible volumes
```

---

## Reduced Potential Matrix

### Construction (`_build_u_kn`)

For K states and N total samples, the reduced potential matrix is:

```python
u_kn = np.empty((K, N))

for i in range(K):
    if i < number_nested_spheres:
        # Inner sphere: includes Jacobian term
        u_kn[i] = (ndof - 1) * log(r) + 0.5 * k[i] * r^2
    else:
        if bias == "harmonic":
            u_kn[i] = 0.5 * k[i] * r^2
        elif bias == "radial_gaussian":
            u_kn[i] = 0.5 * k[i] * (r - l0[i])^2
            # Plus Jacobian correction for r > r_cutoff
```

### Bias Types

1. **Harmonic** (default): `U = (k/2) * r^2`
2. **Radial Gaussian**: `U = (k/2) * (r - l0)^2` with soft boundary at `r_cutoff`

---

## Equilibration Detection

### Binary Search Algorithm

The equilibration point is found using pymbar's `detect_equilibration_binary_search`:

```python
def find_eqtime(ts):
    max_eq_time = ts.size // 2
    time = detect_equilibration_binary_search(ts, bs_nodes=30)[0]
    time = min(max_eq_time, time)  # Avoid artifacts near end
    return int(time)
```

The maximum equilibration time across all replicas is used:

```python
results = Parallel(n_jobs=ncores)(
    delayed(find_eqtime)(ts) for ts in timeseries
)
eq_time = int(max(results))
timeseries = timeseries[:, eq_time:]
```

---

## Density of States Computation

### Histogram Construction

Two methods available:

1. **Simple Histogram**: `np.histogram(timeseries, bin_edges, density=True)`
2. **KDE Histogram**: Kernel density estimation with bandwidth selection

### Unbiasing

For each replica, the histogram is unbiased by the potential:

```python
# Harmonic bias
hist_unbiased = 0.5 * k * bin_edges^2

# Inner sphere (includes Jacobian)
hist_unbiased = (ndof-1) * log(bin_edges) + 0.5 * k * bin_edges^2
```

### DOS Combination

The final density of states combines all replicas using MBAR weights:

```python
log_dos = log(hist_visits) + hist_unbiased
ldos = sum(log_dos * visits * weights, axis=0) / sum(visits, axis=0)
```

---

## Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `nbins` | int | Number of histogram bins (default: 1000, rounded to power of 2 + 1) |
| `bootstrap` | bool | Run bootstrap for error estimation |
| `kde` | bool | Use kernel density estimation for histograms |
| `ncores` | int | Number of parallel cores (default: 7) |
| `bias` | str | Bias type: "harmonic" or "radial_gaussian" |
| `ignore_neg_ks` | bool | Ignore replicas with negative k values |

---

## Output Files

### Main Output: `mbar_volume_data`

```ini
#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND
[VOLUME_HS_FLUID]
F0_acc: <hard_sphere_fluid_reference_volume>
F0_ideal_gas: <ideal_gas_reference_volume>
[VOLUME_MBAR]
F0: <basin_volume_negative_log>
sigF0: <standard_error>
unit_box_F0: <volume_in_unit_box_coordinates>
```

### Diagnostic Plots (if `plot_dos_data=True`)

- `time_series.pdf`: Raw displacement time series
- `histograms.pdf`: Histogram of displacements per replica
- `raw_log_dos.pdf`: Unweighted log DOS fragments
- `dos.pdf`: Final density of states
- `log_dos.pdf`: Log DOS and ratio `g(r)/r^(N-1)`
- `ratio_g.pdf`: Shape factor `g(r)/r^(N-1)`
- `hist_var_k.pdf`: Variance vs spring constant

---

## Usage

### Command Line

```bash
python mbar_compute_volume.py \
    -f jammed_packing0 \
    -d explore_bv_ \
    -w /path/to/working/directory \
    --bias harmonic
```

### Python API

```python
from basinvolume.mbar_spheres import mbar_compute_dos

# Initialize
sim = mbar_compute_dos(
    nbins=1000,
    bootstrap=False,
    kde=True,
    plot_dos_data=True,
    ncores=7,
    bias="harmonic"
)

# Run analysis
sim(
    fname="jammed_packing0",
    explore_dir="explore_bv_jammed_packing0",
    packings_dir="packings",
    jammed_packings_dir="jammed_packings"
)

# Access results
print(f"Basin volume F0: {sim.F0}")
print(f"Error: {sim.sigF0}")
```

---

## Relationship to Other Modules

### Prerequisites

1. **find_kmax** (`bv_find_kmax.py`): Determines maximum spring constant
2. **find_kmin** (`bv_find_kmin.py`): Validates k=0 sampling (file existence check only)
3. **parallel_tempering** (`bv_parallel_tempering.py`): Generates PT time series
4. **inner_sphere** (`bv_innersphere_dos.py`): Samples innermost region

### Alternative Implementations

- **FastMBAR** (`fastmbar_compute_volume.py`): GPU-accelerated version
- **WHAM** (`wham_compute_volume.py`): Legacy implementation (testing only)

---

## Implementation Notes

### Statistical Inefficiency

Each time series is decorrelated using the FFT-based statistical inefficiency:

```python
g[k] = statistical_inefficiency_fft(timeseries[k])
indices = subsample_correlated_data(timeseries[k], g=g[k])
```

This ensures samples are approximately independent for MBAR.

### MBAR Initialization

BAR initialization provides a good starting point:

```python
mbar = MBAR(
    u_kn,
    N_k,
    maximum_iterations=10000,
    relative_tolerance=1e-7,
    initialize="BAR",
    verbose=True
)
```

### Center of Mass Handling

The center of mass is fixed throughout, reducing degrees of freedom:

```
ndof = (nparticles - 1) * bdim
```

---

## Source Files

| File | Description |
|------|-------------|
| `mbar_compute_volume.py` | Main MBAR volume computation |
| `fastmbar_compute_volume.py` | GPU-accelerated FastMBAR version |
| `wham_compute_volume.py` | Legacy WHAM implementation |
| `innersphere_mcrunner.py` | Inner sphere MC runner |
| `_config_innersphere_mcrunner.py` | Inner sphere configuration wrapper |
| `bv_innersphere_dos.py` | Inner sphere entry point script |
