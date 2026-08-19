# find_kmin Algorithm Documentation

## Overview

`find_kmin` is a Monte Carlo algorithm designed to compute the **minimum average squared displacement** (`displ_k_min`) when sampling configurations within a basin of attraction at spring constant `k=0`. This value represents the characteristic displacement scale at which particles in a jammed packing can freely diffuse within their basin of attraction without escaping to neighboring minima.

## Physical Context

In the context of basin volume calculations for jammed packings:
- Each jammed packing corresponds to a **local energy minimum** (basin)
- The **basin volume** is the hypervolume of configuration space that flows to this minimum under energy minimization
- `kmax` is the spring constant at which particles are strongly confined near the minimum
- `kmin` corresponds to zero spring constant (`k=0`), where particles can explore the entire basin

The `find_kmin` algorithm measures how far particles can move from their jammed positions while remaining within the same basin of attraction.

## Algorithm Architecture

### Class Hierarchy

```
_BaseMCRunner (mcpele)
    └── BaseSpheresMCrunner (_ss_mcrunner.py)
            └── SpheresMCRunner (mcrunner.py)
                    └── BV_MCrunner (mcrunner.py)
                            └── KminMCRunner (_kmin_mcrunner.py)
```

### Core Components

1. **KminMCRunner** (`_kmin_mcrunner.py:29`): Wrapper class that configures and runs the simulation
2. **BV_MCrunner** (`mcrunner.py:509`): Basin Volume Monte Carlo runner that performs the actual sampling
3. **CheckSameMinimum** (`_conf_test_cpp.pyx:441`): Configuration test that verifies the particle configuration minimizes back to the original basin
4. **RecordDisp2Histogram**: Records the histogram of squared displacements

## Algorithm Steps

### 1. Initialization

```python
# Load jammed packing configuration
imp_packing = read_jammed_packing_config(configpath)

# Extract particle data
coords = imp_packing["coords"]  # Jammed configuration (origin)
hs_radii = imp_packing["hs_radii"]  # Hard sphere radii
boxv = imp_packing["boxv"]  # Box vectors
rattlers = imp_packing["rattlers"]  # Rattler identification
```

### 2. Setup Harmonic Bias Potential

For `find_kmin`, the spring constant `k=0` (no bias):

```python
potential = Harmonic(coords, k=0, bdim=bdim, com=True)
```

The `com=True` flag fixes the center of mass to prevent drift.

### 3. Monte Carlo Sampling Loop

For each of `niter` iterations:

#### Step 3.1: Propose Move
Using `RandomCoordsDisplacement` takestep:
- Either move a single particle (`single=True`) or all particles
- Displacement drawn uniformly within current stepsize
- Stepsize is adaptively adjusted to maintain target acceptance ratio

#### Step 3.2: Energy Evaluation
Calculate the total energy:
```
E_total = E_bias(r) + E_HS_WCA(r)
```
where:
- `E_bias = k * |r - r_origin|^2` (zero for k=0)
- `E_HS_WCA` is the hard-sphere with WCA soft shell potential

#### Step 3.3: Metropolis Accept/Reject
Accept with probability:
```
P_accept = min(1, exp(-β * ΔE))
```
where β = 1/T (temperature typically = 1.0)

#### Step 3.4: Hard-Sphere Overlap Test
`CheckOverlapPeriodic[CellLists]` verifies no hard-sphere overlaps exist.

#### Step 3.5: Same Basin Test (Critical)
`CheckSameMinimum` performs:
1. Energy minimization of current configuration using FIRE/LBFGS optimizer
2. Compute RMS distance to origin: `d_rms = sqrt(||r_min - r_origin||^2 / N_dof)`
3. **Accept only if** `d_rms < dtol` (typically ~1e-3)

This ensures all accepted configurations remain within the original basin of attraction.

#### Step 3.6: Record Statistics
`RecordDisp2Histogram` accumulates:
- Squared displacement: `d^2 = Σ_i |r_i - r_origin,i|^2` (excluding rattlers, fixing COM)
- Histogram of d^2 values
- Running mean and variance

### 4. Output Calculation

```python
# Get mean and variance from histogram
displ_k_min, var_displ_k_min = histogram.get_mean_variance()

# Apply correction factor (1.25x) for finite sampling
displ_k_min *= 1.25
```

The 1.25 correction factor accounts for limited computation time/finite sampling effects.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 0 | Spring constant (zero for kmin) |
| `niter` | 1e5 | Number of MC iterations |
| `stepsize` | 0.1 | Initial displacement step size |
| `acceptance` | 0.2 | Target acceptance ratio |
| `adjustf` | 0.9 | Stepsize adjustment factor |
| `adjustf_niter` | 1e4 | Steps for stepsize equilibration |
| `dtol` | ~1e-3 | Tolerance for same-basin test |
| `opt_tol` | ~1e-5 | Optimizer convergence tolerance |
| `hmin`, `hmax` | 0, 1000 | Histogram bounds |
| `hbinsize` | 1 | Histogram bin size |

## Configuration Tests

### 1. CheckOverlapPeriodic (Early Test)
Rejects configurations with hard-sphere overlaps using:
- O(N^2) naive check, or
- O(N) cell lists for large systems

### 2. CheckSameMinimum (Late Test)
The critical basin verification:

```
For each proposed configuration r:
  1. r_min = Minimize(r) using FIRE/LBFGS/CVODE
  2. Compute d = ||r_min - origin|| / sqrt(N_dof)
  3. If d < dtol: ACCEPT (same basin)
     Else: REJECT (escaped to different minimum)
```

## Data Flow

```
Input: jammed_packing0.xydr (jammed configuration)
   │
   ├── Read configuration, radii, box vectors
   ├── Read kmax config (from findk step) for dtol, minimizer settings
   │
   ▼
Monte Carlo Sampling (k=0)
   │
   ├── Propose random displacement
   ├── Check overlap (hard sphere)
   ├── Metropolis accept/reject
   ├── Minimize and check same basin
   ├── Record d^2 if accepted
   │
   ▼
Output: kmin_jammed_packing0.config
   │
   ├── displ_k_min: Mean squared displacement
   ├── var_displ_k_min: Variance
   ├── mean_coord_dist: Mean coordinate distance
   ├── pca_asphericity: Shape factor from trajectory PCA
```

## Trajectory Analysis

The algorithm also records trajectory data for additional analysis:

1. **PCA of trajectory**: Analyzes the principal components of particle motion
2. **Asphericity factor**: Measures how non-spherical the sampling distribution is
3. **Diffusion time series**: Optional recording for diffusion studies

## Usage Example

```bash
python bv_find_kmin.py jammed_packing0.xydr \
    --niter 1e5 \
    --stepsize 0.1 \
    --acceptance 0.2 \
    --hmax 1000 \
    --hbinsize 1
```

## Relationship to Basin Volume

### In the MBAR Approach (Current Method)

The `mbar_compute_volume.py` is the primary volume computation method. In this approach:

1. **kmin config is a prerequisite check only**: The file `kmin_*.config` is verified to exist (line 201-202 of `mbar_compute_volume.py`) but `displ_k_min` is **not actually read or used**

2. **k=0 data comes from PT**: The actual k=0 displacement data comes from the parallel tempering (PT) timeseries at the k=0 replica, not from the separate `find_kmin` run

3. **MBAR reweighting**: MBAR combines data from all replicas (including k=0) to compute free energy differences directly, without needing the variable transformation that used `displ_k_min`

The volume is computed via:
```python
# From mbar_compute_volume.py:_mbar_compute_volume()
F0 = Fmin - Deltaf_ij[1,0] - log(vcavity)
```
where `Deltaf_ij` comes from MBAR's `compute_perturbed_free_energies()`.

### In the Older Thermodynamic Integration Approach (Obsolete)

The older `_collect_u2_vs_k.py` (marked obsolete) used `displ_k_min` for a **variable transformation** to improve numerical integration accuracy:

```python
# Variable transform parameter
κ = N * d / displ_k_min

# Transformed spring constants for Gauss-Lobatto quadrature
k(t) = -κ + κ * (1 + kmax/κ)^((1+t)/2)
```

This transformation "flattens" the integrand `<d²(k)>` which varies dramatically between k=0 and kmax, enabling accurate Gauss-Lobatto integration of:

```
ln(V_basin) ∝ ∫_0^kmax <d²(k)> dk
```

### Why find_kmin Still Runs

Even though MBAR doesn't use `displ_k_min` directly:
1. **Pipeline validation**: Ensures the k=0 sampling converged properly before PT
2. **Sanity check**: The `displ_k_min` value can be compared against PT k=0 data
3. **Legacy compatibility**: Some older analysis scripts may still use it

## Implementation Notes

1. **Rattler exclusion**: Rattler particles (not part of the jammed network) are excluded from displacement calculations
2. **Center of mass fixing**: COM is fixed to prevent translational drift
3. **Cell lists**: Used for efficient overlap detection in large systems
4. **Adaptive stepsize**: Maintains ~20% acceptance ratio for efficient sampling
5. **Equilibration**: Initial `adjustf_niter` steps are for stepsize adaptation and excluded from statistics

## Source Files

| File | Description |
|------|-------------|
| `bv_find_kmin.py` | Entry point script with CLI arguments |
| `_kmin_mcrunner.py` | KminMCRunner configuration class |
| `mcrunner.py` | BV_MCrunner Monte Carlo implementation |
| `_ss_mcrunner.py` | BaseSpheresMCrunner base class |
| `_config_mcrunner.py` | ConfigMCRunner abstract base |
| `monte_carlo/_conf_test_cpp.pyx` | CheckSameMinimum Cython wrapper |
| `monte_carlo/_action_cpp.pyx` | RecordDisp2Histogram Cython wrapper |
