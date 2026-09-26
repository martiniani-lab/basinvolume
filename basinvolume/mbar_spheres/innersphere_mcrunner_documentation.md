# Inner Sphere MC Runner Documentation

## Overview

The inner sphere Monte Carlo runner samples the **innermost region** of a basin of attraction, providing a reference volume for MBAR basin volume calculations. This sampling is essential for anchoring the free energy scale.

### Purpose

The inner sphere sampling serves two critical functions:

1. **Reference Volume**: Provides a known volume (small n-ball) for calibrating the MBAR free energy scale
2. **High-k Bridge**: Connects the parallel tempering data (which starts at some finite kmax) to a well-defined geometric region

### Why It's Needed

Parallel tempering simulations sample configurations at various spring constants, but:
- They don't directly sample the innermost region (very small r)
- MBAR needs a reference point with known volume to compute absolute volumes

The inner sphere provides this reference by:
1. Sampling uniformly within a small ball of radius `rmin`
2. Measuring the acceptance fraction (fraction staying in basin)
3. Using `V_reference = V_ball * acceptance` as the calibration point

---

## Physical Background

### Sampling Region

The inner sphere samples configurations within a small radius `rmin` around the origin (jammed packing position):

```
r = |x - x_origin| < rmin
```

where `rmin` is chosen such that most configurations minimize back to the same basin.

### Effective Volume

The effective volume sampled is:

```
V_effective = V_nball(rmin, ndof) * acceptance_fraction
```

where:
- `V_nball(rmin, ndof)` is the volume of an n-dimensional ball with radius `rmin`
- `acceptance_fraction` is the probability that a configuration stays in the basin

### Connection to Spring Constant

The inner sphere stepsize relates to a spring constant via:

```
k = 1 / stepsize^2
```

This k value should be high enough that the acceptance is significant (typically > 50%).

---

## Algorithm Description

### Two Sampling Modes

The `ConfigInnerSphereMCRunner` runs **two** MC runners sequentially:

1. **Gaussian Step** (`mcrunner_gaussian`):
   - Samples from a Gaussian distribution centered at origin
   - Stepsize = `sqrt(1/k)` where k is inferred from PT data
   - Records displacement time series for MBAR

2. **Ball Pick** (`mcrunner_ballpick`):
   - Uniform sampling within a ball of radius `ref_radius = stepsize/2`
   - Used to measure acceptance fraction for volume calibration
   - Smaller radius = higher acceptance

### Workflow

```
1. INITIALIZATION
   ├── Read packing configuration
   ├── Import findk config (kmax, dtol, etc.)
   ├── Read mean displacement from k=0 replica (u2_k0)
   ├── Compute k = 1/u2_k0, stepsize = sqrt(1/k)
   └── Set ref_radius = stepsize/2

2. CREATE MC RUNNERS
   ├── mcrunner_gaussian: Gaussian step with stepsize
   └── mcrunner_ballpick: Uniform ball with ref_radius

3. RUN SIMULATIONS
   ├── Run mcrunner_gaussian → generates time series
   └── Run mcrunner_ballpick → measures acceptance

4. OUTPUT
   ├── Write config file with MC status
   └── Save time series to inner_sphere.timeseries
```

---

## Key Classes

### BVInnerSphereMCrunner

Located in `innersphere_mcrunner.py:55`, this is the core MC runner.

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `potential` | Potential | Should be NullPotential (no actual potential) |
| `full_coords` | array | Initial/origin coordinates |
| `temperature` | float | Temperature (typically 1.0) |
| `stepsize` | float | Sampling radius or Gaussian sigma |
| `niter` | int | Number of MC iterations |
| `origin` | array | Reference configuration (jammed packing) |
| `hs_radii` | array | Hard sphere radii |
| `boxv` | array | Box vectors |
| `sca` | float | WCA shell thickness factor |
| `gaussian_step` | bool | True for Gaussian, False for uniform ball |

#### Key Methods

```python
# Get current spring constant
def get_k(self):
    stepsize = self.get_stepsize()
    return 1.0 / (stepsize * stepsize)

# Save time series to file
def dump_timeseries(self, fname, clear=True):
    timeseries = np.array(self.time_series.get_time_series())
    np.savetxt(fname, timeseries)
    return timeseries

# Check convergence
def check_convergence(self, nr_steps_to_check=10000, rel_std_threshold=0.05):
    return self.time_series.check_convergence(...)
```

### ConfigInnerSphereMCRunner

Located in `_config_innersphere_mcrunner.py:28`, this is the configuration wrapper.

#### Key Attributes Set from Config Files

| Attribute | Source | Description |
|-----------|--------|-------------|
| `kmax` | findk_*.config | Maximum spring constant from find_kmax |
| `dtol` | findk_*.config | Distance tolerance for basin test |
| `opt_tol` | findk_*.config | Optimizer convergence tolerance |
| `u2_k0` | hist_mean file | Mean squared displacement at k=0 |
| `k` | Computed | `1.0 / u2_k0` |
| `stepsize` | Computed | `1.0 / sqrt(k)` |
| `ref_radius` | Computed | `stepsize / 2` |

---

## Sampling Methods

### Gaussian Sampling (`SampleUniformSphereGaussian`)

Generates samples from:

```
x_new = origin + sigma * randn(ndim)
```

where `sigma = stepsize`. This produces a Gaussian distribution in r:

```
P(r) ~ r^(N-1) * exp(-r^2 / (2*sigma^2))
```

### Uniform Ball Sampling (`UniformSphericalSampling`)

Generates samples uniformly within a ball of radius R:

```
x_new = origin + R * (random_direction) * (random_radius)^(1/N)
```

This produces a uniform distribution in configuration space within the ball.

---

## Time Series Recording

### RecordDisplacementTimeseries

The MC runner records the radial displacement at each step:

```python
self.time_series = RecordDisplacementTimeseries(
    self.red_origin,  # Reference coordinates (rattlers removed)
    self.bdim,        # Box dimension
    self.ts_niter,    # Number of iterations to record
    self.ts_freq,     # Recording frequency
    fix_com=True      # Fix center of mass
)
```

The recorded quantity is:

```
r = sqrt(sum((x - origin)^2))  # Radial displacement
```

with center of mass subtracted if `fix_com=True`.

---

## Configuration Tests

### No Overlap Test

Hard sphere overlaps are detected using:
- `CheckOverlapPeriodic` for naive O(N^2) check
- `CheckOverlapPeriodicCellLists` for O(N) cell list check

### Same Basin Test

Inherited from parent `SpheresMCRunner`, verifies configurations minimize to same basin using `CheckSameMinimum`.

---

## Output Files

### innersphere_*.config

Contains simulation parameters and results:

```ini
#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND
[INNERSPHERE_IMPORTED_JAMMED_PACKING]
nparticles: 64
packing_fraction: 0.64
boxdim: 3
ndim: 192
boxv: 4.0 4.0 4.0
sca: 0.05

[INNERSPHERE_MCRUNNER]
k: 1234.5678
temperature: 1.0
niter: 100000
stepsize: 0.0284

[INNERSPHERE_GAUSSIAN_MCRUNNER_STATUS]
niter: 100000
acc_frac: 0.85
stepsize: 0.0284

[INNERSPHERE_BALLPICK_MCRUNNER_STATUS]
niter: 100000
acc_frac: 0.72
stepsize: 0.0142  # This is ref_radius, used for volume calculation
```

### inner_sphere.timeseries

Raw time series of radial displacements from Gaussian sampling:

```
0.0123
0.0145
0.0098
...
```

One value per line, used by MBAR for reweighting.

---

## Usage

### Command Line

```bash
python bv_innersphere_dos.py jammed_packing0.xydr \
    -p jammed_packings \
    --explore-dir explore_bv_jammed_packing \
    --niter 1e5
```

### Within Pipeline

The inner sphere run is typically executed after parallel tempering completes:

```
1. bv_find_kmax.py      → findk_*.config
2. bv_find_kmin.py      → kmin_*.config
3. bv_parallel_tempering.py → PT time series
4. bv_innersphere_dos.py    → inner_sphere.timeseries  ← THIS STEP
5. mbar_compute_volume.py   → volume computation
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `niter` | 1e5 | Number of MC iterations |
| `eps` | 1.0 | WCA potential epsilon |
| `opt_nsteps` | 1e5 | Maximum optimizer steps |
| `use_cell_lists` | True | Use cell lists for overlap detection |

---

## Relationship to Volume Computation

### In MBAR

The inner sphere provides the reference point for volume calculation:

```python
# From mbar_compute_volume.py
rmin = ref_radii[0]  # Ball-pick stepsize
ref_acceptance = ref_acceptances[0]  # Ball-pick acceptance

# Reference free energy
logvmin = log_volume_nball(rmin, ndof)
Fmin = -logvmin - log(ref_acceptance)

# Final volume
F0 = Fmin - Deltaf_ij[1,0] - log(vcavity)
```

The inner sphere time series is also included in MBAR as an additional state with:

```python
u_kn[0] = (ndof - 1) * log(r) + 0.5 * k_innersphere * r^2
```

This Jacobian term `(ndof-1)*log(r)` accounts for the phase space volume element.

### Multiple Nested Spheres

The module supports multiple nested inner spheres for better coverage:

```python
innersphere_dir_list = glob.glob(explore_dir + "innersphere_*")
for dir in innersphere_dir_list:
    # Load config and timeseries from each
```

---

## Implementation Details

### Stepsize from PT Data

The inner sphere stepsize is determined from the parallel tempering k=0 replica:

```python
# Read mean displacement from replica 0
path = os.path.join(base_directory, "0", "hist_mean")
niter, u2, var, std_err = lineList[-1].split()
u2_k0 = float(u2)

# Compute stepsize
k = 1.0 / u2_k0
stepsize = 1.0 / np.sqrt(k)
ref_radius = stepsize / 2
```

### Center of Mass Fixing

For physical systems, the center of mass is fixed:

```python
self.fix_com = interaction is not Interaction.NEGATIVE_COS

if self.fix_com:
    coords = _subtract_com(coords, ndim=bdim)
```

This reduces the degrees of freedom by `bdim`:

```
ndof = (nparticles - 1) * bdim
```

---

## Source Files

| File | Description |
|------|-------------|
| `innersphere_mcrunner.py` | BVInnerSphereMCrunner class |
| `_config_innersphere_mcrunner.py` | ConfigInnerSphereMCRunner wrapper |
| `bv_innersphere_dos.py` | CLI entry point |
| `__init__.py` | Module exports |
