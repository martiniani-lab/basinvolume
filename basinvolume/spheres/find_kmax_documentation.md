# find_kmax Algorithm Documentation

## Overview

The `find_kmax` algorithm determines the maximum harmonic spring constant (`kmax`) for which a jammed particle packing remains within its basin of attraction with a specified target probability (typically 90%). This value is critical for basin volume calculations in soft sphere systems.

### Purpose

In basin volume calculations, particles are tethered to their equilibrium positions by harmonic springs with spring constant `k`. The algorithm finds the largest `k` such that random Gaussian displacements from the origin still keep the system within the same energy basin - i.e., when minimized, the configuration returns to the original jammed packing rather than escaping to a different minimum.

### Physical Intuition

- **Small k**: Large allowed displacements, particles can easily escape to neighboring basins
- **Large k**: Small allowed displacements, particles stay confined to the original basin
- **kmax**: The threshold value where the acceptance probability equals the target (e.g., 90%)

---

## Physical Background

### Harmonic Spring Model

Each particle `i` is connected to its origin position by a harmonic spring potential:

```
U(r) = (k/2) * |r - r_0|^2
```

Where:
- `r` is the current position
- `r_0` is the origin (jammed packing configuration)
- `k` is the spring constant

### Relationship Between k and Step Size

The algorithm uses a clever trick: instead of directly adjusting `k`, it modifies the standard deviation (`stepsize`) of the Gaussian displacement distribution:

```
stepsize = sqrt(1/k)
```

This relationship arises because:
- For a harmonic potential at temperature T=1, the equilibrium distribution is Gaussian
- The variance of the Gaussian is `sigma^2 = 1/k` (in units where kT=1)
- Sampling from this Gaussian is equivalent to sampling the Boltzmann distribution

### Basin Membership Test

After each displacement, the algorithm:
1. Takes the displaced configuration
2. Performs energy minimization (quenching)
3. Compares the minimized structure to the original packing
4. If the distance `|r_min - r_0|^2 < dtol`, the configuration is **accepted** (still in basin)
5. Otherwise, it's **rejected** (escaped to a different basin)

---

## Algorithm Description

### Monte Carlo Sampling Loop

```
1. Initialize: k = k_start (default: 500)
2. While not converged:
   a. Sample N coordinates from Gaussian(origin, stepsize=sqrt(1/k))
   b. Check if sample is in original basin (via quench + distance test)
   c. Track acceptance/rejection over window of knavg steps
   d. Every knavg steps:
      - Compute acceptance fraction
      - If |acceptance - target| < ktol: CONVERGED
      - Otherwise: adjust k using adaptive formula
3. After convergence: collect avgcount samples for statistics
```

### Key Insight: Fictitious Potential

The harmonic potential used in the MC runner is **fictitious** - it's set to zero and doesn't affect the energy test. The actual "potential" effect comes from:
1. The Gaussian sampling distribution (controlled by stepsize)
2. The basin membership test (via `CheckSameMinimum`)

From `_findk_mcrunner.py:92-95`:
```python
potential = Harmonic(
    self.coords, 0, bdim=self.bdim, com=False
)  # set the potential to 0, the potential is completely fictitious here
```

---

## Adaptive k Adjustment Formula

The core of the algorithm is the adaptive adjustment of `k` based on the observed acceptance fraction. From `findk.cpp:92-125`:

### The Formula

```cpp
// Period for damping oscillations
const size_t period = 3;

// Get current k from stepsize
_k = 1 / (stepsize * stepsize);

// Check convergence
if (fabs(_target - _acceptedf) < _tol) {
    _converged = true;
    return;
}

// Adaptive adjustment
const double tmp1 = 1.0 / (iterations % period + 1);           // Damping factor
const double tmp2 = 1 + (_target - _acceptedf) / (_target + _acceptedf);  // Correction
const double tmp = (1 - tmp1) + tmp1 * tmp2;                   // Blended adjustment
_k *= tmp * tmp;                                                // Apply squared
```

### Understanding the Formula

1. **Damping Factor (`tmp1`)**:
   - Cycles through values 1.0, 0.5, 0.33... every 3 iterations
   - Prevents overshooting by reducing adjustment magnitude periodically
   - Range: `[0.33, 1.0]`

2. **Correction Factor (`tmp2`)**:
   - When `acceptance > target`: `tmp2 < 1` → decrease k (allow larger steps)
   - When `acceptance < target`: `tmp2 > 1` → increase k (force smaller steps)
   - Range approximately `[0, 2]` for typical acceptance values

3. **Blended Factor (`tmp`)**:
   - Interpolates between 1 (no change) and `tmp2` (full correction)
   - The damping factor controls how much correction to apply

4. **Squared Application**:
   - `k *= tmp^2` provides smoother convergence
   - Equivalent to adjusting stepsize by factor `tmp` (since `stepsize = 1/sqrt(k)`)

### Example Calculation

If `target = 0.9`, `acceptedf = 0.8`, and iteration 4 (so `iterations % 3 = 1`):
```
tmp1 = 1/(1+1) = 0.5
tmp2 = 1 + (0.9-0.8)/(0.9+0.8) = 1 + 0.059 = 1.059
tmp = (1-0.5) + 0.5*1.059 = 0.5 + 0.529 = 1.029
k_new = k_old * 1.029^2 = k_old * 1.059
```

So k increases by ~6% to reduce the step size and increase acceptance.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kstart` | 500 | Initial guess for k (from command line `-k`) |
| `ktarget` | 0.9 | Target acceptance probability |
| `knavg` | 1e4 | Number of MC steps per averaging window |
| `ktol` | 0.025 | Convergence tolerance: stop when `|acceptance - target| < ktol` |
| `avgcount` | 1e4 | Number of accepted samples to collect after convergence |
| `dtol` | 1e-4 | Distance tolerance for basin membership test |
| `niter` | 1e8 | Maximum number of MC iterations |

### Parameter Guidelines

- **kstart**: Should be within an order of magnitude of the true kmax. Too small → slow convergence. Too large → may never find basin escapes.
- **ktarget = 0.9**: Standard choice ensuring 90% of samples stay in basin at kmax
- **knavg**: Larger values give more accurate acceptance estimates but slower convergence
- **ktol**: Smaller values give more precise kmax but require more iterations

---

## Code Structure

### Entry Points

**CLI Script**: `bv_find_kmax.py`
- Parses command line arguments
- Creates `_findk_mcrunner` instance
- Runs the simulation

### Python Classes

**`_findk_mcrunner` class** (`_findk_mcrunner.py:20-245`)
- Wrapper class handling file I/O and configuration
- Reads jammed packing configuration
- Sets up the MC runner
- Writes results to config file

**`Findk_MCrunner` class** (`mcrunner.py:920-1096`)
- Monte Carlo runner specialized for finding kmax
- Inherits from `SpheresMCRunner`
- Key methods:
  - `_set_takestep()`: Uses `SampleGaussian` for Gaussian displacements
  - `_set_actions()`: Adds the `Findk` action
  - `get_k()`: Returns current k from stepsize

### C++ Core

**`Findk` class** (`source/basinvolume/findk.h`, `source/findk.cpp`)
- MC action that tracks acceptance and adjusts k
- Key methods:
  - `action()`: Called after each MC step, tracks acceptance, triggers adjustment
  - `adjust_k()`: Applies the adaptive formula
  - `_get_vec_distance()`: Computes displacement from origin (with COM correction)

### Class Hierarchy

```
bv_find_kmax.py (CLI)
    └── _findk_mcrunner (Python wrapper)
            └── Findk_MCrunner (MC runner)
                    ├── SampleGaussian (takestep)
                    ├── CheckOverlapPeriodic (config test)
                    ├── CheckSameMinimum (basin test)
                    └── Findk (action - C++)
                            └── adjust_k() (core algorithm)
```

---

## Output

After running, the algorithm outputs to a config file:

```ini
[FINDK]
kmax: 759.1234567890123456
prob: 0.9012345678901234
displ_k_max: 0.0123456789012345
var_displ_k_max: 0.0001234567890123
```

- **kmax**: The found maximum spring constant
- **prob**: Actual acceptance probability achieved (should be ~ktarget)
- **displ_k_max**: Mean squared displacement at kmax
- **var_displ_k_max**: Variance of squared displacement

---

## Usage Example

```bash
# Basic usage
python bv_find_kmax.py jammed_packing0.xydr

# With custom starting k
python bv_find_kmax.py jammed_packing0.xydr -k 1000

# With custom target acceptance
python bv_find_kmax.py jammed_packing0.xydr --ktarget 0.85

# With cell lists for larger systems
python bv_find_kmax.py jammed_packing0.xydr --nocell  # disable cell lists

# Verbose output
python bv_find_kmax.py jammed_packing0.xydr -v
```

---

## Relationship to Basin Volume

The found `kmax` is used in thermodynamic integration to compute basin volumes:

1. **At k=0**: System samples freely, volume = total phase space
2. **At k=kmax**: System is confined to original basin with probability ~90%
3. **Integration**: Basin volume is computed by integrating displacement statistics from k=0 to k=kmax

The basin volume calculation (performed by `bv_find_kmin.py` and `bv_parallel_tempering.py`) uses kmax as an upper bound for the spring constant range.
