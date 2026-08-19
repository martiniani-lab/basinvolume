# PT_Master and PT_Worker: Parallel Tempering Implementation

This document explains how `PT_Master` and `PT_Worker` work together to perform parallel tempering simulations for basin volume calculations.

## Table of Contents
1. [Overview](#overview)
2. [Entry Point](#entry-point-bv_parallel_temperingpy)
3. [ReplicaState Class](#replicastate-class)
4. [PT_Master Class](#pt_master-class)
5. [PT_Worker Class](#pt_worker-class)
6. [MPI Communication Protocol](#mpi-communication-protocol)
7. [Complete Workflow](#complete-workflow)

---

## Overview

### What is Parallel Tempering?

Parallel tempering (also called replica exchange) is a technique to enhance sampling in Monte Carlo simulations. Multiple replicas of the system run simultaneously at different "temperatures" (in this case, different bias potential strengths). Periodically, replicas attempt to exchange configurations, allowing the system to escape local energy minima.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MPI ENVIRONMENT                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐                                               │
│   │  PT_Master  │  (Rank 0)                                     │
│   │             │                                               │
│   │ - Job Queue │                                               │
│   │ - Exchanges │                                               │
│   │ - Converge  │                                               │
│   └──────┬──────┘                                               │
│          │                                                      │
│          │  MPI Send/Recv                                       │
│          │                                                      │
│   ┌──────┴──────┬──────────────┬──────────────┐                 │
│   │             │              │              │                 │
│   ▼             ▼              ▼              ▼                 │
│ ┌──────┐    ┌──────┐      ┌──────┐      ┌──────┐               │
│ │Worker│    │Worker│      │Worker│      │Worker│               │
│ │Rank 1│    │Rank 2│  ... │Rank N│  ... │Rank M│               │
│ │      │    │      │      │      │      │      │               │
│ │MC Run│    │MC Run│      │MC Run│      │MC Run│               │
│ └──────┘    └──────┘      └──────┘      └──────┘               │
│                                                                 │
│   nworkers = nprocs - 1 (master doesn't run MC)                 │
│   nreplicas can be > nworkers (job queue model)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

1. **Job Queue Model**: The master distributes replica jobs to workers dynamically. When `nreplicas > nworkers`, workers receive new jobs as they complete previous ones.

2. **Centralized Exchange**: Only the master decides which replicas exchange coordinates. Workers are idle during exchange decisions.

3. **Serialized Communication**: Replica states are serialized to numpy arrays for efficient MPI transfer.

---

## Entry Point: bv_parallel_tempering.py

The simulation is launched via MPI:

```bash
mpirun -np 8 python bv_parallel_tempering.py config.yaml
```

The entry point assigns roles based on MPI rank:

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

if rank == 0:
    # Master process
    master = PT_Master(comm, nprocs, mcrunner, ...)
    master.run()
else:
    # Worker process
    worker = PT_Worker(comm, mcrunner, ...)
    worker.run()
```

**Role Assignment:**
- **Rank 0**: Runs `PT_Master` - coordinates jobs, manages exchanges, tests convergence
- **Ranks 1 to N-1**: Run `PT_Worker` - execute Monte Carlo simulations

---

## ReplicaState Class

`ReplicaState` (defined in `_pt_master.py:56-138`) encapsulates all data needed to describe a replica's current state.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | int | Unique replica identifier (0 to nreplicas-1) |
| `dx` | float | Distance from center of mass (or origin) |
| `energy` | float | Current potential energy |
| `bias_params` | array | Spring constant(s) for bias potential |
| `stepsize` | float | Current MC step size |
| `coords` | array | Particle coordinates (natoms * ndim) |
| `counters` | array | MC statistics (iterations, accepts, rejects) |
| `swap_accepted_count` | int | Number of accepted exchanges |
| `swap_rejected_count` | int | Number of rejected exchanges |

### Serialization Format

For MPI communication, `ReplicaState` is serialized to a 1D numpy array:

```python
def serialize(self):
    """Convert state to numpy array for MPI transfer."""
    return np.concatenate([
        [self.id, self.dx, self.energy],
        self.bias_params,              # 1 or 3 values
        [self.stepsize, self.takestep_count],
        self.coords.flatten(),         # natoms * ndim values
        self.counters,                 # 3 values
        self.step_adaptation_counters  # 2 values
    ])
```

The `deserialize()` class method reconstructs the state from this array.

### Buffer Size Calculation

```python
def size(self):
    """Total size of serialized state."""
    return (5 +                          # id, dx, energy, stepsize, takestep_count
            len(self.bias_params) +      # 1 (harmonic) or 3 (radial gaussian)
            len(self.coords) +           # natoms * ndim
            len(self.counters) +         # 3
            len(self.step_adaptation_counters))  # 2
```

---

## PT_Master Class

`PT_Master` (defined in `_pt_master.py:140-1279`) is the brain of the parallel tempering simulation.

### Initialization

```python
master = PT_Master(
    comm,              # MPI communicator
    nprocs,            # Total MPI processes
    mcrunner,          # MC simulation object
    nreplicas,         # Number of PT replicas
    Ks,                # Bias parameters for each replica
    max_ptiter,        # Maximum PT iterations
    pfreq=100,         # Print/dump frequency
    skip=0,            # Skip exchanges for first N iterations
    exchange_scheme="neighbor_exchange",
    ...
)
```

### Key Attributes

```python
self.nreplicas = nreplicas           # Number of replicas
self.nworkers = nprocs - 1           # Workers (master doesn't do MC)
self.replica_states = [...]          # Array of ReplicaState objects
self.recv_buffer = np.zeros(...)     # Pre-allocated receive buffer
self.exchange_scheme = ExchangeScheme.NEIGHBOR_EXCHANGE
self.ptiter = 0                      # Current iteration
self.timeseries = np.zeros(...)      # dx values over time
self.timeseries2 = np.zeros(...)     # dx^2 values for variance
```

### Main Loop: `run()`

```python
def run(self):
    """Main execution loop."""
    while self.ptiter < self.max_ptiter:
        # Check if time to checkpoint
        if checkpoint_time_exceeded:
            self._create_checkpoint()
            self._send_stop_signal()
            return

        # Run one PT iteration
        self._one_iteration()

        # Test convergence periodically
        if self.ptiter % self.pfreq == 0:
            if self._test_convergence():
                break

    # Signal workers to stop
    self._send_stop_signal()
    self._final_output()
```

### Core Iteration: `_one_iteration()`

This is where the magic happens (`_pt_master.py:790-841`):

```python
def _one_iteration(self):
    """Execute one parallel tempering iteration."""

    # Phase 1: Initial Distribution
    # Send first batch of replicas to workers
    for i in range(min(self.nworkers, self.nreplicas)):
        self._send_replica_to_worker(i, worker_rank=i+1)

    next_replica = self.nworkers  # Next replica to assign
    received = 0

    # Phase 2: Dynamic Load Balancing
    while received < self.nreplicas:
        # Wait for any worker to finish (non-blocking probe)
        while not self.comm.Iprobe(source=MPI.ANY_SOURCE):
            time.sleep(self.sleep_seconds)

        # Receive result from worker
        worker_rank = self._receive_result()
        received += 1

        # Assign next job if replicas remain
        if next_replica < self.nreplicas:
            self._send_replica_to_worker(next_replica, worker_rank)
            next_replica += 1

    # Phase 3: Exchange (after skip iterations)
    if self.ptiter >= self.skip:
        self._exchange_coords()

    self.ptiter += 1
```

### Exchange Mechanisms

#### Neighbor Exchange (`_neighbor_exchange()`)

Adjacent replicas attempt to swap coordinates based on the Metropolis criterion:

```python
def _neighbor_exchange(self):
    """Try swapping adjacent replicas."""
    pattern = list(range(self.nreplicas))  # Identity permutation

    # Alternate even/odd pairs each iteration
    start = self.ptiter % 2

    for i in range(start, self.nreplicas - 1, 2):
        # Replicas i and i+1 attempt exchange
        state1 = self.replica_states[i]
        state2 = self.replica_states[i + 1]

        # Calculate acceptance probability
        if self.bias_type == "harmonic":
            # w = exp(0.5 * (dx2^2 - dx1^2) * (k2 - k1))
            delta_dx2 = state2.dx**2 - state1.dx**2
            delta_k = state2.bias_params[0] - state1.bias_params[0]
            w = np.exp(0.5 * delta_dx2 * delta_k)
        else:  # radial_gaussian
            # Compare full potential energies
            E1_at_1 = self._radial_gaussian_energy(state1.dx, state1.bias_params)
            E1_at_2 = self._radial_gaussian_energy(state1.dx, state2.bias_params)
            E2_at_1 = self._radial_gaussian_energy(state2.dx, state1.bias_params)
            E2_at_2 = self._radial_gaussian_energy(state2.dx, state2.bias_params)
            w = np.exp((E1_at_1 + E2_at_2) - (E1_at_2 + E2_at_1))

        # Accept or reject
        if np.random.random() < min(1.0, w):
            pattern[i], pattern[i+1] = pattern[i+1], pattern[i]
            self.exchange_cnts[i, i+1] += 1

    return pattern
```

#### After Exchange

When replicas swap, their coordinates and `dx` values are exchanged, but **energy is set to NaN**:

```python
def _exchange_coords(self):
    pattern = self._find_exchange_buddies()

    for i, j in swapped_pairs:
        # Swap coordinates
        self.replica_states[i].coords, self.replica_states[j].coords = \
            self.replica_states[j].coords, self.replica_states[i].coords

        # Swap dx values
        self.replica_states[i].dx, self.replica_states[j].dx = \
            self.replica_states[j].dx, self.replica_states[i].dx

        # Mark energy for recalculation
        self.replica_states[i].energy = np.nan
        self.replica_states[j].energy = np.nan
```

The `NaN` energy signals workers to recalculate energy on the next iteration.

### Convergence Testing

The master periodically checks if enough uncorrelated samples have been collected:

```python
def _test_convergence(self):
    """Check if simulation has converged."""
    # Find equilibration point using binary search
    equil_point = self._find_equilibration_point()

    # Calculate integrated autocorrelation time
    tau = self._compute_autocorrelation_time(equil_point)

    # Effective sample size
    N_eff = (self.ptiter - equil_point) / (1 + 2*tau)

    # Check relative standard error
    mean = self.timeseries[equil_point:].mean()
    var = self.timeseries2[equil_point:].mean() - mean**2
    rel_err = np.sqrt(var / N_eff) / mean

    return rel_err < self.rel_std_err  # Default: 0.03
```

---

## PT_Worker Class

`PT_Worker` (defined in `_pt_worker.py:1-44`) is intentionally simple - it just runs MC simulations.

### Implementation

```python
class PT_Worker:
    def __init__(self, comm, mcrunner, fix_com=True):
        self.comm = comm
        self.mcrunner = mcrunner
        self.fix_com = fix_com

    def run(self):
        """Worker event loop - receive jobs until termination."""
        while True:
            # Receive replica state from master
            state = self._receive_state()

            # Check for termination signal
            if state.id == -1:
                break

            # Run MC simulation
            state, timeseries = self._one_iteration(state)

            # Send results back to master
            self._send_result(state, timeseries)

    def _one_iteration(self, state):
        """Run MC simulation for one replica."""
        # Set up mcrunner with replica state
        self.mcrunner.set_coords(state.coords)
        self.mcrunner.set_bias_params(state.bias_params)
        self.mcrunner.set_stepsize(state.stepsize)

        # Recalculate energy if NaN (after exchange)
        if np.isnan(state.energy):
            state.energy = self.mcrunner.get_energy()

        # Run MC iterations
        self.mcrunner.run()

        # Collect updated state
        state.coords = self.mcrunner.get_coords()
        state.energy = self.mcrunner.get_energy()
        state.counters = self.mcrunner.get_counters()
        state.stepsize = self.mcrunner.get_stepsize()

        # Calculate dx (distance metric)
        if self.fix_com:
            com = state.coords.reshape(-1, self.ndim).mean(axis=0)
            state.dx = np.linalg.norm(com)
        else:
            state.dx = np.linalg.norm(state.coords[0])

        # Get timeseries for convergence analysis
        timeseries = self.mcrunner.get_timeseries()

        return state, timeseries
```

---

## MPI Communication Protocol

### Message Flow Diagram

```
Time ──────────────────────────────────────────────────────────────────►

Master (Rank 0)          Worker 1           Worker 2           Worker 3
     │                      │                  │                  │
     │──Send(replica[0])───►│                  │                  │
     │──Send(replica[1])────────────────────►  │                  │
     │──Send(replica[2])──────────────────────────────────────►   │
     │                      │                  │                  │
     │   [Workers run MC simulations in parallel]                 │
     │                      │                  │                  │
     │◄──Recv(state+ts)─────│                  │                  │
     │──Send(replica[3])───►│                  │                  │
     │                      │                  │                  │
     │◄──Recv(state+ts)───────────────────────│                  │
     │──Send(replica[4])────────────────────►  │                  │
     │                      │                  │                  │
     │   [Continue until all replicas processed]                  │
     │                      │                  │                  │
     │   [Master performs exchange decisions]                     │
     │                      │                  │                  │
     │   [Next iteration begins]                                  │
     │                      │                  │                  │
     │──Send(id=-1)────────►│                  │                  │
     │──Send(id=-1)──────────────────────────►│                  │
     │──Send(id=-1)──────────────────────────────────────────────►│
     │                      │                  │                  │
     ▼                      ▼                  ▼                  ▼
   [Exit]               [Exit]             [Exit]             [Exit]
```

### Buffer Format

The receive buffer contains both state and timeseries data:

```
┌─────────────────────────────────────────────────────────────────┐
│                        recv_buffer                              │
├────────────────────────────────┬────────────────────────────────┤
│      Serialized ReplicaState   │         Timeseries Data        │
│                                │                                │
│  [id, dx, energy, bias_params, │  [dx_0, dx_1, ..., dx_niter]   │
│   stepsize, coords, counters]  │                                │
├────────────────────────────────┼────────────────────────────────┤
│        state.size() values     │        mcrunner.niter values   │
└────────────────────────────────┴────────────────────────────────┘
```

### MPI Operations Used

| Operation | Purpose |
|-----------|---------|
| `comm.Send(data, dest)` | Blocking send of numpy array |
| `comm.Recv(buffer, source)` | Blocking receive into pre-allocated buffer |
| `comm.Iprobe(source)` | Non-blocking check for incoming message |
| `status.Get_source()` | Identify which worker sent the message |

### Non-Blocking Probe with Sleep

To avoid busy-waiting (100% CPU), the master uses `Iprobe` with sleep:

```python
while not self.comm.Iprobe(source=MPI.ANY_SOURCE, status=status):
    time.sleep(self.sleep_seconds)  # Default: 100 microseconds
```

---

## Complete Workflow

### Step-by-Step Walkthrough

```
1. INITIALIZATION
   ├── Master creates nreplicas ReplicaState objects
   │   └── Each replica has different bias_params (e.g., spring constants)
   │       Replica 0: k=0.0 (free diffusion)
   │       Replica 1: k=0.5
   │       ...
   │       Replica N: k=10.0 (strongly constrained)
   │
   └── Workers initialize mcrunner objects

2. ITERATION LOOP (ptiter = 0 to max_ptiter)
   │
   ├── 2a. DISTRIBUTION PHASE
   │   └── Master sends replica states to available workers
   │       (Dynamic assignment - not round-robin)
   │
   ├── 2b. EXECUTION PHASE (parallel)
   │   ├── Each worker:
   │   │   ├── Receives replica state
   │   │   ├── Sets up mcrunner
   │   │   ├── If energy is NaN: recalculate
   │   │   ├── Runs MC simulation (niter steps)
   │   │   ├── Collects updated state + timeseries
   │   │   └── Sends results to master
   │   │
   │   └── Master:
   │       ├── Probes for completed workers
   │       ├── Receives results
   │       └── Assigns remaining replicas
   │
   ├── 2c. COLLECTION PHASE
   │   └── Master waits for all workers to finish
   │       Updates replica_states and timeseries arrays
   │
   ├── 2d. EXCHANGE PHASE (if ptiter >= skip)
   │   ├── Neighbor exchange (even/odd alternating):
   │   │   ├── Compare replicas i and i+1
   │   │   ├── Calculate w = exp(0.5*(dx2²-dx1²)*(k2-k1))
   │   │   ├── Accept swap with probability min(1, w)
   │   │   └── If accepted: swap coords and dx, set energy=NaN
   │   │
   │   └── Update exchange statistics
   │
   └── 2e. OUTPUT (every pfreq iterations)
       ├── Print replica status
       ├── Dump timeseries to disk
       └── Test convergence

3. CONVERGENCE CHECK
   ├── Find equilibration point (binary search)
   ├── Compute autocorrelation time τ
   ├── Calculate relative standard error
   └── If error < threshold (3%): STOP

4. TERMINATION
   ├── Master sends id=-1 to all workers
   ├── Workers exit their loops
   └── Master outputs final results
```

### Exchange Acceptance Criterion

For **harmonic bias** with spring constant `k`:

```
U_bias(r) = 0.5 * k * dx²

where dx = distance from center of mass
```

When replicas i (at dx₁, with k₁) and j (at dx₂, with k₂) attempt exchange:

```
ΔE = [U_bias(dx₁, k₂) + U_bias(dx₂, k₁)] - [U_bias(dx₁, k₁) + U_bias(dx₂, k₂)]
   = 0.5*k₂*dx₁² + 0.5*k₁*dx₂² - 0.5*k₁*dx₁² - 0.5*k₂*dx₂²
   = 0.5*(dx₂² - dx₁²)*(k₁ - k₂)

w = exp(-ΔE) = exp(0.5*(dx₂² - dx₁²)*(k₂ - k₁))

Accept exchange with probability: min(1, w)
```

**Physical Intuition:**
- Low-k replicas explore large displacements (large dx)
- High-k replicas are constrained near the origin (small dx)
- Exchange allows high-k configurations to "borrow" exploration from low-k
- This enhances sampling across the entire dx range

---

## Source File References

| Component | File | Lines |
|-----------|------|-------|
| `ReplicaState` | `_pt_master.py` | 56-138 |
| `PT_Master` | `_pt_master.py` | 140-1279 |
| `PT_Master.run()` | `_pt_master.py` | 432-463 |
| `PT_Master._one_iteration()` | `_pt_master.py` | 790-841 |
| `PT_Master._neighbor_exchange()` | `_pt_master.py` | 946-1010 |
| `PT_Worker` | `_pt_worker.py` | 1-44 |
| Entry point | `bv_parallel_tempering.py` | - |
