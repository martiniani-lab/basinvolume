from __future__ import division
from future import standard_library

standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object
import logging
import time
import copy
import os
import random
import time
import pickle
import numpy as np
from mpi4py import MPI
from enum import Enum, unique  # Package enum34
from pymbar.timeseries import detect_equilibration_binary_search
from basinvolume.utils import trymakedir, integratedAutocorrelationTime_fft
from basinvolume.post_processing import (
    spring_constants_variable_transform,
    spring_constants_positionlinspace,
    spring_constants_linspace,
    spring_constants_logspace,
    neg_spring_constants_positionlinspace,
    neg_spring_constants_logspace,
)
from basinvolume.monte_carlo import IndependenceSampling
from basinvolume.spheres import BV_MCRunner_State
import pandas as pd

# Additional imports for checkpointing
import json
import tempfile
import shutil
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    
try:
    import dill
    HAS_DILL = True
except ImportError:
    HAS_DILL = False

# Additional imports for checkpointing
import json
import tempfile
import shutil
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    
try:
    import dill
    HAS_DILL = True
except ImportError:
    HAS_DILL = False


@unique
class ExchangeScheme(Enum):
    NEIGHBOR_EXCHANGE = 1
    INDEPENDENCE_SAMPLING = 2


class ReplicaState(BV_MCRunner_State):
    """
    This class represents the state of a parallel tempering replica
    """

    def __init__(self, id, mcrunner_state):
        super(ReplicaState, self).__init__(state=mcrunner_state)
        self.id = id
        self.dx = 0
        self.swap_accepted_count = 0
        self.swap_rejected_count = 0

    def set_mc_state(self, mcrunner_state):
        self._set_state(mcrunner_state)

    def serialize(self):
        if isinstance(self.bias_params, float):
            n_bias_params = 1
        else:
            n_bias_params = len(self.bias_params)

        data = np.empty(self.size(), dtype="d")
        data[0] = self.id
        data[1] = self.dx
        data[2] = self.energy
        data[3 : 3 + n_bias_params] = self.bias_params
        data[3 + n_bias_params] = self.stepsize
        data[4 + n_bias_params] = self.takestep_count
        data[5 + n_bias_params : 5 + n_bias_params + len(self.coords)] = self.coords
        data[
            5
            + n_bias_params
            + len(self.coords) : 5
            + n_bias_params
            + len(self.coords)
            + len(self.counters)
        ] = self.counters
        data[
            5 + n_bias_params + len(self.coords) + len(self.counters) :
        ] = self.step_adaptation_counters
        return data

    def deserialize(self, value):
        if isinstance(self.bias_params, float):
            n_bias_params = 1
        else:
            n_bias_params = len(self.bias_params)

        self.id = int(value[0])
        self.dx = value[1]
        self.energy = value[2]
        self.bias_params = value[3 : 3 + n_bias_params]
        self.stepsize = value[3 + n_bias_params]
        self.takestep_count = int(value[4 + n_bias_params])
        self.coords = value[5 + n_bias_params : 5 + n_bias_params + len(self.coords)]
        self.counters = np.array(
            value[
                5
                + n_bias_params
                + len(self.coords) : 5
                + n_bias_params
                + len(self.coords)
                + len(self.counters)
            ],
            dtype="uintp",
        )
        self.step_adaptation_counters = np.array(
            value[5 + n_bias_params + len(self.coords) + len(self.counters) :], dtype="uintp"
        )

    def size(self):
        if isinstance(self.bias_params, float):
            n_bias_params = 1
        else:
            n_bias_params = len(self.bias_params)
        return (
            5
            + n_bias_params
            + len(self.coords)
            + len(self.counters)
            + len(self.step_adaptation_counters)
        )


class PT_Master(object):
    """
    This class manages a job queue that sends jobs to Parallel Tempering replicas.
    It is run in a parallel process on rank 0.

    Attributes:
        nreplicas (int): The number of replicas.
        example_mcrunner (object): An instance of the example_mcrunner class.
        kmax (float): The maximum spring constant.
        kmin (float): The minimum spring constant.
        u2meank0 (float): The mean potential energy.
        max_ptiter (int): The maximum number of parallel tempering iterations.
        pfreq (int): The frequency of printing status updates.
        skip (int): The number of swaps to skip for equilibration.
        test_convergence (bool): Flag indicating whether to test for convergence.
        fast_ct (bool): Flag indicating whether to use fast convergence testing.
        rel_std_err (float): The relative standard error.
        min_window (float): The minimum window size.
        max_eq_time (float): The maximum equilibration time.
        numnegk (int): The number of negative spring constants.
        lownegk (float): The lowest negative spring constant.
        k_spreading (str): The spreading scheme for spring constants.
        bias (str): The biasing scheme.
        print_status (bool): Flag indicating whether to print status updates.
        base_directory (str): The base directory for storing results.
        bs_nodes (int): The number of nodes for block sampling.
        eq_min_ptiter (int): The minimum number of PT iterations for equilibration.
        eq_max_ptiter (int): The maximum number of PT iterations for equilibration.
        sleep_seconds (float): The sleep time in seconds.
        exchange_scheme (ExchangeScheme): The exchange scheme.
        checkpoint_time (float): The checkpoint time.
        checkpoint_file (str): The checkpoint file name.
        record_traj_npoints (int): The number of points to record in the trajectory.
    """

    def __init__(
        self,
        nreplicas,
        example_mcrunner,
        kmax,
        kmin,
        u2meank0,
        max_ptiter=10,
        pfreq=1,
        skip=0,
        test_convergence=True,
        fast_ct=False,
        rel_std_err=0.03,
        min_window=2.5e5,
        max_eq_time=2.5e5,
        numnegk=0,
        lownegk=-2.5,
        k_spreading="gausslobato",
        bias="harmonic",
        print_status=False,
        base_directory=None,
        bs_nodes=100,
        eq_min_ptiter=None,
        eq_max_ptiter=None,
        sleep_seconds=0.0001,
        exchange_scheme=ExchangeScheme.NEIGHBOR_EXCHANGE,
        checkpoint_time=None,
        checkpoint_file="checkpoint.dmp",
        checkpoint_format="auto",
    ):
        self.nreplicas = nreplicas
        self.sleep_seconds = sleep_seconds
        self.comm = MPI.COMM_WORLD
        self.nworkers = self.comm.Get_size() - 1  # total number of workers
        self.rank = self.comm.Get_rank()  # this is the unique identifier for the process
        self.nparticles = example_mcrunner.nparticles
        self.bdim = example_mcrunner.bdim
        self.kmax = kmax
        self.kmin = kmin
        self.max_ptiter = max_ptiter
        self.ptiter = 0
        self.print_status = print_status
        self.skip = skip  # might want to skip the first few swaps to allow for equilibration
        self.pfreq = pfreq
        self.record_traj_npoints = record_traj_npoints
        self.NO_EXCHANGE = (
            -12345
        )  # this NEGATIVE number in exchange pattern means that no exchange should be attempted
        if base_directory is None:
            self.base_directory = os.path.join(os.getcwd(), "ptmc_results")
        else:
            self.base_directory = base_directory
        self.exchange_choice = random.randint(0, 1)
        self.anyswap = False  # set to true if any swap happened
        self.permutation_pattern = np.zeros(
            self.nreplicas, dtype="int32"
        )  # useful for printing exchange permutations
        self.u2meank0 = u2meank0
        self.mcrunner_niter = example_mcrunner.niter
        self.mcrunner_eqsteps = int(example_mcrunner.equilibration_steps)
        self.mcrunner_potential = example_mcrunner.potential
        self.test_convergence = test_convergence
        self.eq_time = 0  # time at which equilibration was reached
        self.fast_ct = fast_ct
        self.rel_std_err = rel_std_err  # relative standard error
        self.last_rel_std_errs = np.zeros(
            self.nreplicas
        )  # last measured relative standard errors for each replica
        if eq_min_ptiter is None:
            eq_min_ptiter = int(self.max_ptiter * 0.95)
        self.eq_min_ptiter = int(eq_min_ptiter)
        if eq_max_ptiter is None:
            eq_max_ptiter = int(2e6 / example_mcrunner.niter)
        self.eq_max_ptiter = int(eq_max_ptiter)
        self.min_window = int(min_window)
        self.max_eq_time = int(max_eq_time)
        self.bs_nodes = int(bs_nodes)
        self.numnegk = int(numnegk)
        self.lownegk = lownegk
        self.k_spreading = k_spreading
        self.bias = bias
        self._init_replicas(example_mcrunner)
        self._init_timeseries()
        self._init_print()
        self.recv_buffer = np.empty(self.replica_states[0].size() + self.mcrunner_niter, dtype="d")
        self.exchange_cnts = np.zeros((self.nreplicas, self.nreplicas), dtype="int32")
        self.exchange_scheme = exchange_scheme
        self._init_sampling()
        self.checkpoint_time = checkpoint_time
        self.checkpoint_file = checkpoint_file
        self.checkpoint_format = checkpoint_format
        self._validate_checkpoint_format()
        assert self.nreplicas > self.nworkers
        assert self.eq_min_ptiter > self.skip
        assert self.max_ptiter > self.eq_min_ptiter
        assert self.eq_max_ptiter > self.eq_min_ptiter
        assert (
            self.eq_max_ptiter - self.eq_min_ptiter
        ) * self.mcrunner_niter > self.min_window  # Condition on the minimal window size
        if not (self.min_window > self.mcrunner_eqsteps):
            logging.info("self.min_window: {}".format(self.min_window))
            logging.info("self.mcrunner_eqsteps: {}".format(self.mcrunner_eqsteps))
        assert self.min_window > self.mcrunner_eqsteps
        assert self.max_eq_time > self.mcrunner_eqsteps

    def init_state(self, base_directory=None, checkpoint_time=None):
        if base_directory is not None:
            self.base_directory = base_directory
        if checkpoint_time is not None:
            self.checkpoint_time = checkpoint_time
        self.comm = MPI.COMM_WORLD
        self._init_sampling()
        self._init_print(append=True)

    def _init_sampling(self):
        i32max = np.iinfo(np.int32).max
        self.seed_exchanges = random.randint(0, i32max)
        logging.info("seed_exchanges: %i" % self.seed_exchanges)
        if self.exchange_scheme is ExchangeScheme.NEIGHBOR_EXCHANGE:
            self._calculate_exchange = self._neighbor_exchange
            np.random.seed(self.seed_exchanges)
        elif self.exchange_scheme is ExchangeScheme.INDEPENDENCE_SAMPLING:
            self._calculate_exchange = self._independence_sampling
            self.indep_sampling = IndependenceSampling(self.seed_exchanges)
        else:
            raise ValueError("Unknown exchange scheme (%s)" % self.exchange_scheme.name)

    def _init_replicas(self, example_mcrunner):
        bias_params = self._get_bias_params()
        start_state = example_mcrunner.get_complete_state()
        self.replica_states = []
        for i in range(self.nreplicas):
            self.replica_states.append(ReplicaState(i, start_state))
            self.replica_states[-1].bias_params = bias_params[i]

    def _init_timeseries(self):
        self.replica_timeseries = [[] for _ in range(self.nreplicas)]
        self.replica_timeseries2 = [[] for _ in range(self.nreplicas)]

    def _init_print(self, append=False):
        current_dir = os.getcwd()
        if os.path.basename(current_dir) == self.base_directory:
            self.base_directory = current_dir

        if append:
            mode = "a"
        else:
            mode = "w"
            trymakedir(self.base_directory)
            self._print_bias_params()
        self.ex_outstream = open(os.path.join(self.base_directory, "exchanges"), mode)
        self.permutations_stream = open(
            os.path.join(self.base_directory, "rem_permutations"), mode
        )
        self.status_streams = []
        self.histogram_mean_streams = []
        for ireplica in range(self.nreplicas):
            directory = os.path.join(self.base_directory, str(ireplica))
            if not append:
                trymakedir(directory)
                self._print_parameters(ireplica)
            self.status_streams.append(open(os.path.join(directory, "status"), mode))
            self.histogram_mean_streams.append(open(os.path.join(directory, "hist_mean"), mode))
            if not append:
                self.histogram_mean_streams[ireplica].write(
                    "{:<15}\t{:<15}\t{:<15}\t{:<15}\n".format(
                        "iteration", "<(x-x0)**2>", "variance", "std_err"
                    )
                )

    def _get_bias_params(self):
        """
        set up the spring constants (temperatures).
        They can be distributed exponentially if the k_spreading option reads gausslobato.
        The other options are linspace (linearly spaced) or logspace (log spaced).
        We order the spring constants from highest to lowest, to calculate the
        more costly replica first.
        """
        if self.bias == "harmonic":
            nposk = self.nreplicas - self.numnegk  # number of positive k

            if self.k_spreading == "gausslobato":
                Karray = spring_constants_variable_transform(
                    nposk + 1,
                    self.kmax,
                    self.u2meank0,
                    self.nparticles,
                    self.bdim,
                    self.kmin,
                )

            elif self.k_spreading == "linspace":
                Karray = spring_constants_linspace(nposk + 1, self.kmax, self.kmin)
            elif self.k_spreading == "logspace":
                Karray = spring_constants_logspace(nposk + 1, self.kmax, self.kmin)
            elif self.k_spreading == "positionlinspace":
                Karray = spring_constants_positionlinspace(
                    nposk + 1, self.kmax, self.u2meank0, self.nparticles, self.bdim
                )
            else:
                raise NotImplementedError
            Karray = Karray[
                :-1
            ]  # exclude kmax entry, no need to be simulated, mean is already available
            if self.numnegk > 0:
                if self.k_spreading == "positionlinspace":
                    negKarray = neg_spring_constants_positionlinspace(
                        self.numnegk,
                        nposk + 1,
                        self.kmax,
                        self.u2meank0,
                        self.nparticles,
                        self.bdim,
                    )
                    for x in negKarray[::-1]:
                        Karray.insert(0, x)
                else:
                    negKarray = neg_spring_constants_logspace(self.numnegk, self.lownegk)
                    for x in negKarray[::-1]:
                        Karray.insert(0, x)
            # Reverse Karray for backwards compatibility
            Karray = Karray[::-1]
            return Karray

        elif self.bias == "radial_gaussian":
            # Need both centerings and widths for the gaussians

            nposk = self.nreplicas - self.numnegk  # number of positive k

            # Set all widths equal to RMSD of kmin / nposk, or k =(nposk / RMSD)^2 to keep spring constants
            u2meankmax = 0.5 * (self.nparticles * self.bdim) / self.kmax
            r_kmax = np.sqrt(u2meankmax)
            width = (np.sqrt(self.u2meank0) - r_kmax) / nposk
            k = 1 / (width * width)
            Karray = k * np.ones(self.nreplicas)

            # Set center positions of 1d gaussians for "positive" replicas to be linearly spaced between rmin >=0 and RMSD (both excluded)
            # If rmin is too close to the origin, accumulation at 0 happens
            # As a rule of thumb, start from kmax to avoid silly issues at large nreplicas
            rmin = r_kmax
            spacing = (np.sqrt(self.u2meank0) - rmin) / (nposk + 1)
            l0array = rmin + (np.arange(self.nreplicas)) * spacing

            # only start the log part at some cut-off distance to avoid bad behaviour near 0
            r_cutoffarray = 0.5 * rmin * np.ones(self.nreplicas)

            # Force the k = 0 case
            Karray[nposk] = 0.0
            l0array[nposk] = 0.0
            r_cutoffarray[nposk] = 0.0

            return np.transpose(np.vstack([Karray, l0array, r_cutoffarray]))

        else:
            raise NotImplementedError

    def run(self):
        if self.checkpoint_time is not None:
            start_time = time.time()
        self.created_checkpoint = False
        while self.ptiter < self.max_ptiter and not self.created_checkpoint:
            logging.debug("Iteration {}".format(self.ptiter))
            self._one_iteration()
            if self.ptiter == self.skip:
                self._print_stepsizes()
            if self.ptiter >= self.max_ptiter:
                self.max_ptiter = self._test_convergence()
            if (
                self.checkpoint_time is not None
                and time.time() - start_time > self.checkpoint_time
            ):
                self.created_checkpoint = True

        # Stop workers
        for iworker in range(self.nworkers):
            self.comm.Send(np.array([-1], dtype="d"), dest=iworker + 1)

        if self.created_checkpoint:
            self._create_checkpoint()
            logging.info("Created checkpoint")
        else:
            self._print_data()
            # Always write status at the very end
            # if self.print_status:
            self._print_status()
            self._print_exchanges()
            self._flush_close_streams()
            logging.info("Master finished")

    def _create_checkpoint(self):
        """Create checkpoint with improved error handling and multiple format support"""
        logging.info("Starting checkpoint creation...")
        
        try:
            # Prepare data for checkpointing
            temp_attrs = self._prepare_checkpoint_data()
            
            checkpoint_path = os.path.join(self.base_directory, self.checkpoint_file)
            format_type = self._get_checkpoint_format(self.checkpoint_file)
            
            # Choose what to save based on format
            if format_type == 'json':
                # For JSON, create a dictionary representation
                data = self._create_checkpoint_dict()
            else:
                # For pickle/joblib/dill, save the entire object
                data = self
            
            # Save checkpoint atomically
            self._save_checkpoint_atomic(data, checkpoint_path, format_type)
            
            logging.info(f"Checkpoint created successfully using {format_type} format")
            
        except Exception as e:
            logging.error(f"Failed to create checkpoint: {e}")
            logging.error(f"Checkpoint format: {format_type}")
            logging.error(f"Checkpoint path: {checkpoint_path}")
            raise

    def _prepare_checkpoint_data(self):
        """Prepare data for checkpointing by removing non-serializable objects"""
        # Store references to objects we'll remove temporarily
        temp_attrs = {}
        
        # Remove MPI comm and function references
        if hasattr(self, 'comm'):
            temp_attrs['comm'] = self.comm
            del self.comm
        if hasattr(self, '_calculate_exchange'):
            temp_attrs['_calculate_exchange'] = self._calculate_exchange
            del self._calculate_exchange
        if hasattr(self, 'indep_sampling'):
            temp_attrs['indep_sampling'] = self.indep_sampling
            del self.indep_sampling
            
        # Close and remove file streams
        self._flush_close_streams()
        if hasattr(self, 'ex_outstream'):
            del self.ex_outstream
        if hasattr(self, 'permutations_stream'):
            del self.permutations_stream
        if hasattr(self, 'histogram_mean_streams'):
            del self.histogram_mean_streams
        if hasattr(self, 'status_streams'):
            del self.status_streams
            
        return temp_attrs

    def _create_checkpoint_dict(self):
        """Create a dictionary representation of the checkpoint data for JSON serialization"""
        # Helper function to convert numpy arrays to lists for JSON serialization
        def serialize_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Serialize replica states data
        replica_states_data = []
        for rs in self.replica_states:
            state_dict = {}
            for key, value in rs.__dict__.items():
                if key == 'coords':
                    state_dict[key] = value.tolist() if isinstance(value, np.ndarray) else value
                elif key == 'counters':
                    state_dict[key] = value.tolist() if isinstance(value, np.ndarray) else value
                elif key == 'step_adaptation_counters':
                    state_dict[key] = value.tolist() if isinstance(value, np.ndarray) else value
                elif key == 'bias_params':
                    if isinstance(value, np.ndarray):
                        state_dict[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        state_dict[key] = float(value)
                    else:
                        state_dict[key] = value
                else:
                    state_dict[key] = serialize_numpy(value)
            replica_states_data.append(state_dict)

        checkpoint_data = {
            'nreplicas': int(self.nreplicas),
            'sleep_seconds': float(self.sleep_seconds),
            'nworkers': int(self.nworkers),
            'rank': int(self.rank),
            'nparticles': int(self.nparticles),
            'bdim': int(self.bdim),
            'kmax': float(self.kmax),
            'kmin': float(self.kmin),
            'max_ptiter': int(self.max_ptiter),
            'ptiter': int(self.ptiter),
            'print_status': bool(self.print_status),
            'skip': int(self.skip),
            'pfreq': int(self.pfreq),
            'NO_EXCHANGE': int(self.NO_EXCHANGE),
            'base_directory': str(self.base_directory),
            'exchange_choice': int(self.exchange_choice),
            'anyswap': bool(self.anyswap),
            'permutation_pattern': self.permutation_pattern.tolist(),
            'u2meank0': float(self.u2meank0),
            'mcrunner_niter': int(self.mcrunner_niter),
            'mcrunner_eqsteps': int(self.mcrunner_eqsteps),
            'test_convergence': bool(self.test_convergence),
            'eq_time': int(self.eq_time),
            'fast_ct': bool(self.fast_ct),
            'rel_std_err': float(self.rel_std_err),
            'last_rel_std_errs': self.last_rel_std_errs.tolist(),
            'eq_min_ptiter': int(self.eq_min_ptiter),
            'eq_max_ptiter': int(self.eq_max_ptiter),
            'min_window': int(self.min_window),
            'max_eq_time': int(self.max_eq_time),
            'bs_nodes': int(self.bs_nodes),
            'numnegk': int(self.numnegk),
            'lownegk': float(self.lownegk),
            'k_spreading': str(self.k_spreading),
            'bias': str(self.bias),
            'recv_buffer': self.recv_buffer.tolist(),
            'exchange_cnts': self.exchange_cnts.tolist(),
            'exchange_scheme': int(self.exchange_scheme.value),
            'checkpoint_time': self.checkpoint_time,
            'checkpoint_file': str(self.checkpoint_file),
            'checkpoint_format': str(self.checkpoint_format),
            'replica_timeseries': self.replica_timeseries,
            'replica_timeseries2': self.replica_timeseries2,
            'replica_states_data': replica_states_data,
            'seed_exchanges': getattr(self, 'seed_exchanges', None),
            'created_checkpoint': getattr(self, 'created_checkpoint', False)
        }
        return checkpoint_data

    def _save_checkpoint_atomic(self, data, filepath, format_type):
        """Save checkpoint data atomically using a temporary file"""
        temp_dir = os.path.dirname(filepath)
        temp_file = None
        
        try:
            # Create temporary file in the same directory
            with tempfile.NamedTemporaryFile(mode='wb', dir=temp_dir, delete=False) as temp_file:
                temp_filepath = temp_file.name
                
                logging.info(f"Creating checkpoint with {format_type} format...")
                logging.info(f"Temporary file: {temp_filepath}")
                
                if format_type == 'pickle':
                    pickle.dump(data, temp_file, protocol=pickle.HIGHEST_PROTOCOL)
                elif format_type == 'joblib':
                    joblib.dump(data, temp_file, compress=3)
                elif format_type == 'dill':
                    dill.dump(data, temp_file, protocol=dill.HIGHEST_PROTOCOL)
                elif format_type == 'json':
                    # For JSON, we need text mode
                    temp_file.close()
                    with open(temp_filepath, 'w') as json_file:
                        json.dump(data, json_file, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
                
                if format_type != 'json':
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
            
            # Verify the file was written correctly
            if not os.path.exists(temp_filepath):
                raise IOError(f"Temporary checkpoint file was not created: {temp_filepath}")
            
            file_size = os.path.getsize(temp_filepath)
            if file_size == 0:
                raise IOError(f"Temporary checkpoint file is empty: {temp_filepath}")
            
            logging.info(f"Checkpoint file size: {file_size} bytes")
            
            # Atomically move temp file to final location
            shutil.move(temp_filepath, filepath)
            logging.info(f"Checkpoint saved successfully to: {filepath}")
            
            # Verify final file
            final_size = os.path.getsize(filepath)
            if final_size != file_size:
                raise IOError(f"File size mismatch after move: expected {file_size}, got {final_size}")
                
        except Exception as e:
            # Clean up temp file if it exists
            if temp_file and os.path.exists(temp_filepath):
                try:
                    os.unlink(temp_filepath)
                except:
                    pass
            logging.error(f"Failed to save checkpoint: {e}")
            raise

    def _get_checkpoint_format(self, filename):
        """Determine checkpoint format from filename or configured format"""
        if self.checkpoint_format == "auto":
            # Auto-detect from file extension
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.json':
                return 'json'
            elif ext == '.joblib' and HAS_JOBLIB:
                return 'joblib'
            elif ext == '.dill' and HAS_DILL:
                return 'dill'
            else:
                return 'pickle'  # default
        else:
            return self.checkpoint_format

    @classmethod
    def load_checkpoint(cls, checkpoint_path, example_mcrunner=None):
        """Load checkpoint with automatic format detection"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Determine format from file extension
        ext = os.path.splitext(checkpoint_path)[1].lower()
        
        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        logging.info(f"File size: {os.path.getsize(checkpoint_path)} bytes")
        
        try:
            if ext == '.json':
                return cls._load_checkpoint_json(checkpoint_path, example_mcrunner)
            elif ext == '.joblib' and HAS_JOBLIB:
                logging.info("Loading with joblib...")
                return joblib.load(checkpoint_path)
            elif ext == '.dill' and HAS_DILL:
                logging.info("Loading with dill...")
                with open(checkpoint_path, 'rb') as f:
                    return dill.load(f)
            else:
                # Default to pickle
                logging.info("Loading with pickle...")
                with open(checkpoint_path, 'rb') as f:
                    return pickle.load(f)
                    
        except EOFError as e:
            logging.error(f"EOF error loading checkpoint - file may be corrupted: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            raise

    @classmethod
    def _load_checkpoint_json(cls, checkpoint_path, example_mcrunner):
        """Load checkpoint from JSON format"""
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        # Create a new instance
        if example_mcrunner is None:
            raise ValueError("example_mcrunner is required for loading JSON checkpoints")
        
        # Create instance with data from checkpoint
        instance = cls(
            nreplicas=data['nreplicas'],
            example_mcrunner=example_mcrunner,
            kmax=data['kmax'],
            kmin=data['kmin'],
            u2meank0=data['u2meank0'],
            max_ptiter=data['max_ptiter'],
            pfreq=data['pfreq'],
            skip=data['skip'],
            test_convergence=data['test_convergence'],
            fast_ct=data['fast_ct'],
            rel_std_err=data['rel_std_err'],
            min_window=data['min_window'],
            max_eq_time=data['max_eq_time'],
            numnegk=data['numnegk'],
            lownegk=data['lownegk'],
            k_spreading=data['k_spreading'],
            bias=data['bias'],
            print_status=data['print_status'],
            base_directory=data['base_directory'],
            bs_nodes=data['bs_nodes'],
            eq_min_ptiter=data['eq_min_ptiter'],
            eq_max_ptiter=data['eq_max_ptiter'],
            sleep_seconds=data['sleep_seconds'],
            exchange_scheme=ExchangeScheme(data['exchange_scheme']),
            checkpoint_time=data['checkpoint_time'],
            checkpoint_file=data['checkpoint_file'],
            checkpoint_format=data['checkpoint_format']
        )
        
        # Restore state
        instance.ptiter = data['ptiter']
        instance.exchange_choice = data['exchange_choice']
        instance.anyswap = data['anyswap']
        instance.permutation_pattern = np.array(data['permutation_pattern'], dtype='int32')
        instance.eq_time = data['eq_time']
        instance.last_rel_std_errs = np.array(data['last_rel_std_errs'])
        instance.recv_buffer = np.array(data['recv_buffer'])
        instance.exchange_cnts = np.array(data['exchange_cnts'], dtype='int32')
        instance.replica_timeseries = data['replica_timeseries']
        instance.replica_timeseries2 = data['replica_timeseries2']
        instance.seed_exchanges = data.get('seed_exchanges')
        instance.created_checkpoint = data.get('created_checkpoint', False)
        
        # Restore replica states
        for i, state_data in enumerate(data['replica_states_data']):
            for key, value in state_data.items():
                if key == 'coords':
                    instance.replica_states[i].coords = np.array(value)
                elif key == 'counters':
                    instance.replica_states[i].counters = np.array(value, dtype='uintp')
                elif key == 'step_adaptation_counters':
                    instance.replica_states[i].step_adaptation_counters = np.array(value, dtype='uintp')
                else:
                    setattr(instance.replica_states[i], key, value)
        
        return instance

    def _one_iteration(self):
        """Perform one parallel tempering iteration

        Each PT iteration consists of the following steps:

        * distribute work, i.e. run the MCrunners for a predefined number of steps
        * collect the results
        * attempt an exchange
        """
        # Start with the slowest replica (lowest k)
        current_replica = self.nreplicas - 1

        # Send the first job to every worker
        for i in range(self.nworkers):
            self.comm.Send(self.replica_states[current_replica].serialize(), dest=i + 1)
            current_replica -= 1

        # Send the remaining jobs to finished workers
        while current_replica >= 0:
            if self.sleep_seconds > 0:
                while not self.comm.Iprobe(source=MPI.ANY_SOURCE):
                    time.sleep(self.sleep_seconds)
            finished_worker = self._receive_result()
            self.comm.Send(
                self.replica_states[current_replica].serialize(),
                dest=finished_worker,
            )
            current_replica -= 1

        # Wait for all workers to finish
        for _ in range(self.nworkers):
            # calculate total time
            start_time = time.time()
            if self.sleep_seconds > 0:
                while not self.comm.Iprobe(source=MPI.ANY_SOURCE):
                    time.sleep(self.sleep_seconds)
            self._receive_result()
            end_time = time.time()
            logging.debug("Time to recieve all results: %f" % (end_time - start_time))

        if self.ptiter >= self.skip:
            self._exchange_coords()
            if self.record_traj_npoints != -1 and self.ptiter % self.record_traj_npoints == 0:
                self._dump_traj()

            # print and increase parallel tempering count and test convergence
            if self.ptiter % self.pfreq == 0:
                self.max_ptiter = self._test_convergence()
                self._print_data()
            if self.print_status:
                self._print_status()
        self.ptiter += 1

    def _receive_result(self):
        status = MPI.Status()
        self.comm.Recv(self.recv_buffer, source=MPI.ANY_SOURCE, status=status)
        src = status.Get_source()
        replica_id = int(self.recv_buffer[0])
        state_size = self.replica_states[0].size()
        self.replica_states[replica_id].deserialize(self.recv_buffer[:state_size].copy())
        recv_timeseries = self.recv_buffer[state_size:]
        self.replica_timeseries[replica_id].extend(recv_timeseries)
        recv_timeseries2 = np.power(recv_timeseries, 2)
        self.replica_timeseries2[replica_id].extend(recv_timeseries2)
        return src

    def _exchange_coords(self):
        """
        Exchange the replica states according to _find_exchange_buddies
        """
        # dx_string = "dx: "
        # for i in xrange(self.nreplicas):
        #     dx_string += str(self.replica_states[i].dx) + ", "
        # logging.debug(dx_string)

        # find exchange pattern (list of exchange buddies)
        exchange_pattern = self._find_exchange_buddies()

        # swap replica coordinates and dx
        old_coords = [replica.coords for replica in self.replica_states]
        old_dxs = [replica.dx for replica in self.replica_states]
        for istate, ibuddy in enumerate(exchange_pattern):
            if ibuddy != self.NO_EXCHANGE:
                # Swap coordinates and dx
                self.replica_states[istate].coords = old_coords[ibuddy]
                self.replica_states[istate].dx = old_dxs[ibuddy]

                # Set energy to NaN, since it needs to be recalculated
                self.replica_states[istate].energy = np.nan

    def _find_exchange_buddies(self):
        """
        This function determines the exchange pattern.
        An exchange pattern array is constructed, filled with self.NO_EXCHANGE
        which signifies that no exchange should be attempted. If the swap attempt
        is successful this value is replaced with the id of the replica with
        which to perform the swap.
        """
        exchange_pattern = np.empty(self.nreplicas, dtype="int32")
        exchange_pattern.fill(self.NO_EXCHANGE)  # reset exchange pattern to no exchange
        self.anyswap = False

        self._calculate_exchange(exchange_pattern)

        for i in range(self.nreplicas):
            if exchange_pattern[i] == i:
                exchange_pattern[i] = self.NO_EXCHANGE

        # record self.permutation_pattern to print permutations in print function
        if self.anyswap:
            for i, buddy in enumerate(exchange_pattern):
                if buddy != self.NO_EXCHANGE:
                    self.replica_states[i].swap_accepted_count += 1
                    self.exchange_cnts[i, buddy] += 1
                    self.permutation_pattern[i] = buddy + 1  # to conform to fortran notation
                else:
                    self.replica_states[i].swap_rejected_count += 1
                    self.permutation_pattern[i] = i + 1  # to conform to fortran notation
            self._print_permutations()

        return exchange_pattern

    def _independence_sampling(self, exchange_pattern):
        if self.bias == "harmonic":
            dxs = np.array([replica.dx for replica in self.replica_states])
            betas = np.array([replica.bias_params for replica in self.replica_states])

            # According to Chodera & Shirts 2011 nreplicas**3 to nreplicas**5 exchanges
            # should be sufficient
            nexchanges = self.nreplicas**3

            naccept = self.indep_sampling.exchange(exchange_pattern, dxs, betas, nexchanges)

            if naccept > 0:
                self.anyswap = True

            logging.debug("Acceptance ratio: %f" % (naccept / nexchanges))
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                for i in range(self.nreplicas):
                    j = exchange_pattern[i]
                    if i != j:
                        self.ex_outstream.write(
                            "{}: Accepting exchange {:>2} -> {:<2}: "
                            "{:.4g} -> {:.4g}, {:.4g} -> {:.4g}\n".format(
                                self.ptiter,
                                i,
                                j,
                                self.replica_states[i].dx,
                                self.replica_states[j].dx,
                                self.replica_states[i].bias_params,
                                self.replica_states[j].bias_params,
                            )
                        )
        else:
            raise NotImplementedError

    def _neighbor_exchange(self, exchange_pattern):
        """
        This function determines the exchange pattern using alternating swaps
        with the right and left neighbours.
        """
        for i in range(self.exchange_choice, self.nreplicas - 1, 2):
            dx1 = self.replica_states[i].dx
            dx2 = self.replica_states[i + 1].dx

            bias_params1 = self.replica_states[i].bias_params
            bias_params2 = self.replica_states[i + 1].bias_params

            if self.bias == "harmonic":
                # Hamiltonian replica exchange for pure springs
                k1 = bias_params1
                k2 = bias_params2
                deltaE = 0.5 * dx2 * dx2 - 0.5 * dx1 * dx1
                deltabeta = k2 - k1

                w = np.exp(deltaE * deltabeta)

            elif self.bias == "radial_gaussian":
                k1 = bias_params1[0]
                k2 = bias_params2[0]
                l1 = bias_params1[1]
                l2 = bias_params2[1]
                r_cutoff1 = bias_params1[2]
                r_cutoff2 = bias_params2[2]

                # Swap-MC-like Metropolis criterion
                Eold = self.energies_radial_gaussian(
                    dx1, k1, l1, r_cutoff1
                ) + self.energies_radial_gaussian(dx2, k2, l2, r_cutoff2)
                Enew = self.energies_radial_gaussian(
                    dx2, k1, l1, r_cutoff1
                ) + self.energies_radial_gaussian(dx1, k2, l2, r_cutoff2)

                w = np.exp(Eold - Enew)

            else:
                raise NotImplementedError

            rand = np.random.rand()
            if w > rand:
                # accept exchange

                # verify that we are not using the same rank twice for swaps
                assert exchange_pattern[i] == self.NO_EXCHANGE
                assert exchange_pattern[i + 1] == self.NO_EXCHANGE

                exchange_pattern[i] = i + 1
                exchange_pattern[i + 1] = i
                self.anyswap = True

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    self.ex_outstream.write(
                        "{}: Accepting exchange {:>2} <-> {:<2} ({:.4g} > {:.4g}): "
                        "{:.4g} <-> {:.4g}, {:.4g} <-> {:.4g}\n".format(
                            self.ptiter, i, i + 1, w, rand, dx1, dx2, bias_params1, bias_params2
                        )
                    )
        if self.exchange_choice == 0:
            self.exchange_choice = 1
        else:
            self.exchange_choice = 0

    def energies_radial_gaussian(self, dx, k, l0, r_cutoff):
        E = 0.5 * k * (dx - l0) ** 2
        if k != 0.0 and dx > r_cutoff:
            E += (self.nparticles * self.bdim - 1) * np.log(dx / r_cutoff)
        return E

    def _test_convergence(self):
        if self.test_convergence and self.ptiter > self.eq_min_ptiter:
            iteration = self.mcrunner_niter * (self.ptiter + 1)
            if iteration < self.mcrunner_eqsteps:
                logging.warning(
                    "Attempted to test convergence before the "
                    "mcrunner equilibration steps had terminated."
                )
                return self.max_ptiter
            else:
                return self._test_ts_convergence()
        else:
            return self.max_ptiter

    def _test_ts_convergence(self):
        """
        in_timeseries is the last segment of the time series
        self.timeseries is the whole recorded timeseries
        """
        if self.eq_time == 0:
            if self.fast_ct:
                iteration = self.mcrunner_niter * (self.ptiter + 1)
                self.eq_time = min([self.max_eq_time, iteration])
            else:
                self.eq_time = self._find_new_eq_time()
        # only keep time series from after the equilibration point, this references original data
        new_max_ptiter = self._find_new_max_ptiter()
        logging.info("new max_ptiter {}, current ptiter {}".format(new_max_ptiter, self.ptiter))
        return new_max_ptiter

    def _find_new_eq_time(self):
        iteration = self.mcrunner_niter * (self.ptiter + 1)
        logging.info("_find_new_eq_time, iteration: {}".format(iteration))
        new_eq_times = []
        for ireplica in range(self.nreplicas):
            eq_time = detect_equilibration_binary_search(
                np.array(self.replica_timeseries2[ireplica], dtype="d"),
                bs_nodes=self.bs_nodes,
            )[0]

            # this should avoid detecting artifacts near the end of the series
            eq_time = min([self.max_eq_time, eq_time])

            # guarantees that eq_time is larger than the mcrunner adapted number of steps
            new_eq_times.append(max([eq_time, self.mcrunner_eqsteps]))

        new_eq_time = max(new_eq_times)
        logging.debug(
            "New eq_times: {}, mcrunner_eqsteps: {}, len(timeseseries2): {}".format(
                new_eq_times,
                self.mcrunner_eqsteps,
                len(self.replica_timeseries2[0]),
            )
        )
        logging.info("New eq_time: {}".format(new_eq_time))
        return new_eq_time

    def _find_new_max_ptiter(self):
        """
        resets max_ptiter based on desired relative standard error that one wants to achieve. The longest estimate
        is chosen for the full pt. In order to estimate the number of extra steps to perform uses the correlated
        estimate for the standard error (see Troyer Am. J. Phys. 78 (2)) from which one can easily find that
        M = sig^2*(1+2t)/(mu rel_std_err)^2
        it returns an estimate of the new maxptiter only once the timeseries is longer than min_window
        """
        iteration = self.mcrunner_niter * (self.ptiter + 1)
        logging.info("_find_new_max_ptiter, iteration: {}".format(iteration))
        new_max_ptiters = []
        for ireplica in range(self.nreplicas):
            # to reduce nskip (use more points) make the factor by which len(timeseries) is divided by larger
            current_timeseries2 = self.replica_timeseries2[ireplica][self.eq_time :]
            nskip = max(int(np.round(len(current_timeseries2) / 1e6)), 1)
            tau = (
                integratedAutocorrelationTime_fft(
                    np.array(current_timeseries2[::nskip], dtype="d")
                )
                * nskip
            )
            var = np.var(current_timeseries2)
            mean = np.mean(current_timeseries2)
            sample_size = len(current_timeseries2)
            rel_err = np.sqrt(var * (1 + 2 * tau) / sample_size) / mean
            self.last_rel_std_errs[ireplica] = rel_err
            logging.info("Replica {} relative standard error: {}".format(ireplica, rel_err))
            logging.debug("Replica {} sample_size: {}".format(ireplica, sample_size))
            logging.debug("Replica {} autocorrelation time: {}".format(ireplica, tau))

            # compute by how much to extend the time series, if has at least 1e5
            if sample_size < self.min_window:
                new_max_ptiters.append(self.eq_max_ptiter)
            elif rel_err < self.rel_std_err:
                m = 0
                new_max_ptiters.append(self.ptiter)
            else:
                m = var * (1 + 2 * tau) / np.power(mean * self.rel_std_err, 2)
                new_max_ptiters.append(self.ptiter + int((m - sample_size) / self.mcrunner_niter))

        logging.debug("self.rel_std_err: %s" % self.rel_std_err)
        logging.debug("self.eq_max_ptiter: %s" % self.eq_max_ptiter)
        logging.debug("new_max_ptiters: %s" % new_max_ptiters)
        max_ptiter = max(new_max_ptiters)
        return min(max_ptiter, self.eq_max_ptiter)

    def _print_data(self):
        logging.debug("_print_data -- BEGIN")
        logging.debug("self.ptiter %s" % self.ptiter)
        logging.debug("self.eq_min_ptiter %s" % self.eq_min_ptiter)
        logging.debug("self.mcrunner_eqsteps %s" % self.mcrunner_eqsteps)
        iteration = self.mcrunner_niter * (self.ptiter + 1)
        for ireplica in range(self.nreplicas):
            if len(self.replica_timeseries[ireplica]) > 0:  # DO NOT WRITE EMPTY FILES!
                self._dump_timeseries(ireplica)
            if self.ptiter >= self.eq_min_ptiter and iteration > self.mcrunner_eqsteps:
                self._dump_histogram(ireplica)

        logging.debug("_print_data -- END")

    def _dump_timeseries(self, ireplica):
        directory = os.path.join(self.base_directory, str(ireplica))
        iteration = self.mcrunner_niter * (self.ptiter + 1)
        fname = os.path.join(directory, "TimeSeries.{}".format(iteration))
        np.savetxt(fname, self.replica_timeseries[ireplica])
        self.replica_timeseries[ireplica] = []  # Clear timeseries

    def _dump_traj(self):
        for ireplica in range(self.nreplicas):
            directory = os.path.join(self.base_directory, str(ireplica))
            iteration = self.mcrunner_niter * (self.ptiter + 1)
            coords = self.replica_states[ireplica].coords
            df = pd.DataFrame([coords], columns=[f'{i}' for i in range(len(coords))])
            df["iteration"] = iteration
            hdf5_path = os.path.join(directory, "trajectory.hd5")
            df.to_hdf(hdf5_path, key="data", mode="a", append=True)

    def _dump_histogram(self, ireplica):
        iteration = self.mcrunner_niter * (self.ptiter + 1)
        mean = np.mean(self.replica_timeseries2[ireplica][self.eq_time :])
        variance = np.var(self.replica_timeseries2[ireplica][self.eq_time :])
        std_err = self.last_rel_std_errs[ireplica] * mean
        self.histogram_mean_streams[ireplica].write(
            "{:<15}\t{:>15.15e}\t{:>15.15e}\t{:>15.15e}\n".format(
                iteration, mean, variance, std_err
            )
        )
        self.histogram_mean_streams[ireplica].flush()  # flush every time, so we don't loose data

    def _print_status(self):
        for ireplica in range(self.nreplicas):
            # Counters: 0: m_nitercount, 1: m_accept_count, 2: m_E_reject_count,
            #           3: m_conf_reject_count, 4: m_neval
            counters = self.replica_states[ireplica].counters

            status = {}
            status["iteration"] = counters[0]
            status["acc_frac"] = counters[1] / counters[0]
            status["E_reject_frac"] = counters[2] / counters[0]
            status["conf_reject_frac"] = counters[3] / counters[0]
            # Energy will be NaN at this point if the replica has been swapped,
            # since only the workers recalculate it.
            # Since this is a rather uncommon print, energy can be recomputed if it is nan here
            energy = self.replica_states[ireplica].energy
            if np.isnan(energy):
                energy = self.mcrunner_potential.getEnergy(self.replica_states[ireplica].coords)
            status["energy"] = energy
            status["neval"] = counters[4]

            nswaps = (
                self.replica_states[ireplica].swap_accepted_count
                + self.replica_states[ireplica].swap_rejected_count
            )
            if nswaps == 0:
                status["frac_acc_swaps"] = np.nan
            else:
                status["frac_acc_swaps"] = (
                    self.replica_states[ireplica].swap_accepted_count / nswaps
                )
            if self.ptiter == self.skip or self.print_status is False:
                self.status_streams[ireplica].write("#")
                for key, _ in list(status.items()):
                    self.status_streams[ireplica].write("{:<12}\t".format(key))
                self.status_streams[ireplica].write("\n")
            for _, value in list(status.items()):
                self.status_streams[ireplica].write("{:>12.3f}\t".format(value))
            self.status_streams[ireplica].write("\n")

    def _print_bias_params(self):
        fname = os.path.join(self.base_directory, "biases")
        with open(fname, "w") as kfile:
            for ireplica in range(self.nreplicas):
                param_row = self.replica_states[ireplica].bias_params
                if isinstance(param_row, float):
                    kfile.write(f"{param_row}")
                else:
                    for param in param_row:
                        kfile.write(f"{param} ")
                kfile.write(f"\n")

    def _print_stepsizes(self):
        fname = os.path.join(self.base_directory, "stepsizes")
        with open(fname, "w") as stepfile:
            for ireplica in range(self.nreplicas):
                stepfile.write("{:1.16f}\n".format(self.replica_states[ireplica].stepsize))

    def _print_parameters(self, ireplica):
        directory = os.path.join(self.base_directory, str(ireplica))
        fname = os.path.join(directory, "parameters")
        with open(fname, "w") as paramfile:
            paramfile.write("node:\t{0}\n".format(ireplica))
            paramfile.write("bias:\t{0}\n".format(self.replica_states[ireplica].bias_params))
            paramfile.write("PT iterations:\t{0}\n".format(self.max_ptiter))
            paramfile.write("total MC iterations:\t{0}\n".format(self.mcrunner_niter))

    def _print_permutations(self):
        if self.anyswap:
            iteration = self.mcrunner_niter * (self.ptiter + 1)
            f = self.permutations_stream
            f.write("{0}\t".format(iteration))
            for p in self.permutation_pattern:
                f.write("{0}\t".format(p))
            f.write("\n")
            f.flush()

    def _print_exchanges(self):
        logging.info("Number of exchanges:")
        exchange_header = "       "
        for i in range(self.nreplicas):
            exchange_header += "{:>6}".format(i)
        logging.info(exchange_header)
        for i in range(self.nreplicas):
            line = "{:>2} -> _".format(i)
            for j in range(self.nreplicas):
                line += "{:>6}".format(self.exchange_cnts[i, j])
            logging.info(line)

    def _flush_close_streams(self):
        self.ex_outstream.flush()
        self.ex_outstream.close()
        self.permutations_stream.flush()
        self.permutations_stream.close()
        for ireplica in range(self.nreplicas):
            self.histogram_mean_streams[ireplica].flush()
            self.histogram_mean_streams[ireplica].close()
            self.status_streams[ireplica].flush()
            self.status_streams[ireplica].close()

    def _validate_checkpoint_format(self):
        """Validate checkpoint format and check for required libraries"""
        valid_formats = ["auto", "pickle", "json"]
        if HAS_JOBLIB:
            valid_formats.append("joblib")
        if HAS_DILL:
            valid_formats.append("dill")
            
        if self.checkpoint_format not in valid_formats:
            raise ValueError(f"Invalid checkpoint_format '{self.checkpoint_format}'. Must be one of {valid_formats}")
        
        # Warn about missing libraries
        if self.checkpoint_format == "joblib" and not HAS_JOBLIB:
            raise ImportError("joblib is required for joblib checkpoint format. Install with: pip install joblib")
        if self.checkpoint_format == "dill" and not HAS_DILL:
            raise ImportError("dill is required for dill checkpoint format. Install with: pip install dill")
