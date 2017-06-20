from __future__ import division
import logging
import time
import copy
import os
import random
import time
import cPickle
import numpy as np
from mpi4py import MPI
from enum import Enum, unique  # Package enum34
from pymbar.timeseries import detectEquilibration_binary_search
from basinvolume.utils import trymakedir, integratedAutocorrelationTime_fft
from basinvolume.post_processing import spring_constants_variable_transform
from basinvolume.monte_carlo import IndependenceSampling
from basinvolume.spheres import BV_MCRunner_State


@unique
class ExchangeScheme(Enum):
    NEIGHBOR_EXCHANGE = 1
    INDEPENDENCE_SAMPLING = 2


class RunnerState(BV_MCRunner_State):
    """
    This class represents the state of a parallel tempering runner
    """
    def __init__(self, id, mcrunner_state):
        super(RunnerState, self).__init__(state=mcrunner_state)
        self.id = id
        self.dx = 0
        self.swap_accepted_count = 0
        self.swap_rejected_count = 0

    def set_mc_state(self, mcrunner_state):
        self._set_state(mcrunner_state)

    def serialize(self):
        data = np.empty(self.size(), dtype='d')
        data[0] = self.id
        data[1] = self.dx
        data[2] = self.energy
        data[3] = self.k
        data[4] = self.stepsize
        data[5] = self.takestep_count
        data[6 : 6+len(self.coords)] = self.coords
        data[6+len(self.coords) : 6+len(self.coords)+len(self.counters)] = self.counters
        data[6+len(self.coords)+len(self.counters) : ] = self.step_adaptation_counters
        return data

    def deserialize(self, value):
        self.id = int(value[0])
        self.dx = value[1]
        self.energy = value[2]
        self.k = value[3]
        self.stepsize = value[4]
        self.takestep_count = int(value[5])
        self.coords = value[6 : 6+len(self.coords)]
        self.counters = np.array(value[6+len(self.coords) : 6+len(self.coords)+len(self.counters)],
                                 dtype='uintp')
        self.step_adaptation_counters = np.array(value[6+len(self.coords)+len(self.counters) : ],
                                                 dtype='uintp')

    def size(self):
        return 6 + len(self.coords) + len(self.counters) + len(self.step_adaptation_counters)


class PT_Master(object):
    """
    This class manages a job queue that sends jobs to Parallel Tempering runners.
    It is run in a parallel process on rank 0.
    """

    def __init__(self, nrunners, example_mcrunner, kmax, kmin, u2meank0,
                 max_ptiter=10, pfreq=1, skip=0, test_convergence=True,
                 fast_ct=False, rel_std_err=0.03, min_window=2.5e5,
                 max_eq_time=2.5e5, numnegk=0, lownegk=-2.5, print_status=False,
                 base_directory=None, bs_nodes=100, eq_min_ptiter=None,
                 eq_max_ptiter=None, sleep_seconds=0.0001, exchange_scheme=ExchangeScheme.NEIGHBOR_EXCHANGE,
                 checkpoint_time=None, checkpoint_file='checkpoint.dmp'):
        self.nrunners = nrunners
        self.sleep_seconds = sleep_seconds
        self.comm = MPI.COMM_WORLD
        self.nworkers = self.comm.Get_size() - 1 # total number of workers
        self.rank = self.comm.Get_rank() # this is the unique identifier for the process
        self.nparticles = example_mcrunner.nparticles
        self.bdim = example_mcrunner.bdim
        self.kmax = kmax
        self.kmin = kmin
        self.max_ptiter = max_ptiter
        self.ptiter = 0
        self.print_status = print_status
        self.skip = skip  # might want to skip the first few swaps to allow for equilibration
        self.pfreq = pfreq
        self.NO_EXCHANGE = -12345  # this NEGATIVE number in exchange pattern means that no exchange should be attempted
        if base_directory is None:
            self.base_directory = os.path.join(os.getcwd(), 'ptmc_results')
        else:
            self.base_directory = base_directory
        self.exchange_choice = random.randint(0, 1)
        self.anyswap = False  # set to true if any swap happened
        self.permutation_pattern = np.zeros(self.nrunners, dtype='int32')  # useful for printing exchange permutations
        self.u2meank0 = u2meank0
        self.mcrunner_niter = example_mcrunner.niter
        self.mcrunner_eqsteps = int(example_mcrunner.equilibration_steps)
        self.test_convergence = test_convergence
        self.eq_time = 0  # time at which equilibration was reached
        self.fast_ct = fast_ct
        self.rel_std_err = rel_std_err  # relative standard error
        self.last_rel_std_errs = np.zeros(self.nrunners)  # last measured relative standard errors for each runner
        if eq_min_ptiter is None:
            eq_min_ptiter = int(self.max_ptiter*0.95)
        self.eq_min_ptiter = int(eq_min_ptiter)
        if eq_max_ptiter is None:
            eq_max_ptiter = int(2e6/example_mcrunner.niter)
        self.eq_max_ptiter = int(eq_max_ptiter)
        self.min_window = int(min_window)
        self.max_eq_time = int(max_eq_time)
        self.bs_nodes = int(bs_nodes)
        self.numnegk = int(numnegk)
        self.lownegk = int(lownegk)
        self._init_runners(example_mcrunner)
        self._init_timeseries()
        self._init_print()
        self.recv_buffer = np.empty(self.runner_states[0].size() + self.mcrunner_niter, dtype='d')
        self.exchange_cnts = np.zeros((self.nrunners, self.nrunners), dtype='int32')
        self.exchange_scheme = exchange_scheme
        self._init_sampling()
        self.checkpoint_time = checkpoint_time
        self.checkpoint_file = checkpoint_file
        assert(self.nrunners > self.nworkers)
        assert(self.eq_min_ptiter > self.skip)
        assert(self.max_ptiter > self.eq_min_ptiter)
        assert(self.eq_max_ptiter > self.eq_min_ptiter)
        assert((self.eq_max_ptiter-self.eq_min_ptiter) * self.mcrunner_niter > self.min_window)  # Condition on the minimal window size
        if not (self.min_window > self.mcrunner_eqsteps):
            logging.info("self.min_window: {}".format(self.min_window))
            logging.info("self.mcrunner_eqsteps: {}".format(self.mcrunner_eqsteps))
        assert(self.min_window > self.mcrunner_eqsteps)
        assert(self.max_eq_time > self.mcrunner_eqsteps)

    def init_state(self):
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

    def _init_runners(self, example_mcrunner):
        ks = self._get_ks()
        start_state = example_mcrunner.get_complete_state()
        self.runner_states = []
        for i in xrange(self.nrunners):
            self.runner_states.append(RunnerState(i, start_state))
            self.runner_states[-1].k = ks[i]

    def _init_timeseries(self):
        self.runner_timeseries = [[] for _ in xrange(self.nrunners)]
        self.runner_timeseries2 = [[] for _ in xrange(self.nrunners)]

    def _init_print(self, append=False):
        if append:
            mode = 'a'
        else:
            mode = 'w'
            trymakedir(self.base_directory)
            self._print_ks()
        self.ex_outstream = open(os.path.join(self.base_directory, 'exchanges'), mode)
        self.permutations_stream = open(os.path.join(self.base_directory, 'rem_permutations'), mode)
        self.status_streams = []
        self.histogram_mean_streams = []
        for irunner in xrange(self.nrunners):
            directory = os.path.join(self.base_directory, str(irunner))
            if not append:
                trymakedir(directory)
                self._print_parameters(irunner)
            self.status_streams.append(open(os.path.join(directory, 'status'), mode))
            self.histogram_mean_streams.append(open(os.path.join(directory, 'hist_mean'), mode))
            if not append:
                self.histogram_mean_streams[irunner].write(
                    '{:<15}\t{:<15}\t{:<15}\t{:<15}\n'
                    .format('iteration','<(x-x0)**2>','variance','std_err'))

    def _get_ks(self):
        """
        set up the spring constants (temperatures) by distributing them exponentially.
        We order the spring constants from highest to lowest, to calculate the
        more costly runners first.
        """
        nposk = self.nrunners - self.numnegk  # number of positive k
        Karray = spring_constants_variable_transform(nposk+1, self.kmax, self.u2meank0,
                                                     self.nparticles, self.bdim, self.kmin)
        Karray = Karray[:-1]  # exclude kmax entry, no need to be simulated, mean is already available
        if self.numnegk > 0:
            assert np.abs(self.lownegk) > 0
            grid = -(np.abs(self.lownegk) + 1 -
                     np.exp(np.linspace(np.log(1), np.log(np.abs(self.lownegk)+1),
                                        self.numnegk+1))
                     )[:-1]
            assert grid == self.numnegk
            for x in grid[::-1]:
                Karray.insert(0, x)
        # Reverse Karray for backwards compatibility
        Karray = Karray[::-1]
        return Karray

    def run(self):
        if self.checkpoint_time is not None:
            start_time = time.time()
        self.created_checkpoint = False
        while (self.ptiter < self.max_ptiter
               and not self.created_checkpoint):
            logging.debug("Iteration {}".format(self.ptiter))
            self._one_iteration()
            if self.ptiter >= self.max_ptiter:
                self.max_ptiter = self._test_convergence()
            if (self.checkpoint_time is not None
                and time.time() - start_time > self.checkpoint_time):
                self.created_checkpoint = True

        # Stop workers
        for iworker in xrange(self.nworkers):
            self.comm.Send(np.array([-1], dtype='d'), dest=iworker+1)

        if self.created_checkpoint:
            self._create_checkpoint()
            logging.info("Created checkpoint")
        else:
            self._print_data()
            if self.print_status:
                self._print_status()
            self._print_exchanges()
            self._flush_close_streams()
            logging.info("Master finished")

    def _create_checkpoint(self):
        del self.comm
        del self._calculate_exchange
        del self.indep_sampling
        self._flush_close_streams()
        del self.ex_outstream
        del self.permutations_stream
        del self.histogram_mean_streams
        del self.status_streams
        checkpoint_path = os.path.join(self.base_directory, self.checkpoint_file)
        with open(checkpoint_path, 'wb') as outfile:
            cPickle.dump(self, outfile)

    def _one_iteration(self):
        """Perform one parallel tempering iteration

        Each PT iteration consists of the following steps:

        * distribute work, i.e. run the MCrunners for a predefined number of steps
        * collect the results
        * attempt an exchange
        """
        # Start with slowest runner (lowest k)
        current_runner = self.nrunners - 1

        # Send the first job to every worker
        for i in xrange(self.nworkers):
            self.comm.Send(self.runner_states[current_runner].serialize(), dest=i+1)
            current_runner -= 1

        # Send the remaining jobs to finished workers
        while current_runner >= 0:
            if self.sleep_seconds > 0:
                while not self.comm.Iprobe(source=MPI.ANY_SOURCE):
                    time.sleep(self.sleep_seconds)
            finished_worker = self._receive_result()
            self.comm.Send(self.runner_states[current_runner].serialize(), dest=finished_worker)
            current_runner -= 1

        # Wait for all workers to finish
        for _ in xrange(self.nworkers):
            if self.sleep_seconds > 0:
                while not self.comm.Iprobe(source=MPI.ANY_SOURCE):
                    time.sleep(self.sleep_seconds)
            self._receive_result()

        if self.ptiter >= self.skip:
            self._exchange_coords()
            # print and increase parallel tempering count and test convergence
            if (self.ptiter % self.pfreq == 0):
                self.max_ptiter = self._test_convergence()
                self._print_data()
            if self.print_status:
                self._print_status()
        self.ptiter += 1

    def _receive_result(self):
        status = MPI.Status()
        self.comm.Recv(self.recv_buffer, source=MPI.ANY_SOURCE, status=status)
        src = status.Get_source()
        runner_id = int(self.recv_buffer[0])
        state_size = self.runner_states[0].size()
        self.runner_states[runner_id].deserialize(self.recv_buffer[:state_size].copy())
        recv_timeseries = self.recv_buffer[state_size:]
        self.runner_timeseries[runner_id].extend(recv_timeseries)
        recv_timeseries2 = np.power(recv_timeseries, 2)
        self.runner_timeseries2[runner_id].extend(recv_timeseries2)
        return src

    def _exchange_coords(self):
        """
        Exchange the runner states according to _find_exchange_buddies
        """
        # dx_string = "dx: "
        # for i in xrange(self.nrunners):
        #     dx_string += str(self.runner_states[i].dx) + ", "
        # logging.debug(dx_string)

        # find exchange pattern (list of exchange buddies)
        exchange_pattern = self._find_exchange_buddies()

        # swap runner coordinates and dx
        old_coords = [runner.coords for runner in self.runner_states]
        old_dxs = [runner.dx for runner in self.runner_states]
        for istate, ibuddy in enumerate(exchange_pattern):
            if ibuddy != self.NO_EXCHANGE:
                # Swap coordinates and dx
                self.runner_states[istate].coords = old_coords[ibuddy]
                self.runner_states[istate].dx = old_dxs[ibuddy]

                # Set energy to NaN, since it needs to be recalculated
                self.runner_states[istate].energy = np.nan

    def _find_exchange_buddies(self):
        """
        This function determines the exchange pattern using alternating swaps
        with the right and left neighbours.
        An exchange pattern array is constructed, filled with self.NO_EXCHANGE
        which signifies that no exchange should be attempted. If the swap attempt
        is successful this value is replaced with the number of the runner with
        which to perform the swap.
        """
        exchange_pattern = np.empty(self.nrunners, dtype='int32')
        exchange_pattern.fill(self.NO_EXCHANGE) # reset exchange pattern to no exchange
        self.anyswap = False

        self._calculate_exchange(exchange_pattern)

        for i in xrange(self.nrunners):
            if exchange_pattern[i] == i:
                exchange_pattern[i] = self.NO_EXCHANGE

        # record self.permutation_pattern to print permutations in print function
        if self.anyswap:
            for i, buddy in enumerate(exchange_pattern):
                if (buddy != self.NO_EXCHANGE):
                    self.runner_states[i].swap_accepted_count += 1
                    self.exchange_cnts[i, buddy] += 1
                    self.permutation_pattern[i] = buddy + 1  # to conform to fortran notation
                else:
                    self.runner_states[i].swap_rejected_count += 1
                    self.permutation_pattern[i] = i + 1  # to conform to fortran notation
            self._print_permutations()

        return exchange_pattern

    def _independence_sampling(self, exchange_pattern):
        dxs = np.array([runner.dx for runner in self.runner_states])
        betas = np.array([runner.k for runner in self.runner_states])

        # According to Chodera & Shirts 2011 nrunners**3 to nrunners**5 exchanges
        # should be sufficient
        nexchanges = self.nrunners ** 3

        naccept = self.indep_sampling.exchange(exchange_pattern, dxs, betas, nexchanges)

        if naccept > 0:
            self.anyswap = True

        logging.debug("Acceptance ratio: %f" % (naccept / nexchanges))
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            for i in xrange(self.nrunners):
                j = exchange_pattern[i]
                self.ex_outstream.write(
                    "{}: Accepting exchange {:>2} -> {:<2}: "
                    "{:.4g} -> {:.4g}, {:.4g} -> {:.4g}\n".format(
                        self.ptiter, i, j,
                        self.runner_states[i].dx, self.runner_states[j].dx,
                        self.runner_states[i].k, self.runner_states[j].k))

    def _neighbor_exchange(self, exchange_pattern):
        for i in xrange(self.exchange_choice, self.nrunners-1, 2):
            dx1 = self.runner_states[i].dx
            k1 = self.runner_states[i].k
            dx2 = self.runner_states[i + 1].dx
            k2 = self.runner_states[i + 1].k

            # Hamiltonian replica exchange
            deltaE = 0.5*dx2*dx2 - 0.5*dx1*dx1
            deltabeta = k2 - k1
            w = np.exp(deltaE * deltabeta)

            rand = np.random.rand()
            if w > rand:
                # accept exchange

                # verify that we are not using the same rank twice for swaps
                assert(exchange_pattern[i] == self.NO_EXCHANGE)
                assert(exchange_pattern[i + 1] == self.NO_EXCHANGE)

                exchange_pattern[i] = i + 1
                exchange_pattern[i + 1] = i
                self.anyswap = True

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    self.ex_outstream.write(
                        "{}: Accepting exchange {:>2} <-> {:<2} ({:.4g} > {:.4g}): "
                        "{:.4g} <-> {:.4g}, {:.4g} <-> {:.4g}\n".format(
                            self.ptiter, i, i + 1, w, rand, dx1, dx2, k1, k2))
        if self.exchange_choice == 0:
            self.exchange_choice = 1
        else:
            self.exchange_choice = 0

    def _test_convergence(self):
        if self.test_convergence and self.ptiter > self.eq_min_ptiter:
            iteration = self.mcrunner_niter * (self.ptiter+1)
            if iteration < self.mcrunner_eqsteps:
                logging.warning("Attempted to test convergence before the "
                                "mcrunner equilibration steps had terminated.")
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
                iteration = self.mcrunner_niter * (self.ptiter+1)
                self.eq_time = min([self.max_eq_time, iteration])
            else:
                self.eq_time = self._find_new_eq_time()
        # only keep time series from after the equilibration point, this references original data
        new_max_ptiter = self._find_new_max_ptiter()
        logging.info("new max_ptiter {}, current ptiter {}".format(new_max_ptiter, self.ptiter))
        return new_max_ptiter

    def _find_new_eq_time(self):
        iteration = self.mcrunner_niter * (self.ptiter+1)
        logging.info("_find_new_eq_time, iteration: {}".format(iteration))
        new_eq_times = []
        for irunner in xrange(self.nrunners):
            eq_time = detectEquilibration_binary_search(
                np.array(self.runner_timeseries2[irunner], dtype='d'), bs_nodes=self.bs_nodes)[0]

            # this should avoid detecting artifacts near the end of the series
            eq_time = min([self.max_eq_time, eq_time])

            # guarantees that eq_time is larger than the mcrunner adapted number of steps
            new_eq_times.append(max([eq_time, self.mcrunner_eqsteps]))

        new_eq_time = max(new_eq_times)
        logging.debug("New eq_times: {}, mcrunner_eqsteps: {}, len(timeseseries2): {}"
                      .format(new_eq_times, self.mcrunner_eqsteps,
                              len(self.runner_timeseries2[0])))
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
        iteration = self.mcrunner_niter * (self.ptiter+1)
        logging.info("_find_new_max_ptiter, iteration: {}".format(iteration))
        new_max_ptiters = []
        for irunner in xrange(self.nrunners):
            # to reduce nskip (use more points) make the factor by which len(timeseries) is divided by larger
            current_timeseries2 = self.runner_timeseries2[irunner][self.eq_time:]
            nskip = max(int(np.round(len(current_timeseries2)/1e6)),1)
            tau = (integratedAutocorrelationTime_fft(np.array(current_timeseries2[::nskip],
                                                              dtype='d'))
                   * nskip)
            var = np.var(current_timeseries2)
            mean = np.mean(current_timeseries2)
            sample_size = len(current_timeseries2)
            rel_err = np.sqrt(var*(1+2*tau)/sample_size) / mean
            self.last_rel_std_errs[irunner] = rel_err
            logging.info("Runner {} relative standard error: {}".format(irunner, rel_err))
            logging.debug("Runner {} sample_size: {}".format(irunner, sample_size))
            logging.debug("Runner {} autocorrelation time: {}".format(irunner, tau))

            #compute by how much to extend the time series, if has at least 1e5
            if sample_size < self.min_window:
                new_max_ptiters.append(self.eq_max_ptiter)
            elif rel_err < self.rel_std_err:
                m = 0
                new_max_ptiters.append(self.ptiter)
            else:
                m = var * (1+2*tau) / np.power(mean * self.rel_std_err, 2)
                new_max_ptiters.append(self.ptiter + int((m-sample_size)/self.mcrunner_niter))

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
        iteration = self.mcrunner_niter * (self.ptiter+1)
        for irunner in xrange(self.nrunners):
            self._dump_timeseries(irunner)
            if self.ptiter >= self.eq_min_ptiter and iteration > self.mcrunner_eqsteps:
                self._dump_histogram(irunner)
        logging.debug("_print_data -- END")

    def _dump_timeseries(self, irunner):
        directory = os.path.join(self.base_directory, str(irunner))
        iteration = self.mcrunner_niter * (self.ptiter+1)
        fname = os.path.join(directory, 'TimeSeries.{}'.format(iteration))
        np.savetxt(fname, self.runner_timeseries[irunner])
        self.runner_timeseries[irunner] = []  # Clear timeseries

    def _dump_histogram(self, irunner):
        directory = os.path.join(self.base_directory, str(irunner))
        iteration = self.mcrunner_niter * (self.ptiter+1)
        fname = os.path.join(directory, 'Visits.his.{}'.format(iteration))
        mean = np.mean(self.runner_timeseries2[irunner][self.eq_time:])
        variance = np.var(self.runner_timeseries2[irunner][self.eq_time:])
        std_err = self.last_rel_std_errs[irunner] * mean
        self.histogram_mean_streams[irunner].write(
            '{:<15}\t{:>15.15e}\t{:>15.15e}\t{:>15.15e}\n'.format(
                iteration, mean, variance, std_err))
        self.histogram_mean_streams[irunner].flush()  # flush every time, so we don't loose data

    def _print_status(self):
        for irunner in xrange(self.nrunners):
            # Counters: 0: m_nitercount, 1: m_accept_count, 2: m_E_reject_count,
            #           3: m_conf_reject_count, 4: m_neval
            counters = self.runner_states[irunner].counters

            status = {}
            status['iteration'] = counters[0]
            status['acc_frac'] = counters[1] / counters[0]
            status['E_reject_frac'] = counters[2] / counters[0]
            status['conf_reject_frac'] = counters[3] / counters[0]
            # Energy will be NaN at this point if the runner has been swapped,
            # since only the workers recalculate it.
            status['energy'] = self.runner_states[irunner].energy
            status['neval'] = counters[4]

            status['frac_acc_swaps'] = (self.runner_states[irunner].swap_accepted_count /
                                        (self.runner_states[irunner].swap_accepted_count
                                         + self.runner_states[irunner].swap_rejected_count))
            if self.ptiter == self.skip:
                self.status_streams[irunner].write('#')
                for key, _ in status.iteritems():
                    self.status_streams[irunner].write('{:<12}\t'.format(key))
                self.status_streams[irunner].write('\n')
            for _, value in status.iteritems():
                self.status_streams[irunner].write('{:>12.3f}\t'.format(value))
            self.status_streams[irunner].write('\n')

    def _print_ks(self):
        fname = os.path.join(self.base_directory, 'temperatures')
        with open(fname, 'w') as kfile:
            for irunner in xrange(self.nrunners):
                kfile.write('{:1.16f}\n'.format(self.runner_states[irunner].k))

    def _print_parameters(self, irunner):
        directory = os.path.join(self.base_directory, str(irunner))
        fname = os.path.join(directory, 'parameters')
        with open(fname, 'w') as paramfile:
            paramfile.write('node:\t{0}\n'.format(irunner))
            paramfile.write('temperature:\t{0}\n'.format(self.runner_states[irunner].k))
            paramfile.write('PT iterations:\t{0}\n'.format(self.max_ptiter))
            paramfile.write('total MC iterations:\t{0}\n'.format(self.mcrunner_niter))

    def _print_permutations(self):
        if self.anyswap:
            iteration = self.mcrunner_niter * (self.ptiter+1)
            f = self.permutations_stream
            f.write('{0}\t'.format(iteration))
            for p in self.permutation_pattern:
                f.write('{0}\t'.format(p))
            f.write('\n')
            f.flush()

    def _print_exchanges(self):
        logging.info("Number of exchanges:")
        exchange_header = "        "
        for i in xrange(self.nrunners):
            exchange_header += "{:>6}".format(i)
        logging.info(exchange_header)
        for i in range(self.nrunners):
            line = "{:>2} <-> _".format(i)
            for j in range(self.nrunners):
                line += "{:>6}".format(self.exchange_cnts[i, j])
            logging.info(line)

    def _flush_close_streams(self):
        self.ex_outstream.flush()
        self.ex_outstream.close()
        self.permutations_stream.flush()
        self.permutations_stream.close()
        for irunner in xrange(self.nrunners):
            self.histogram_mean_streams[irunner].flush()
            self.histogram_mean_streams[irunner].close()
            self.status_streams[irunner].flush()
            self.status_streams[irunner].close()
