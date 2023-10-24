from __future__ import division
from __future__ import print_function

import numpy as np
import time

from pymbar.timeseries import detect_equilibration_binary_search

from basinvolume.geometry.point_sampling import HyperElemMCrunner
from basinvolume.geometry.cloud_sampling import HyperElemOracleMCrunner
from basinvolume.post_processing import spring_constants_variable_transform
from basinvolume.spheres import BV_MCrunner
from basinvolume.utils import get_dist_com, integratedAutocorrelationTime_fft
from mcpele.parallel_tempering import MPI_PT_RLhandshake, trymakedir

try:
    from basinvolume.gaussian_benchmark import GaussianBenchmarkKminRun
except:
    print("gaussian import failed")


class MPI_BV_PT_RLhandshake_cloud(MPI_PT_RLhandshake):
    """
    Note that ptiter is a single REM step, for each ptiter there are self.mcrunner.niter Monte Carlo steps

    u2meank0 : float
        mean of histogram from simulation done at k=0
    Tmax, Tmin : float
        here correspond to kmin and kmax, they should be computed by bv_find_params
    fast_ct: bool
        if false perform full convergence test computing the segment of the recorded time series that maximises the number of uncorrelated samples
        else return the maximum equilibration time
    numnegk : int
        number of negative k's, by default 0
    lownegk : float
        lowest negative k
    skip: int
        number of pt iteration where swaps should be skipped. For instance while the stepsize is adjusted, pt swaps shuold be avoided
    max_ptiter: int
        inherited max_ptiter, in this class it plays as the minimum number of pt_iter. In other words it's eq_min_ptiter
    eq_min_ptiter: int
        95% of max_ptiter, this is the minimum length for which the calculation will run
    eq_max_ptiter: int
        maximum number of pt iterations that the class will permorm, it will literally abort past this number ot ptiter
    min_window: int
        minimal sample size of array to measure correlation length. So once the equilibration point is computed the next
        convergence test is not performed until we have enough data to fill the min_window
    max_eq_time: int
        when computing the equilibration time we choose the minimum value between max_eq_time and the computed one
    bs_nodes : int
        number of binary search nodes to use when computing the equilibration point
    """

    def __init__(
        self,
        mcrunner,
        Tmax,
        Tmin,
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
        print_status=False,
        base_directory=None,
        swap=True,
        verbose=False,
        bs_nodes=100,
        eq_min_ptiter=None,
        eq_max_ptiter=None,
        fix_com=True,
    ):
        super(MPI_BV_PT_RLhandshake_cloud, self).__init__(
            mcrunner,
            Tmax,
            Tmin,
            max_ptiter=max_ptiter,
            pfreq=pfreq,
            skip=skip,
            print_status=print_status,
            base_directory=base_directory,
            swap=swap,
            verbose=verbose,
        )
        self.u2meank0 = u2meank0
        self.mcrunner_eqsteps = mcrunner.equilibration_steps
        self.test_convergence = test_convergence
        self.autocorr = []
        self.timeseries2 = np.array([])
        self.eq_time = 0  # time at which equilibration was reached
        self.fast_ct = fast_ct
        self.rel_std_err = rel_std_err  # relative standard error
        self.rel_std_err_arr = []  # array of measured relative standard errors
        self.cloud_frac_pos = 0  # fraction of values > 0
        self.test_convergence_count = (
            0  # count how many times test for convergence
        )
        if eq_min_ptiter is None:
            eq_min_ptiter = int(
                self.max_ptiter * 0.95
            )  # initial maxptiter is passed from command line #int(1e5/self.mcrunner.niter)#
        self.eq_min_ptiter = int(eq_min_ptiter)
        if eq_max_ptiter is None:
            eq_max_ptiter = int(2e6 / self.mcrunner.niter)
        self.eq_max_ptiter = int(eq_max_ptiter)
        self.min_window = int(min_window)
        self.max_eq_time = int(max_eq_time)
        self.bs_nodes = int(bs_nodes)
        self.numnegk = int(numnegk)
        self.lownegk = int(lownegk)
        self.fix_com = fix_com
        assert self.eq_min_ptiter > self.skip
        assert self.max_ptiter > self.eq_min_ptiter
        assert self.eq_max_ptiter > self.eq_min_ptiter
        assert (
            self.eq_max_ptiter - self.eq_min_ptiter
        ) * self.mcrunner.niter > self.min_window  # condition on the minimal window size
        if not (self.min_window > self.mcrunner_eqsteps):
            print(("self.min_window", self.min_window))
            print(("self.mcrunner_eqsteps", self.mcrunner_eqsteps))
        assert self.min_window > self.mcrunner_eqsteps
        assert self.max_eq_time > self.mcrunner_eqsteps

    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        directory = "{0}/{1}".format(base_directory, self.rank)
        trymakedir(directory)
        self._master_print_temperatures()
        self._all_print_parameters()
        self.status_stream = open("{0}/{1}".format(directory, "status"), "w")
        self.histogram_mean_stream = open(
            "{0}/{1}".format(directory, "hist_mean"), "w"
        )
        self.histogram_mean_stream.write(
            "{:<15}\t{:<15}\t{:<15}\t{:<15}\n".format(
                "iteration", "<(x-x0)**2>", "variance", "std_err"
            )
        )
        if self.rank == 0:
            self.permutations_stream = open(
                r"{0}/rem_permutations".format(base_directory), "w"
            )

    def _print_data(self):
        self._all_dump_timeseries()  # convergence is tested in this function
        # the histogram depends on self.timeseries that is not empty only once the ts test is passed
        print("_print_data -- BEGIN")
        print(("self.ptiter", self.ptiter))
        print(("self.eq_min_ptiter", self.eq_min_ptiter))
        print(("self.timeseries2.size", self.timeseries2.size))
        print(("self.mcrunner_eqsteps", self.mcrunner_eqsteps))
        if (
            self.ptiter >= self.eq_min_ptiter
            and self.timeseries2.size > self.mcrunner_eqsteps
        ):
            self._all_dump_histogram()
        print("_print_data -- END")

    def _test_convergence(self):
        tail_timeseries = self.mcrunner.get_timeseries()
        self.ndrops = tail_timeseries.shape[1]
        tail_timeseries_all = tail_timeseries.flatten()
        tail_timeseries = tail_timeseries_all[
            tail_timeseries_all > 0
        ]  # onlykeep positive
        self.test_convergence_count += 1
        self.cloud_frac_pos = (
            self.cloud_frac_pos
            + (
                tail_timeseries.size / tail_timeseries_all.size
                - self.cloud_frac_pos
            )
            / self.test_convergence_count
        )
        assert self.cloud_frac_pos <= 1
        tail_timeseries = tail_timeseries[
            :: max(1, int(self.ndrops * self.cloud_frac_pos))
        ]  # subsample here (keeps in check size of array)
        tail_timeseries2 = np.power(tail_timeseries, 2)
        self.timeseries2 = np.append(self.timeseries2, tail_timeseries2)
        assert self.timeseries2.shape == (
            self.timeseries2.size,
        ), "timeseries2 shape is not flat"
        if self.test_convergence and self.ptiter > self.eq_min_ptiter:
            if self.timeseries2.size < self.mcrunner_eqsteps:
                print(
                    "core {} attempted to test convergence before the mcrunner equilibration steps had terminated".format(
                        self.rank
                    )
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
            start = time.time()
            if self.fast_ct:
                eq_time = np.amin([self.max_eq_time, self.timeseries2.size])
                self.eq_time = self.eq_time
            else:
                print("detecting equilibration point")
                print("timeseries size", self.timeseries2.size)
                # self.timeseries2[::self.ndrops] subsampling now done in test convergence function one level up
                eq_time = detect_equilibration_binary_search(
                    self.timeseries2, bs_nodes=self.bs_nodes
                )[0]
                eq_time = int(
                    np.amin([self.timeseries2.size - 1, int(eq_time)])
                )
                eq_time = np.amin(
                    [self.timeseries2.size // 1.5, eq_time]
                )  # this should avoid detecting artifacts near the end of the series
                new_eq_time = int(
                    eq_time / self.cloud_frac_pos
                )  # this makes the eq_times comparable
                new_eq_time = np.amax(
                    [new_eq_time, self.mcrunner_eqsteps]
                )  # guarantees that eq_time is larger than the mcrunner adapted number of steps
                # gather values, find largest, then broadcast it
                new_eq_time_array = self._gather_data([new_eq_time])
                if self.rank == 0:
                    new_eq_time = np.amax(new_eq_time_array)
                else:
                    new_eq_time = None
                self.eq_time = self._broadcast_data([new_eq_time], 1)[0]
                self.eq_time *= (
                    self.cloud_frac_pos
                )  # adjust eq_time to system cloud_frac_pos
                self.eq_time = np.amin(
                    [self.timeseries2.size // 1.5, self.eq_time]
                )
                self.eq_time = int(self.eq_time)
            end = time.time()
            print(
                (
                    "core {} set_eq_time: {} comp_eq_time: {} "
                    "mcrunner_eqsteps: {} len(timeseseries2): {} "
                    "time detect equilibration: {}"
                ).format(
                    self.rank,
                    self.eq_time,
                    eq_time,
                    self.mcrunner_eqsteps,
                    self.timeseries2.size,
                    end - start,
                )
            )
        # only keep time series from after the equilibration point, this references original data
        timeseries2 = self.timeseries2[self.eq_time :]
        new_max_ptiter = self._find_new_max_ptiter(timeseries2)
        # if self.verbose:
        print(
            "new max_ptiter {}, current ptiter {}".format(
                new_max_ptiter, self.ptiter
            )
        )
        print(
            "core {} autocorrelation time {}".format(self.rank, self.autocorr)
        )
        return new_max_ptiter

    def _find_new_max_ptiter(self, timeseries2):
        """
        resets max_ptiter based on desired relative standard error that one wants to achieve. The longest estimate
        is chosen for the full pt. In order to estimate the number of extra steps to perform uses the correlated
        estimate for the standard error (see Troyer Am. J. Phys. 78 (2)) from which one can easily find that
        M = sig^2*(1+2t)/(mu rel_std_err)^2
        it returns an estimate of the new maxptiter only once the timeseries is longer than min_window
        """
        # to reduce nskip (use more points) make the factor by which timeseries.size is divided by larger
        maxsize = int(1e6)
        nskip = max(int(np.round(timeseries2.size / maxsize)), 1)
        print(
            "core {}: timeseries2.size {}, maxsizze {}, nskip {}".format(
                self.rank, timeseries2.size, maxsize, nskip
            )
        )
        tau = integratedAutocorrelationTime_fft(timeseries2[::nskip]) * nskip
        assert np.isfinite(tau)
        self.autocorr.extend([tau])
        var = np.var(timeseries2)
        mean = np.mean(timeseries2)
        sample_size = timeseries2.size
        rel_err = np.sqrt(var * (1 + 2 * tau) / sample_size) / mean
        self.rel_std_err_arr.extend([rel_err])
        print("core {} relative standard error {}".format(self.rank, rel_err))

        # compute by how much to extend the time series, if has at least 1e5
        if sample_size < self.min_window:  # self.autocorr[-1]*100
            new_max_ptiter = self.eq_max_ptiter
        elif rel_err < self.rel_std_err:
            m = 0
            new_max_ptiter = self.ptiter
        else:
            m = var * (1 + 2 * tau) / np.power(mean * self.rel_std_err, 2)
            new_max_ptiter = self.ptiter + int(
                (m - sample_size) / self.mcrunner.niter
            )

        new_max_ptiter_array = self._gather_data([new_max_ptiter])
        if self.rank == 0:
            max_ptiter = np.amax(new_max_ptiter_array)
        else:
            max_ptiter = None

        max_ptiter = self._broadcast_data([max_ptiter], 1)[0]
        return min(int(max_ptiter), self.eq_max_ptiter)

    def _all_dump_timeseries(self):
        """for this to work the directory must have been initialised in _print_initialise"""
        base_directory = self.base_directory
        directory = "{0}/{1}".format(base_directory, self.rank)
        iteration = self.mcrunner.get_iterations_count()
        fname = "{0}/TimeSeries.{1}".format(directory, int(iteration))
        self.mcrunner.dump_timeseries(fname, clear=True)

    def _all_dump_histogram(self):
        """for this to work the directory must have been initialised in _print_initialise"""
        base_directory = self.base_directory
        directory = "{0}/{1}".format(base_directory, self.rank)
        iteration = self.mcrunner.get_iterations_count()
        fname = "{0}/Visits.his.{1}".format(directory, float(iteration))
        if not self.suppress_histogram:
            mean, variance = self.mcrunner.dump_histogram(fname)
            self.histogram_mean_stream.write(
                "{:<15}\t{:>15.15e}\t{:>15.15e}\n".format(
                    iteration, mean, variance
                )
            )
        else:
            # mean = np.mean(self.timeseries2[self.eq_time:])
            # variance = np.var(self.timeseries2[self.eq_time:])
            mean, variance = self.mcrunner.get_displ2_kmin()  # debug
            std_err = self.rel_std_err_arr[-1] * mean
            self.histogram_mean_stream.write(
                "{:<15}\t{:>15.15e}\t{:>15.15e}\t{:>15.15e}\n".format(
                    iteration, mean, variance, std_err
                )
            )
        self.histogram_mean_stream.flush()  # print every time not to lose data

    #    def _get_temps(self):
    #        """
    #        NOTE: BECAUSE K0 IS INCLUDED IN THE CALCULATION TARRAY CANNOT BE REVERSED AS [::-1]
    #        set up the spring constant. We give root the lowest temperature.
    #        This should increase performance when pair lists are used (they are updated less often at low temperature
    #        or when steps involve minimisation, as the low temperatures are closer to the minimum)
    #        """
    #        if (self.rank == 0):
    #            Tarray = spring_constants_variable_transform(self.nproc+1, self.Tmax, self.u2meank0,
    #                                                         self.mcrunner.nparticles, self.mcrunner.bdim, self.Tmin)
    #            Tarray = Tarray[::-1]
    #            Tarray = np.array(Tarray[1:],dtype='d') #exclude kmax entry, no need to be simulated, mean is already available
    #            self.Tarray = Tarray
    #        else:
    #            self.Tarray = None

    # THIS _get_temps CAN DEAL WITH NEGATIVE Ks
    def _get_temps(self):
        """
        set up the temperatures by distributing them exponentially. We give root the lowest temperature.
        This should increase performance when pair lists are used (they are updated less often at low temperature
        or when steps involve minimisation, as the low temperatures are closer to the minimum)
        """
        if self.rank == 0:
            nposk = self.nproc - self.numnegk  # number of positive k
            Tarray = spring_constants_variable_transform(
                nposk + 1,
                self.Tmax,
                self.u2meank0,
                self.mcrunner.nparticles,
                self.mcrunner.bdim,
                self.Tmin,
            )
            Tarray = Tarray[
                :-1
            ]  # exclude kmax entry, no need to be simulated, mean is already available
            if self.numnegk > 0:
                assert np.abs(self.lownegk) > 0
                grid = -(
                    np.abs(self.lownegk)
                    + 1
                    - (
                        np.exp(
                            np.linspace(
                                np.log(1),
                                np.log(np.abs(self.lownegk) + 1),
                                self.numnegk + 1,
                            )
                        )
                    )
                )[:-1]
                assert grid.size == self.numnegk
                for x in grid[::-1]:
                    Tarray.insert(0, x)
            print("len Tarray", len(Tarray))
            print("Tarray:", Tarray)
            self.Tarray = np.array(Tarray[::-1], dtype="d")
        else:
            self.Tarray = None

    def _gather_objects(self, in_send_object):
        """Method to gather data in equal ordered chunks from replicas (it relies on the rank of the replica)

        .. note :: gather assumes that all the subprocess are sending the same amount of data to root, to send
                   variable amounts of data must use the MPI_gatherv directive
        """

        recv_objects = self.comm.gather(in_send_object, root=0)

        if self.rank != 0:
            assert recv_objects is None

        return recv_objects

    def _attempt_exchange(self):
        """
        this function brings together all the functions necessary to attempt a configuration swap, it is structures as
        following:
        *root gathers the energies from the slaves
        *red_origin is just the origin for systems with pbc and is the reduced set of coordinates for systems with frozen coordinates
        """
        # compute dx with com correction for each replica

        assert (
            isinstance(self.mcrunner, BV_MCrunner)
            or isinstance(self.mcrunner, HyperElemMCrunner)
            or isinstance(self.mcrunner, HyperElemOracleMCrunner)
        )

        # gather clouds, only root will do so
        clouds = self._gather_objects(self.cloud)
        if self.verbose:
            if clouds is not None:
                print("len(clouds)", len(clouds))
        # find exchange pattern (list of exchange buddies)
        exchange_pattern = self._find_exchange_buddy(clouds)
        # now scatter the exchange pattern so that everybody knows who their buddy is
        exchange_buddy = self._scatter_single_value(
            np.array(exchange_pattern, dtype="d")
        )
        exchange_buddy = int(exchange_buddy)
        # attempt configurations swap
        assert self.mcrunner.potential.get_k() == self.T  # debug
        self.config = self._exchange_pairs(
            exchange_buddy, np.array(self.config, dtype="d")
        )
        self.cloud = self._exchange_pairs_object(exchange_buddy, self.cloud)
        if exchange_buddy != self.no_exchange_int:
            # recompute energy (this assumes that mcrunner has member origin)
            self.energy = self.mcrunner.potential.getEnergy(
                np.array(self.config, dtype="d")
            )
            self.update_cloud_bias()

    def update_cloud_bias(self):
        dx2 = np.linalg.norm(self.cloud.x - self.mcrunner.origin, axis=1) ** 2
        new_bias = np.exp(-0.5 * self.T * dx2)
        self.cloud.bias = np.asarray(new_bias)

    def _find_exchange_buddy(self, clouds):
        """
        This function determines the exchange pattern alternating swaps with right and left neighbours.
        An exchange pattern array is constructed, filled with self.no_exchange_int which
        signifies that no exchange should be attempted. This value is replaced with the
        rank of the processor with which to perform the swap if the swap attempt is successful.
        The exchange partner is then scattered to the other processors.
        """
        if self.rank == 0:
            assert len(clouds) == len(self.Tarray)
            exchange_pattern = np.empty(len(clouds), dtype="int32")
            exchange_pattern.fill(
                self.no_exchange_int
            )  # reset exchange pattern to no exchange
            self.anyswap = False

            for i in self.nodelist[1::2]:
                if self.verbose:
                    print(
                        "exchange choice: ",
                        self.exchange_dic[self.exchange_choice],
                    )  # this is a print statement that has to be removed after initial implementation

                cloud1 = clouds[i]
                T1 = self.Tarray[i]
                cloud2 = clouds[i + self.exchange_choice]
                T2 = self.Tarray[i + self.exchange_choice]
                origin = self.mcrunner.origin

                # Hamiltonia replica exchange

                rosenbluth11 = 0
                rosenbluth12 = 0
                for (x, o, b) in cloud1:
                    dx2 = np.linalg.norm(x - origin) ** 2
                    rosenbluth11 += o * np.exp(-0.5 * T1 * dx2)
                    rosenbluth12 += o * np.exp(-0.5 * T2 * dx2)

                rosenbluth21 = 0
                rosenbluth22 = 0
                for (x, o, b) in cloud2:
                    dx2 = np.linalg.norm(x - origin) ** 2
                    rosenbluth21 += o * np.exp(-0.5 * T1 * dx2)
                    rosenbluth22 += o * np.exp(-0.5 * T2 * dx2)

                w_num = rosenbluth12 * rosenbluth21
                w_den = rosenbluth11 * rosenbluth22

                if w_num == 0 or w_den == 0:
                    raise RuntimeError

                w = w_num / w_den
                rand = np.random.rand()

                # print 'w {} rand {}'.format(w,rand)
                # print 'deltaE {} deltaT {}'.format(deltaE, deltabeta)
                # print "E1 {0} T1 {1} E2 {2} T2 {3} w {4}".format(E1,T1,E2,T2,w)
                if w > rand:
                    # accept exchange
                    if self.verbose:
                        self.ex_outstream.write(
                            "accepting exchange %d %d %g %g %d\n"
                            % (
                                self.nodelist[i],
                                self.nodelist[i + self.exchange_choice],
                                w_num,
                                w_den,
                                self.ptiter,
                            )
                        )
                    assert (
                        exchange_pattern[i] == self.no_exchange_int
                    )  # verify that is not using the same processor twice for swaps
                    assert (
                        exchange_pattern[i + self.exchange_choice]
                        == self.no_exchange_int
                    )  # verify that is not using the same processor twice for swaps
                    exchange_pattern[i] = self.nodelist[
                        i + self.exchange_choice
                    ]
                    exchange_pattern[i + self.exchange_choice] = self.nodelist[
                        i
                    ]
                    self.anyswap = True
            ############end of for loop###############
            # record self.permutation_pattern to print permutations in print function
            if self.anyswap:
                for i, buddy in enumerate(exchange_pattern):
                    if buddy != self.no_exchange_int:
                        self.permutation_pattern[i] = (
                            buddy + 1
                        )  # to conform to fortran notation
                    else:
                        self.permutation_pattern[i] = (
                            i + 1
                        )  # to conform to fortran notation
                self._master_print_permutations()
        else:
            exchange_pattern = None

        self.exchange_choice *= -1  # swap direction of exchange choice
        # print "exchange_pattern",exchange_pattern
        return exchange_pattern

    def _point_to_point_exchange_replace_object(self, dest, source, data):
        """swap data between two processors

        .. note :: the message sent buffer is replaced with the received message

        Parameters
        ----------
        dest : int
            rank of processor with which to swap
        source : int
            rank of processor with which to swap
        data : numpy.array
            array of data to exchage

        """
        assert dest == source
        newdata = self.comm.sendrecv(data, dest=dest, source=source)
        return newdata

    def _exchange_pairs_object(self, exchange_buddy, data):
        """Return data from the pair exchange, otherwise return the data unaltered.

        .. warning :: the replica sends to exchange_partner and receives from it,
                      replacing source with self.rank would cause a deadlock

        Parameters
        ----------
        exchange_buddy : int
            rank of processor with which to swap
        data : numpy.array
            array of data to exchage

        Returns
        -------
        data : numpy.array
            the send data buffer is replaced with the receive data
        """
        if exchange_buddy != self.no_exchange_int:
            # print "processor {0} p-to-p exchange, old data {1}".format(self.rank, data)
            data = self._point_to_point_exchange_replace_object(
                exchange_buddy, exchange_buddy, data
            )
            # print "processor {0} p-to-p exchange, new data {1}".format(self.rank, data)
            self.swap_accepted_count += 1
        else:
            self.swap_rejected_count += 1
        return data

    def one_iteration(self):
        """Perform one parallel tempering iteration

        Each PT iteration consists of the following steps:

        * set the coordinates
        * run the MCrunner for a predefined number of steps
        * collect the results (energy and new coordinates)
        * attempt an exchange
        """
        # set configuration and temperature at which want to perform run
        self.mcrunner.set_config(np.array(self.config, dtype="d"), self.energy)
        self.mcrunner.cloud_test.set_cloud(self.cloud)
        # now run the MCMC walk
        self.mcrunner.run()
        # collect the results
        result = self.mcrunner.get_results()
        self.energy = result.energy
        self.config = np.array(result.coords, dtype="d")
        self.cloud = self.mcrunner.cloud_test.get_old_cloud()
        if self.ptiter >= self.skip:
            if self.swap:
                self._attempt_exchange()
            # print and increase parallel tempering count and test convergence
            if self.ptiter % self.pfreq == 0:
                self.max_ptiter = self._test_convergence()
                self._print_data()
            if self.print_status:
                self._print_status()
        self.ptiter += 1

    def _initialise(self):
        """
        perform all the tasks required prior to starting the computation
        """
        self._get_temps()
        self.T = self._scatter_single_value(self.Tarray)
        if self.verbose:
            print("processor {0} temperature {1}".format(self.rank, self.T))
        self.mcrunner.set_control(self.T)
        self.config, self.energy = self.mcrunner.get_config()
        self.cloud = self.mcrunner.cloud_test.get_old_cloud()
        self._print_initialise()
        self.initialised = True
