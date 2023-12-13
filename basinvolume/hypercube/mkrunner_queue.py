from __future__ import division, print_function
from builtins import range
import numpy as np
import sys
import argparse
import os
from mpi4py import MPI

from basinvolume.spheres import (
    MPI_BV_PT_RLhandshake,
    PT_Worker,
    PT_Master,
    ExchangeScheme,
)
from basinvolume.utils import (
    view_traceback,
    check_kmax_reasonable,
    import_pt_time_series,
)

from basinvolume.hypercube import _hypercube_findk_mcrunner
from basinvolume.hypercube import _hypercube_kmin_mcrunner
from basinvolume.hypercube import _hypercube_bv_mcrunner
from basinvolume.hypercube import _hypercube_innersphere_mcrunner
from basinvolume.hypercube.hypercube_compute_volume import (
    hypercube_mbar_compute_dos,
)

# for plotting histogram
from itertools import cycle
from scipy.integrate import quad

try:
    import matplotlib.pyplot as plt

    # more stuff for plotting histogram and comparing to prediction
    #######################SET LATEX OPTIONS###################
    plt.rc("text", usetex=False)  # True = bugs on the cluster!
    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    # rc('text.latex',preamble=r'\usepackage{times}')
    plt.rcParams.update({"font.size": 20})
    plt.rcParams["xtick.major.pad"] = 8
    plt.rcParams["ytick.major.pad"] = 8
    ##########################################################
    ####SET COLOUR MAP######
    cm = plt.get_cmap("Dark2")
    ########################
    #####################LINE STYLE CYCLER####################
    lines = ["-", "--", "-."]
    linecycler = cycle(lines)
    color_cycle = [cm(1.0 * i / 6) for i in range(6)]
    ##########################################################
except ImportError as err:
    print(err)

if __name__ == "__main__":
    # to run harmonic potential go to tests

    parser = argparse.ArgumentParser(
        description="Run a full analysis for a d-dimensional hypercube"
    )
    parser.add_argument("cubedim", type=int, help="dimension of the cube")
    parser.add_argument(
        "-k",
        "--positivespringnumber",
        type=int,
        help="number of different POSITIVE spring constants, \
                        default: 8",
        default=8,
    )
    parser.add_argument(
        "-negk",
        "--negativespringnumber",
        type=int,
        help="number of different NEGATIVE spring constants, \
                        default: 8",
        default=8,
    )
    parser.add_argument(
        "-min_n",
        "--min_tot_niter",
        type=float,
        help="minimal number of total steps in the random walks, \
                        default: 5e5",
        default=5e5,
    )
    parser.add_argument(
        "-n_spheres",
        "--number_nested_spheres",
        type=int,
        help="number of nested inner spheres to use, \
                        default: 1",
        default=1,
    )
    parser.add_argument(
        "-force_k",
        "--force_kmax_value",
        type=bool,
        help="Option to force k_max to reach the innersphere by hand, \
                        default: False",
        default=False,
    )
    parser.add_argument(
        "-k_sprd",
        "--k_spreading",
        type=str,
        help="Set the way in which the k's are spread. Options: linspace, logspace, positionlinspace, gausslobato,\
                        default = gausslobato",
        default="gausslobato",
    )
    parser.add_argument(
        "--auto_replica_number",
        action="store_true",
        help="overrides --positivespringnumber and increases the number of replicas if needed",
        default=False,
    )
    parser.add_argument(
        "--bias",
        help = "Biasing potentials used in PT\
            default = harmonic",
        default = "harmonic"
    )
    # parser.add_argument("-v","--verbose", action='store_true', help="verbosity",default=False)
    args = parser.parse_args()

    ndof = args.cubedim
    override_replicas = args.auto_replica_number

    import time

    origin = np.zeros(ndof)
    bv_pt_printstatus = False
    # build start configuration
    full_coords = np.array(origin)
    numposk = args.positivespringnumber
    if override_replicas:
        defaultnumber = int(ndof / 5)
        numposk = max(numposk, defaultnumber)
    numnegk = args.negativespringnumber
    nreplicas = numposk + numnegk
    min_tot_niter = int(args.min_tot_niter)
    number_nested_spheres = args.number_nested_spheres
    force_kmax_value = args.force_kmax_value
    k_spreading = args.k_spreading
    bias = args.bias
    i32max = np.iinfo(np.int32).max
    seeds = dict(
        seed_takestep=np.random.randint(i32max),
        seed_metropolis=np.random.randint(i32max),
    )

    # prepare MC runner
    print("Setting up MPI comm links\n")
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    print("nprocs = " + str(nprocs) + "\n")
    rank = comm.Get_rank()
    host = os.uname()[1]
    print(f"hello from process {rank} on host {host}")

    publicdoneflag = False
    rank0doneflag = False
    # Create a string with the name of the relevant directory
    directory_name = (
        "explore_bv_hypercube_n" + str(ndof) + "_l1"
    )  # +'_numposk'+str(numposk)+'_numnegk'+str(numnegk)+'_mintotniter'+str(min_tot_niter)

    # First, run the findk routine
    if rank == 0:
        sidelength = 1.0
        k_guess = (ndof - 1) / (0.5 * sidelength) ** 2
        findk_niter = 1e8
        target_acceptance = 0.9
        if force_kmax_value:
            ktol = 1.0  # The guess above brings to the right value, if the tolerance lets it pass it will be the final value
        else:
            ktol = 0.0025

        # Given the target acceptance, Find an approximate corresponding k
        # in this case the target acceptance is set high, so that the found k
        # will correspond to the maximum k we will use for our simulation
        # i.e the monte carlo simulation will roughly remain in the inner sphere
        sim = _hypercube_findk_mcrunner(
            ndof,
            sidelength=1,
            k=k_guess,
            target_acceptance=target_acceptance,
            ktol=ktol,
            knavg=1e5,
            niter=findk_niter,
            seeds=seeds,
            verbose=True,
        )
        print("\n\nsimulation: Find k started")
        start = time.time()
        sim.run()
        end = time.time()
        print("time elapsed", end - start)
        status = sim.mcrunner.get_status()
        print(status)
        print("self.kmax:", sim.kmax)
        print("self.prob:", sim.prob)
        print("self.displ_k_max:", sim.displ_k_max)
        print("self.var_displ_k_max:", sim.var_displ_k_max)
        print("(N-1)d/k", sim.ndof / sim.kmax)
        print("entries in histogram:", sim.mcrunner.get_entries())

        # kmin calcu
        sim_kmin = _hypercube_kmin_mcrunner(
            ndof,
            sidelength=1,
            niter=1e6,
            bias_params=[0.0],
            seeds=seeds,
            single=True,
            verbose=True,
            hmax=15,
            hbinsize=0.001,
        )
        # record_steps_timeseries=True, record_steps_timeseries_every=[int(np.ceil(1.5**n)) for n in xrange(22)],)
        print("\n\nsimulation: k_min started")
        start = time.time()
        sim_kmin.run()
        end = time.time()
        print("time elapsed", end - start)
        status = sim_kmin.mcrunner.get_status()
        print(status)
        print("d kmin: ", sim_kmin.displ_k_min)
        print("var: ", sim_kmin.var_displ_k_min)
        print("mean_coord_dist: ", sim_kmin.mean_coord_dist)
        print("var_coord_dist: ", sim_kmin.var_coord_dist)
        traj = sim_kmin.trajectory
        print(np.shape(traj))

        plt.plot(sim_kmin.traj_eval / np.amax(sim_kmin.traj_eval))
        print("asphericity factor", sim_kmin.asphericity)
        plt.show()

        # Configure_bv_mcrunner
        sim_bvconfig = _hypercube_bv_mcrunner(0, 1)
        mcrunner_bvconfig = sim_bvconfig(directory_name, seeds=seeds, verbose=True, niter=1e6)
        print("\n\nsimulation: BV config started")
        start = time.time()
        mcrunner_bvconfig.run()
        end = time.time()
        print("time elapsed", end - start)
        status = mcrunner_bvconfig.get_status()
        print(status)
        # The working directory changed during the call, change it back
        os.chdir("../")

        rank0doneflag = True

    # Run BV PT ONCE THE ABOVE IS COMPLETE!
    while publicdoneflag == False:
        publicdoneflag = comm.bcast(rank0doneflag, root=0)

    if publicdoneflag:

        start = time.time()

        moveall = False
        single = not moveall
        # Parallel Tempering
        max_tot_niter = int(4 * min_tot_niter)
        min_ptiter = int(
            min_tot_niter * 0.1
        )  # 10% PT swaps, this is the initial proposed maximum length of the run. at the end of min_ptiter convergence is checked
        niter = int((min_tot_niter - min_ptiter) / min_ptiter)  # 90% MCMC walk
        adjustf_niter = int(min_tot_niter * 0.1)  # equilibrate for the first 1/10th of total steps
        nskip = int(adjustf_niter / niter)  # don't swap while adjusting the step-size
        # pt_eq_niter equilibrate pt for the following 4/10th of total steps (), this has an effect on histogram
        # and on checksameminimum: it only starts recording the neighbouring minima when equilibration is reached
        pt_eq_niter = 0  # set to 0
        # the histogram starts recording the mean after adjustf_niter+pt_eq_niter steps
        pfreq = int(
            (min_ptiter - 1) * 0.1
        )  # print every 1/10th of min_ptiter (this will give 5 snapshots) #this is also frequency of tests
        ts_freq = 1
        ts_niter = int(niter * pfreq / ts_freq)
        test_convergence_ts = True
        record_histogram = False
        assert (
            record_histogram == False and pt_eq_niter == 0 and ts_freq == 1
        )  # ts_freq must be 1 with current output implementation (all based on timeseries)
        rel_std_err = 0.05  # relative standard error in the mean used by convergence test
        min_window = int(
            min_tot_niter * 0.5
        )  # minimum amount of data before trying to check convergence
        max_eq_time = int(
            min_tot_niter * 0.5
        )  # maximum amount of data to discard (throw away max the first 2.5e5 points, to avoid reading spurious features)
        fast_ct = False  # if false skip heuristic search for equilibration point
        i32max = np.iinfo(np.int32).max
        seeds = dict(
            seed_takestep=np.random.randint(i32max),
            seed_metropolis=np.random.randint(i32max),
        )
        print(seeds)
        print("\n\nsimulation: BV PT rank {} started".format(rank))

        sim_pt = _hypercube_bv_mcrunner(rank, nprocs)

        # Specific check overlap with cell lists and job queue doesn't work.
        # Use specific version without cell lists instead, since that's faster than
        # the non-specific cell lists version
        mcrunner_checkoverlap_cell_lists = False

        mcrunner_pt = sim_pt(
            directory_name,
            bias = bias,
            niter=niter,
            stepsize=5e-1,
            hmin=0,
            hmax=1,
            hbinsize=1e-4,
            acceptance=0.2,
            adjustf=0.9,
            adjustf_niter=adjustf_niter,
            adjustf_navg=100,
            pt_eq_niter=pt_eq_niter,
            ts_niter=ts_niter,
            ts_freq=ts_freq,
            seeds=seeds,
            single=single,
            record_histogram=record_histogram,
        )
        if not check_kmax_reasonable(sim_pt.findk_configpath):
            print("bv_parallel_tempering: kmax is unreasonable, exiting")
            logging.error("bv_parallel_tempering: kmax is unreasonable, exiting")
            sys.exit()

        # prepare PT runner
        kmin = 0
        displ_k_min = sim_pt.displ_k_min
        var_displ_k_min = sim_pt.displ_k_min
        kmax = sim_pt.kmax
        path = sim_pt.base_directory

        exchange_scheme = ExchangeScheme.NEIGHBOR_EXCHANGE

        if nprocs < nreplicas:
            if rank == 0:
                print("Using job queue with {} workers.".format(nprocs - 1))
                print("Using {} replicas.".format(nreplicas))
                checkpoint_time = None

                verbose = False
                sleep_seconds = 0.0001

                master = PT_Master(
                    nreplicas,
                    mcrunner_pt,
                    kmax,
                    kmin,
                    displ_k_min,
                    max_ptiter=min_ptiter + 1,
                    pfreq=pfreq,
                    skip=nskip,
                    test_convergence=test_convergence_ts,
                    fast_ct=fast_ct,
                    rel_std_err=rel_std_err,
                    min_window=min_window,
                    max_eq_time=max_eq_time,
                    eq_max_ptiter=int(max_tot_niter / niter),
                    numnegk=numnegk,
                    k_spreading=k_spreading,
                    bias = bias,
                    print_status=bv_pt_printstatus,
                    base_directory=path,
                    sleep_seconds=sleep_seconds,
                    exchange_scheme=exchange_scheme,
                    checkpoint_time=checkpoint_time,
                )

                master.run()
                sim_pt.print_success_all("True")

                print(
                    "ptiter: {} niter: {} adjustf_niter: {} skip: {} pfreq: {}".format(
                        master.ptiter,
                        mcrunner_pt.niter,
                        adjustf_niter,
                        master.skip,
                        master.pfreq,
                    )
                )
            else:
                worker = PT_Worker(mcrunner_pt, fix_com=False)
                worker.run()

        else:
            if rank == 0:
                print("Using handshake with {} replicas.".format(nprocs))

            ptrunner = MPI_BV_PT_RLhandshake(
                mcrunner_pt,
                kmax,
                kmin,
                displ_k_min,
                max_ptiter=min_ptiter + 1,
                pfreq=pfreq,
                skip=nskip,
                test_convergence=test_convergence_ts,
                fast_ct=fast_ct,
                rel_std_err=rel_std_err,
                min_window=min_window,
                max_eq_time=max_eq_time,
                eq_max_ptiter=int(max_tot_niter / niter),
                numnegk=numnegk,
                k_spreading=k_spreading,
                bias = bias,
                base_directory=path,
                fix_com=False,
            )
            assert ptrunner.rank == rank, "rank id does not match"
            assert ptrunner.nprocs == nprocs, "number of processes does not match"

            # run simulation
            try:
                ptrunner.run()
                sim_pt.print_success_all("True")
            except:
                view_traceback()
                try:
                    sim_pt.print_success_all("False")
                except:
                    view_traceback()

            print(
                "core: {} ptiter: {} niter: {} adjustf_niter: {} skip: {} pfreq: {}".format(
                    rank,
                    ptrunner.ptiter,
                    mcrunner_pt.niter,
                    adjustf_niter,
                    ptrunner.skip,
                    ptrunner.pfreq,
                )
            )

        print("convert timeseries to hdf5...")

        if rank == 0:
            # it is imperative that max_series_size=0 to avoid loss of raw data, the objective of this step is to
            # reduce the amount of occupied memory and i/o speed without loosing any information
            timeseries = import_pt_time_series(
                sim_pt.base_directory,
                int(sim_pt.mc_params["adjustf_niter"]),
                max_series_size=0,
                ncores=1,
                del_raw=False,
            )
        end = time.time()
        print("Parallel tempering done")
        print("elapsed time", end - start)

        # The working directory changed during the call, change it back
        os.chdir("../")

        # Run the nested inner spheres
        # They can run in parallel, but there should be a flag to say that everyone is done
        innerspheres_done_flags = np.full(number_nested_spheres, False)
        alldone_spheres_flags = np.prod(innerspheres_done_flags)
        while alldone_spheres_flags == False:
            for i, sphere_number in enumerate(range(number_nested_spheres)):
                # It's convenient to shift indices by one because rank 0 is often busy generating the hdf5 file for a while
                if (sphere_number + 1) % nprocs == rank and innerspheres_done_flags[
                    sphere_number
                ] == False:
                    # Configure_innersphere
                    sim_innersphere = _hypercube_innersphere_mcrunner(
                        directory_name,
                        sphere_number,
                        niter=min_tot_niter,
                        seeds=seeds,
                        number_nested_spheres=number_nested_spheres,
                        verbose=False,
                    )  # switched to min_tot_niter iterations to be consistent!
                    print(
                        "\n\nsimulation: Inner Sphere number {} started on rank {}".format(
                            sphere_number, rank
                        )
                    )
                    start = time.time()
                    sim_innersphere.run()
                    end = time.time()
                    print("time elapsed", end - start)
                    status = sim_innersphere.mcrunner.get_status()
                    print(status)
                    print("stepsize: ", sim_innersphere.mcrunner.get_stepsize())
                    output_directory = directory_name + "/innersphere_" + str(sphere_number)
                    sim_innersphere.mcrunner.show_histogram_analytical(output_directory)
                    # This run is done!
                    innerspheres_done_flags[sphere_number] = True

            if rank != 0:
                comm.send(innerspheres_done_flags, dest=0)
            else:
                for k in range(nprocs - 1):
                    source_rank = k + 1
                    innerspheres_done_flags += comm.recv(source=source_rank)

            innerspheres_done_flags = comm.bcast(innerspheres_done_flags, root=0)
            alldone_spheres_flags = np.prod(innerspheres_done_flags)

        if rank == 0:
            # Compute volume by aggregating data
            # It's better to run the function than launch the script from the os!
            # os.system("python hypercube_compute_volume.py "+directory_name)
            show = False
            bootstrap = False
            kde = False
            cores = 1
            print("\n\nsimulation: Volume computation started")
            print("\nThread {} here!".format(rank))
            sim_compute_volume = hypercube_mbar_compute_dos(
                bootstrap=bootstrap, kde=kde, plot_dos_data=True, ncores=cores, bias = bias
            )
            sim_compute_volume(directory_name, show=show)
