from __future__ import division, print_function
from builtins import range
import numpy as np
import sys
import argparse
import os
from mpi4py import MPI

from pele.potentials import Harmonic
from mcpele.monte_carlo import _BaseMCRunner, NullPotential
from basinvolume.monte_carlo import RecordDisplacementTimeseries, CheckHyperCubicContainer, CheckHyperSphericalContainer, RecordStepsTimeseries, RecordDisp2Histogram
from mcpele.monte_carlo import MetropolisTest, RandomCoordsDisplacement
from basinvolume.monte_carlo import SampleUniformSphereGaussian
from mcpele.monte_carlo import SampleGaussian
from basinvolume.monte_carlo import Findk
from basinvolume.spheres import (configure_bv_mcrunner, MPI_BV_PT_RLhandshake,
                                 PT_Worker, PT_Master, ExchangeScheme)
from basinvolume.utils import write_2d_array_to_hdf5
from basinvolume.utils import view_traceback, check_kmax_reasonable, import_pt_time_series

from basinvolume.hypercube import _hypercube_findk_mcrunner
from basinvolume.hypercube import _hypercube_kmin_mcrunner
from basinvolume.hypercube import _hypercube_bv_mcrunner
from basinvolume.hypercube import _hypercube_innersphere_mcrunner
from basinvolume.hypercube.hypercube_compute_volume import hypercube_mbar_compute_dos

try:
    from mcpele.monte_carlo import ConfTestOR
    from mcpele.monte_carlo import RecordCoordsTimeseries
except Exception as e:
    print(e)

#for plotting histogram
from itertools import cycle
from scipy.integrate import quad

try:
    import matplotlib.pyplot as plt
    #more stuff for plotting histogram and comparing to prediction
    #######################SET LATEX OPTIONS###################
    plt.rc('text', usetex=False) #True = bugs on the cluster!
    plt.rc("font",**{"family":"serif","serif":["Computer Modern"]})
    #rc('text.latex',preamble=r'\usepackage{times}')
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['xtick.major.pad'] = 8
    plt.rcParams['ytick.major.pad'] = 8
    ##########################################################
    ####SET COLOUR MAP######
    cm = plt.get_cmap('Dark2')
    ########################
    #####################LINE STYLE CYCLER####################
    lines = ["-","--","-."]
    linecycler = cycle(lines)
    color_cycle=[cm(1. * i / 6) for i in range(6)]
    ##########################################################
except ImportError as err:
    print(err)

if __name__ == "__main__":
    #to run harmonic potential go to tests
    
    parser = argparse.ArgumentParser(description="Run a full analysis for a d-dimensional hypercube")
    parser.add_argument("cubedim", type=int, help="dimension of the cube")
    parser.add_argument("-k","--positivespringnumber", type=int, help="number of different POSITIVE spring constants, \
                        default: 25",default=8)
    parser.add_argument("-negk", "--negativespringnumber", type=int, help="number of different NEGATIVE spring constants, \
                        default: 10",default=8)
    parser.add_argument("-min_n", "--min_tot_niter", type=float, help="minimal number of total steps in the random walks, \
                        default: 5e5",default=5e5)
    # parser.add_argument("-v","--verbose", action='store_true', help="verbosity",default=False)
    args = parser.parse_args()

    ndof = args.cubedim
    
    import time

    origin = np.zeros(ndof)
    potential = NullPotential()
    bv_pt_printstatus = False
    #build start configuration
    full_coords = np.array(origin)
    numposk = args.positivespringnumber
    numnegk = args.negativespringnumber
    nreplicas = numposk + numnegk
    min_tot_niter = int(args.min_tot_niter)
    i32max = np.iinfo(np.int32).max
    seeds = dict(seed_takestep=np.random.randint(i32max),seed_metropolis=np.random.randint(i32max))
    
    #prepare MC runner
    print("Setting up MPI comm links\n")
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    print("nprocs = "+str(nprocs)+"\n")
    rank = comm.Get_rank()
    host = os.uname()[1]
    print(f"hello from process {rank} on host {host}")
    
    publicdoneflag=False
    rank0doneflag=False
    # Create a string with the name of the relevant directory
    directory_name='explore_bv_hypercube_n'+str(ndof)+'_l1'
    
    # First, run the findk routine
    if rank == 0:
        sidelength = 1.0
        k_guess = 1.0 / np.sqrt(0.5 * sidelength)
        findk_niter = 5e5
        sim = _hypercube_findk_mcrunner(ndof, sidelength=1, k=0.1, ktarget=0.9, knavg=1e3, niter=findk_niter,
                                    seeds=seeds, verbose=True)
        print('\n\nsimulation: Find k started')
        start=time.time()
        sim.run()
        end=time.time()
        print('time elapsed', end-start)
        status = sim.mcrunner.get_status()
        print(status)
        print("self.kmax:", sim.kmax)
        print("self.prob:", sim.prob)
        print("(N-1)d/k", sim.ndof / sim.kmax)
    
        # Then, run the kmin run
        sim_kmin = _hypercube_kmin_mcrunner(ndof, sidelength=1, niter=1e6, k=0, seeds=seeds,
                            single=True, verbose=True, hmax=15, hbinsize=0.001)
        #record_steps_timeseries=True, record_steps_timeseries_every=[int(np.ceil(1.5**n)) for n in xrange(22)],)
        print('\n\nsimulation: Find k_min started')
        start=time.time()
        sim_kmin.run()
        end=time.time()
        print('time elapsed', end-start)
        status = sim_kmin.mcrunner.get_status()
        print(status)
        print('d kmin: ',sim_kmin.displ_k_min)
        print('var: ',sim_kmin.var_displ_k_min)
        print('mean_coord_dist: ',sim_kmin.mean_coord_dist)
        print('var_coord_dist: ', sim_kmin.var_coord_dist)
        traj = sim_kmin.trajectory
        print(np.shape(traj))
        
        plt.plot(sim_kmin.traj_eval/np.amax(sim_kmin.traj_eval))
        print("asphericity factor", sim_kmin.asphericity)
        plt.show()
        
        # Configure_bv_mcrunner            
        sim_bvconfig = _hypercube_bv_mcrunner(0, 1)
        mcrunner_bvconfig = sim_bvconfig(directory_name, seeds=seeds, verbose=True, niter=1e6)
        print('\n\nsimulation: BV config started')
        start=time.time()
        mcrunner_bvconfig.run()
        end=time.time()
        print('time elapsed', end-start)
        status = mcrunner_bvconfig.get_status()
        print(status)
        # The working directory changed during the call, change it back
        os.chdir('../')
        
        rank0doneflag=True
    
    # Run BV PT ONCE THE ABOVE IS COMPLETE!
    while publicdoneflag==False:
        publicdoneflag = comm.bcast(rank0doneflag, root=0)
        
    if publicdoneflag:

        start=time.time()

        moveall=False
        single = not moveall
        #Parallel Tempering
        max_tot_niter = int(4*min_tot_niter)
        min_ptiter = int(min_tot_niter*0.1) #10% PT swaps, this is the initial proposed maximum length of the run. at the end of min_ptiter convergence is checked
        niter = int((min_tot_niter-min_ptiter)/min_ptiter) #90% MCMC walk
        adjustf_niter = int(min_tot_niter*0.1) #equilibrate for the first 1/10th of total steps
        nskip = int(adjustf_niter/niter) #don't swap while adjusting the step-size
        # pt_eq_niter equilibrate pt for the following 4/10th of total steps (), this has an effect on histogram
        # and on checksameminimum: it only starts recording the neighbouring minima when equilibration is reached
        pt_eq_niter = 0 #set to 0
        #the histogram starts recording the mean after adjustf_niter+pt_eq_niter steps
        pfreq = int((min_ptiter-1)*0.1) #print every 1/10th of min_ptiter (this will give 5 snapshots) #this is also frequency of tests
        ts_freq = 1
        ts_niter = int(niter*pfreq/ts_freq)
        test_convergence_ts=True
        record_histogram=False
        assert(record_histogram == False and pt_eq_niter == 0 and ts_freq == 1) #ts_freq must be 1 with current output implementation (all based on timeseries)
        rel_std_err= 0.05 #relative standard error in the mean used by convergence test
        min_window=int(min_tot_niter * 0.5) #minimum amount of data before trying to check convergence
        max_eq_time=int(min_tot_niter * 0.5) #maximum amount of data to discard (throw away max the first 2.5e5 points, to avoid reading spurious features)
        fast_ct=False #if false skip heuristic search for equilibration point
        i32max = np.iinfo(np.int32).max
        seeds = dict(seed_takestep=np.random.randint(i32max),seed_metropolis=np.random.randint(i32max))
        print(seeds)
        print('\n\nsimulation: BV PT rank {} started'.format(rank))
        
        sim_pt = _hypercube_bv_mcrunner(rank, nprocs)
        
        # Specific check overlap with cell lists and job queue doesn't work.
        # Use specific version without cell lists instead, since that's faster than
        # the non-specific cell lists version
        mcrunner_checkoverlap_cell_lists = False
    
        mcrunner_pt = sim_pt(directory_name, niter=niter, stepsize=5e-1, hmin=0, hmax=1, hbinsize=1e-4, acceptance=0.2,
                        adjustf=0.9, adjustf_niter=adjustf_niter, adjustf_navg=100, pt_eq_niter=pt_eq_niter,
                        ts_niter=ts_niter, ts_freq=ts_freq, seeds=seeds, single=single, record_histogram=record_histogram)
        if not check_kmax_reasonable(sim_pt.findk_configpath):
            print('bv_parallel_tempering: kmax is unreasonable, exiting')
            sys.exit()

        if not check_kmax_reasonable(sim_pt.findk_configpath):
            logging.error('bv_parallel_tempering: kmax is unreasonable, exiting')
            sys.exit()

        #prepare PT runner
        kmin = 0
        displ_k_min = sim_pt.displ_k_min
        var_displ_k_min = sim_pt.displ_k_min
        kmax = sim_pt.kmax
        lownegk = -2.5 #lownegk needs to be pretty low for hypercube exploration!
        path = sim_pt.base_directory
        
        exchange_scheme = ExchangeScheme.NEIGHBOR_EXCHANGE
        
        if nprocs < nreplicas:
            if rank == 0:
                print("Using job queue with {} workers.".format(nprocs - 1))
                checkpoint_time = None
                
                verbose=False
                sleep_seconds=0.0001
                
                master = PT_Master(
                    nreplicas, mcrunner_pt, kmax, kmin, displ_k_min, max_ptiter=min_ptiter+1,
                    pfreq=pfreq, skip=nskip, test_convergence=test_convergence_ts,
                    fast_ct=fast_ct, rel_std_err=rel_std_err, min_window=min_window,
                    max_eq_time=max_eq_time, eq_max_ptiter=int(max_tot_niter/niter),
                    numnegk=numnegk, lownegk=lownegk, print_status=bv_pt_printstatus,
                    base_directory=path, sleep_seconds=sleep_seconds,
                    exchange_scheme=exchange_scheme,
                    checkpoint_time=checkpoint_time)

                master.run()
                sim_pt.print_success_all("True")

                print('ptiter: {} niter: {} adjustf_niter: {} skip: {} pfreq: {}'
                            .format(master.ptiter, mcrunner_pt.niter, adjustf_niter,
                                    master.skip, master.pfreq))
            else:
                worker = PT_Worker(mcrunner_pt)
                worker.run()

        else:
            if rank == 0:
                print("Using handshake with {} replicas.".format(nprocs))

            ptreplica = MPI_BV_PT_RLhandshake(
                mcrunner_pt, kmax, kmin, displ_k_min, max_ptiter=min_ptiter+1,
                pfreq=pfreq, skip=nskip, test_convergence=test_convergence_ts,
                fast_ct=fast_ct, rel_std_err=rel_std_err, min_window=min_window,
                max_eq_time=max_eq_time, eq_max_ptiter=int(max_tot_niter/niter),
                numnegk=numnegk, lownegk=lownegk, base_directory=path)
            assert ptreplica.rank == rank, "rank id does not match"
            assert ptreplica.nprocs == nprocs, "number of processes does not match"

            # run simulation
            try:
                ptreplica.run()
                sim_pt.print_success_all("True")
            except:
                view_traceback()
                try:
                    sim_pt.print_success_all("False")
                except:
                    view_traceback()

            print('ptiter: {} niter: {} adjustf_niter: {} skip: {} pfreq: {}'
                        .format(ptreplica.ptiter, mcrunner_pt.niter, adjustf_niter,
                                ptreplica.skip, ptreplica.pfreq))
        
        print('convert timeseries to hdf5...')


        if rank == 0:
            #it is imperative that max_series_size=0 to avoid loss of raw data, the objective of this step is to
            #reduce the amount of occupied memory and i/o speed without loosing any information
            timeseries = import_pt_time_series(sim_pt.base_directory, int(sim_pt.mc_params['adjustf_niter']),
                                            max_series_size=0, ncores=1, del_raw=False)
        end=time.time()
        print('Parallel tempering done')
        print('elapsed time',end-start)
        
        # The working directory changed during the call, change it back
        os.chdir('../')
        
        if rank == 0:
            # Configure_innersphere
            sim_innersphere = _hypercube_innersphere_mcrunner(directory_name, niter=1e5, seeds=seeds, verbose=False)
            print('\n\nsimulation: Inner Sphere started')
            start=time.time()
            sim_innersphere.run()
            end=time.time()
            print('time elapsed', end-start)
            status = sim_innersphere.mcrunner.get_status()
            print(status)
            print('stepsize: ',sim_innersphere.mcrunner.get_stepsize())
            sim_innersphere.mcrunner.show_histogram_analytical()
            
        
            # Compute volume by aggregating data
            # It's better to run the function than launch the script from the os!
            # os.system("python hypercube_compute_volume.py "+directory_name)
            show=False
            bootstrap=False
            kde=False
            cores=1
            print('\n\nsimulation: Volume computation started')
            print("\nThread {} here!".format(rank))
            sim_compute_volume = hypercube_mbar_compute_dos(bootstrap=bootstrap, kde=kde, plot_dos_data=True, ncores=cores)
            sim_compute_volume(directory_name, show=show)
