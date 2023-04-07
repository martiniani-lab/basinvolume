from __future__ import division, print_function
from builtins import range
import numpy as np
import sys
import argparse
import os
from mpi4py import MPI

from basinvolume.spheres import (MPI_BV_PT_RLhandshake,
                                 PT_Worker, PT_Master, ExchangeScheme)
from basinvolume.utils import view_traceback, check_kmax_reasonable, import_pt_time_series

from basinvolume.hypercube import _hypercube_findk_mcrunner
from basinvolume.hypercube import _hypercube_kmin_mcrunner
from basinvolume.hypercube import _hypercube_bv_mcrunner
from basinvolume.hypercube import _hypercube_innersphere_mcrunner
from basinvolume.hypercube.hypercube_compute_volume import hypercube_mbar_compute_dos

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
    parser.add_argument("-min_n", "--min_tot_niter", type=float, help="minimal number of total steps in the random walks, \
                        default: 5e5",default=5e5)
    parser.add_argument("-n_spheres", "--number_nested_spheres", type=int, help="number of nested inner spheres to use, \
                        default: 2", default=2)
    parser.add_argument("-prefix", "--directory_prefix", type=str, help="Path to the output directory. Defaut: ''", default = '')
    # parser.add_argument("-v","--verbose", action='store_true', help="verbosity",default=False)
    args = parser.parse_args()

    ndof = args.cubedim
    
    import time

    origin = np.zeros(ndof)
    bv_pt_printstatus = False
    #build start configuration
    full_coords = np.array(origin)
    min_tot_niter = int(args.min_tot_niter)
    number_nested_spheres = args.number_nested_spheres
    directory_prefix = args.directory_prefix
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
    
    # Create a string with the name of the relevant directory
    directory_name=directory_prefix+'explore_bv_hypercube_n'+str(ndof)+'_l1'# +'_numposk'+str(numposk)+'_numnegk'+str(numnegk)+'_mintotniter'+str(min_tot_niter)
    print(directory_name)

    #Run the nested inner spheres
    #They can run in parallel, but there should be a flag to say that everyone is done
    innerspheres_done_flags = np.full(number_nested_spheres, False)
    alldone_spheres_flags = np.prod(innerspheres_done_flags)
    while alldone_spheres_flags == False:
        for i,sphere_number in enumerate(range(number_nested_spheres)):
            #It's convenient to shift indices by one because rank 0 is often busy generating the hdf5 file for a while
            if (sphere_number+1)%nprocs == rank and innerspheres_done_flags[sphere_number]==False:
                # Configure_innersphere
                sim_innersphere = _hypercube_innersphere_mcrunner(directory_name, sphere_number, niter=min_tot_niter, seeds=seeds, number_nested_spheres=number_nested_spheres, verbose=False) # switched to min_tot_niter iterations to be consistent!
                print('\n\nsimulation: Inner Sphere number {} started on rank {}'.format(sphere_number, rank))
                start=time.time()
                sim_innersphere.run()
                end=time.time()
                print('time elapsed', end-start)
                status = sim_innersphere.mcrunner.get_status()
                print(status)
                print('stepsize: ',sim_innersphere.mcrunner.get_stepsize())
                output_directory = directory_name+'/innersphere_'+str(sphere_number)
                sim_innersphere.mcrunner.show_histogram_analytical(output_directory)
                #This run is done!
                innerspheres_done_flags[sphere_number] = True
        
        if rank != 0:
            comm.send(innerspheres_done_flags, dest=0)
        else:
            for k in range(nprocs-1):
                source_rank = k+1
                innerspheres_done_flags += comm.recv(source=source_rank)
            
        innerspheres_done_flags = comm.bcast(innerspheres_done_flags, root=0)
        alldone_spheres_flags = np.prod(innerspheres_done_flags)


    if rank == 0:
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
