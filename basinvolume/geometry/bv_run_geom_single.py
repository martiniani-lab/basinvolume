import numpy as np
import os
import argparse
from basinvolume.geometry.point_sampling import _hyperelem_findk_mcrunner, _hyperelem_kmin_mcrunner
from basinvolume.cluster_manager import BuildPBSScript

i32max = np.iinfo(np.int32).max

def run_preliminary(ndim, geometry, geom_params, stepsize):

    seeds = dict(seed_takestep=np.random.randint(i32max), seed_metropolis=np.random.randint(i32max),
                 seed_oracle=np.random.randint(i32max), seed_record_drop_r=np.random.randint(i32max))

    sim_kmin = _hyperelem_kmin_mcrunner(ndim, geometry=geometry, geom_params=geom_params, niter=1e6, k=0.,
                                               stepsize=stepsize, seeds=seeds, single=True, verbose=False,
                                               hmax=15, hbinsize=0.001)
    sim_kmin.run()

    seeds = dict(seed_takestep=np.random.randint(i32max), seed_oracle=np.random.randint(i32max))

    sim_findk = _hyperelem_findk_mcrunner(ndim, geometry=geometry, geom_params=geom_params,
                                                 avgcount=1e6, k=0.01, ktarget=0.9, knavg=1e5,
                                                 seeds=seeds, verbose=True, niter=1e8)
    sim_findk.run()
    assert os.path.normpath(sim_kmin.base_directory) == os.path.normpath(sim_kmin.base_directory)
    return os.path.normpath(sim_kmin.base_directory)

def submit_parallel_tempering(base_dir, pt_queue_type, pt_nodes, pt_cores, pt_walltime,
                              in_queue_type, in_nodes, in_cores, in_walltime, numnegk=0,
                              stol=0.05, oddcores=None, noswap=True):
    base_name = os.path.basename(os.path.normpath(base_dir))
    pt_script = "/home/sm958/Work/basinvolume/basinvolume/geometry/point_sampling/bv_parallel_tempering.py"
    if noswap:
        pt_args = "{} --numnegk {} -s {} --noswap".format(os.path.normpath(base_dir), numnegk, stol)
    else:
        pt_args = "{} --numnegk {} -s {}".format(os.path.normpath(base_dir), numnegk, stol)
    pt_command = 'python {} {}'.format(pt_script, pt_args)
    in_script = "/home/sm958/Work/basinvolume/basinvolume/geometry/bv_run_geom_innersphere_single.py"
    in_args = "{}".format(os.path.normpath(base_dir))
    in_command = 'python {} {}'.format(in_script, in_args)
    pt_fname = 'pt_'+base_name+'.sh'
    pt_jobname = 'pt_'+base_name.replace('oracle_hyper','')
    in_fname = 'in_'+base_name+'.sh'
    in_jobname = 'in_' + base_name.replace('oracle_hyper', '')
    in_job = BuildPBSScript(in_queue_type, in_nodes, in_cores, in_walltime, in_command)
    in_job.writePBSscript(in_fname, in_jobname)
    command = '{} && qsub ${{PBS_O_WORKDIR}}/{}'.format(pt_command,in_fname)
    pt_job = BuildPBSScript(pt_queue_type, pt_nodes, pt_cores, pt_walltime, command, oddcores=oddcores)
    pt_job.submit_PBS(pt_fname, pt_jobname)


def run_all(ndim, geometry, geom_params, stepsize,
            pt_queue_type='s16', pt_cores=15, pt_walltime=12,
            in_queue_type='s1', in_nodes=1, in_cores=1, in_walltime=6, numnegk=0,
            stol=0.05, maxppc=8, noswap=True):
    base_dir = run_preliminary(ndim, geometry, geom_params, stepsize)
    pt_nodes = max(1, pt_cores // maxppc)
    evencores = min(maxppc, pt_cores)
    oddcores = pt_cores % maxppc if (pt_cores % maxppc != 0 and pt_cores > maxppc) else None
    submit_parallel_tempering(base_dir, pt_queue_type, pt_nodes, evencores, pt_walltime,
                              in_queue_type, in_nodes, in_cores, in_walltime, numnegk=numnegk,
                              stol=stol, oddcores=oddcores, noswap=noswap)
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="run preliminary computations and parallel tempering")
    parser.add_argument("-d", "--ndim", type=int, help="number of degrees of freedom", default=2)
    parser.add_argument("-g", "--geometry", type=str, help="geometry, options: 1) cube 2) sphere"
                                                          "3) cube_exp_decay 4) sphere_exp_decay"
                                                          "5) sphere_pow_decay", default="sphere")
    parser.add_argument("-p", "--geom-params", type=float, nargs='+', help="geometry parameters", required=True)
    parser.add_argument("-x", "--stepsize", type=float, help="mc step size", required=True)
    parser.add_argument("--pt", action='store_true', help="run parallel tempering", default=False)
    args = parser.parse_args()

    ndim = args.ndim
    geometry=args.geometry
    geom_params=args.geom_params
    stepsize=args.stepsize
    noswap = not args.pt
    run_all(ndim, geometry, geom_params, stepsize, noswap=noswap)

