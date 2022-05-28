import numpy as np
import argparse
from basinvolume.geometry.cloud_sampling import _oracle_hyperelem_innersphere_mcrunner, hyperelem_mbar_compute_dos

i32max = np.iinfo(np.int32).max

def run_innersphere(base_dir, niter):
    seeds = dict(seed_takestep=np.random.randint(i32max), seed_metropolis=np.random.randint(i32max),
                 seed_oracle=np.random.randint(i32max))
    sim_innershphere = _oracle_hyperelem_innersphere_mcrunner(base_dir, niter, seeds=seeds, verbose=False)
    sim_innershphere.run()

def run_compute_volume(base_dir, ncores=1, bootstrap=False, kde=False, plot_dos_data=True):
    sim = hyperelem_mbar_compute_dos(ncores=ncores, bootstrap=bootstrap, kde=bootstrap, plot_dos_data=True)
    sim(base_dir, show=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="run preliminary computations")
    parser.add_argument("base_dir", type=str, help="packing file name")
    parser.add_argument("-n", "--niter", type=float, help="number of energy evaluation, default: 1e6", default=1e6)
    args = parser.parse_args()
    run_innersphere(args.base_dir, args.niter)
    run_compute_volume(args.base_dir)