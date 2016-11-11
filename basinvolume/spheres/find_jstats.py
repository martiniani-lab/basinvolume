from __future__ import division
from pele.potentials import HS_WCA, InversePowerStillingerCut
from pele.optimize import Result
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.utils import *
import argparse
from basinvolume.spheres import HS_Generate_Packing
from basinvolume.spheres.generate_jammed_packing import cartesian_to_polar2d, sum_neighbor_angles2d
from pele.utils._pressure_tensor import pressure_tensor
from joblib import Parallel, delayed
from numpy.random import RandomState
import cPickle as pickle


class SoftPackingDataset(object):
    def __init__(self, phi_ss, phi_hs, sca_ss,
                 mu, sig, seeds, boxv, hs_radii):
        self.phi_ss = phi_ss
        self.phi_hs = phi_hs
        self.sca = sca_ss
        self.rmu = mu
        self.rsig = sig
        self.seeds = seeds
        self.boxv = boxv
        self.hs_radii = hs_radii
        self.success = []
        self.packings_data = []

    def add_success(self, success):
        self.success.append(success)

    def add_packing_data(self, soft_packing_data):
        self.packings_data.append(soft_packing_data)

    def clear(self):
        self.packings_data = []


class SoftPackingData(object):
    def __init__(self, coords, energy, pressure, Z, nrattlers):
        self.nrattlers = nrattlers
        self.energy = energy
        self.pressure = pressure
        self.Z = Z #full contact list
        self.coords = coords

class GeneratePackingFindJ(HS_Generate_Packing):
    def __init__(self, nparticles, workspace=None, method='quench', bdim=3, boxv=None,
                 ss_packing_frac=0.86, sca=0.1212238211627763,
                 hs_radii=None, mu=1, sig=0.05, new_poly=False, hsf_niter=1e6, hsf_stepsize=1e-3,
                 max_iter=10, tol=1e-9, use_cell_lists=True, single=True, seeds=None,
                 opt_pot_str='hs_wca', start_iteration=0):
        if workspace is None:
            workspace = os.getcwd()
        if not os.path.isabs(workspace):
            workspace = os.path.abspath(workspace)
        self.workspace = workspace
        self.sca_ss = sca
        self.eps = 1.
        self.tol = tol
        self.opt_pot_str = opt_pot_str
        self.hs_packing_frac = ss_packing_frac / np.power(1.+sca, bdim)
        self.ss_packing_frac = ss_packing_frac
        print self.hs_packing_frac, self.ss_packing_frac
        super(GeneratePackingFindJ, self).__init__(nparticles, method=method, bdim=bdim, boxv=boxv,
                                                   packing_frac=self.hs_packing_frac, hs_radii=hs_radii, mu=mu,
                                                   sig=sig, new_poly=new_poly, hsf_niter=hsf_niter,
                                                   hsf_stepsize=hsf_stepsize, max_iter=max_iter,
                                                   use_cell_lists=use_cell_lists, single=single,
                                                   seeds=seeds, start_iteration=start_iteration)
        self.initialised_ss = False
        self.max_nrattlers = int(self.nparticles * 0.8)
        self.packing_dataset = SoftPackingDataset(self.ss_packing_frac, self.hs_packing_frac, self.sca_ss,
                                                  mu, sig, seeds, boxv, hs_radii)

    def run(self):
        """run generate packings"""
        while (self.iteration-self.start_iteration) < self.max_iter:
            self.one_iteration() #self iteration is incremented within one_iteration
            if (self.iteration-self.start_iteration) % 1000 == 0 or (self.iteration-self.start_iteration) == self.max_iter:
                self._dump_results()

    def _dump_results(self):
        data_name = "jammed_packings_{}D_mu{}_sig{}_sca{}_phi{}_iter{}.pickle".format(self.bdim,
                                                                                      self.mu,
                                                                                      self.sig,
                                                                                      self.sca_ss,
                                                                                      self.ss_packing_frac,
                                                                                      int(self.iteration)
                                                                                      )
        data_pickle = os.path.join(self.workspace, data_name)
        pickle.dump(self.packing_dataset, open(data_pickle, "wb"), protocol=-1)
        self.packing_dataset.clear()

    def one_iteration(self):
        """perform one iteration"""
        self._initialise()
        success = self._generate_packing_coords()
        if success:
            success = self._one_iteration_ss()
        if success:
            pressure, ptensor = pressure_tensor(self.potential_ss_p, self.coords_ss, np.prod(self.boxv), self.bdim)
            Z = [len(contacts) for contacts in self.contact_list]
            data = SoftPackingData(self.coords_ss, self.energy_ss, pressure, Z, self.nratls_)
            self.packing_dataset.add_packing_data(data)
        self.packing_dataset.add_success(success)
        self.iteration += 1
        print 'iteration ', self.iteration

    def _one_iteration_ss(self):
        """perform one iteration
        """
        if not self.initialised_ss:
            self._setup_one_iteration_ss()
            self.initialised_ss = True
        return self._generate_packing_coords_ss()

    def _setup_one_iteration_ss(self):
        # assert that largest soft particle is not > 1/2 of smallest box size
        if np.amax(self.hs_radii) * 2 * (1 + self.sca_ss) >= np.amin(self.boxv) / 2:
            print "WARNING: max soft diameter >= 1/2 box side!"
        if np.amax(self.hs_radii) * 2 * (1 + self.sca_ss) >= np.amin(self.boxv):
            raise Exception("WARNING: particle does not fit the box")

        ###potential needs to be called because self.coords_ss is an input argument of HS_WCAPeriodicCellLists

        rcut = np.amax(self.hs_radii) * 2.0 * (1.0 + self.sca_ss)  # rcut set to largest particle diameter
        if self.use_cell_lists:
            if np.amin(self.boxv) // rcut <= 3:
                self.use_cell_lists = False
        if self.opt_pot_str.lower() == "hs_wca":
            self.potential_ss_p = HS_WCA(use_periodic=True, eps=self.eps,
                                       sca=self.sca_ss, radii=self.hs_radii, boxvec=self.boxv,
                                       ndim=self.bdim)
            if self.use_cell_lists:
                self.potential_ss = HS_WCA(use_periodic=True,
                                        use_cell_lists=True, eps=self.eps, sca=self.sca_ss,
                                        radii=self.hs_radii, boxvec=self.boxv,
                                        reference_coords=self.coords, rcut=rcut,
                                        ndim=self.bdim, ncellx_scale=1.0)
            else:
                self.potential_ss = self.potential_ss_p
        elif self.opt_pot_str.lower() == "inverse_power_stillinger":
            self.stillinger_a_radii = self.hs_radii * (1 + self.sca_ss)
            pow = self.extra_pot_kwargs["pow"]
            rcut = self.extra_pot_kwargs["rcut"]
            self.potential_ss = InversePowerStillingerCut(pow,
                                                       self.stillinger_a_radii, ndim=self.bdim,
                                                       boxvec=self.boxv, rcut=rcut, use_cell_lists=True)
        else:
            raise NotImplementedError

    def _find_rattlers(self):
        """
        finish this, I need to remove the rattler and break. Also need to get compare to existing jammed_packing option
        :return:
        """
        if self.bdim == 2:
            zmin = 3
        elif self.bdim == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError

        def get_index(x):
            # x is a 3 array with the coordinates of the particles
            dij = np.zeros(self.bdim)
            dmin = np.amin(self.hs_radii) / 10.
            for j in xrange(self.nparticles):
                for k in xrange(self.bdim):
                    # use distances to nearest image convention
                    dij[k] = ((self.coords_ss[j * self.bdim + k] - x[k]) -
                              cround((self.coords_ss[j * self.bdim + k] - x[k]) / self.boxv[k]) * self.boxv[k])
                if np.linalg.norm(dij) < dmin:
                    return j

        coords = np.array(self.coords_ss)
        hs_radii = np.array(self.hs_radii)
        look = True
        self.nratls_ = 0
        self.rattlers_ = np.zeros(self.nparticles)

        while look:
            # print "restarting loop"
            found_rattler = False
            if self.nratls_ > self.max_nrattlers:
                return False
            radii = hs_radii * (1. + self.sca_ss)
            self.contact_list, neighbors_index_list = self._find_nearest_neighbors_ss(coords, radii)
            for i in xrange(len(hs_radii)):
                i1 = self.bdim * i
                no_neighbors = len(self.contact_list[i])
                # print "no_neighbors", no_neighbors
                if no_neighbors < zmin:
                    found_rattler = True
                    print "particle {} is not isostatic".format(i)
                else:
                    angles = [cartesian_to_polar2d(dij)[1] for dij in self.contact_list[i]]
                    neigh_vec = [x for (y, x) in sorted(zip(angles, self.contact_list[i]))]
                    sum_ = sum_neighbor_angles2d(neigh_vec)
                    found_rattler = np.abs(2 * np.pi - sum_) > 1e-10
                    if found_rattler:
                        print "asymmetric contact rattler, 2pi - theta = {}".format(2 * np.pi - sum_)
                # here assign correct index by searchin for the corresponding atom
                j = get_index(coords[i1:i1 + self.bdim])
                self.rattlers_[j] = 0 if found_rattler else 1000
                if found_rattler:
                    coords = np.delete(coords, [i1 + k for k in xrange(self.bdim)])  # remove particle from array
                    hs_radii = np.delete(hs_radii, [i])  # remove particle from array
                    self.nratls_ += 1
                    break
            look = True if found_rattler else False

        # test that number of contacts is sufficient for bulk modulus to be positive,
        # see eq 4 in http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.095704
        N_contacts = int(np.sum([len(contacts) for contacts in self.contact_list]))
        no_stable = len(self.contact_list)
        N_min = int(2 * (self.bdim * (no_stable - 1) + 1))
        print "N_min: {} N_contacts: {}".format(N_min, N_contacts)
        assert (self.nparticles - no_stable) == self.nratls_
        print "n rattlers ", self.nratls_
        if N_contacts >= N_min:
            return True
        else:
            print "packing is not globally stable, N_min: {} N_contacts: {}".format(N_min, N_contacts)
            return False

    def _find_nearest_neighbors_ss(self, coords, radii):
        nparticles = radii.size
        nnatoms_list = [[] for _ in xrange(nparticles)]
        nnatoms_index_list = [[] for _ in xrange(nparticles)]
        for i in xrange(nparticles):
            for j in xrange(i, nparticles):
                if i != j:
                    dij = np.zeros(self.bdim)
                    for k in xrange(self.bdim):
                        # use distances to nearest image convention
                        dij[k] = ((coords[j * self.bdim + k] - coords[i * self.bdim + k]) -
                                  cround((coords[j * self.bdim + k] - coords[i * self.bdim + k]) / self.boxv[k]) *
                                  self.boxv[k])
                    dijnorm = np.linalg.norm(dij)
                    dmin = radii[i] + radii[j]
                    if dijnorm <= dmin:
                        nnatoms_list[i].append(dij)
                        nnatoms_list[j].append(-dij)
                        nnatoms_index_list[i].append(j)
                        nnatoms_index_list[j].append(i)
        return nnatoms_list, nnatoms_index_list

    def _generate_packing_coords_ss(self):
        """
        perform quench and run tests
        """
        success = self._generate_packing_coords_iteration_ss(tol=self.tol)
        return success

    def _generate_packing_coords_iteration_ss(self, tol=1e-9, iprint=-1):
        """quenches the imported structure using FIRE"""
        fire_maxstep = np.amin(self.hs_radii) * self.sca_ss
        res = modifiedfire_cpp(self.coords, self.potential_ss, maxstep=fire_maxstep, nsteps=1e6, tol=tol, iprint=iprint)
        if not res.success:
            print 'quench failed'
            return False

        self.coords_ss = res.coords
        self.energy_ss = res.energy

        # test that on ri-minimisation the structure does not change
        res2 = modifiedfire_cpp(self.coords_ss, self.potential_ss, maxstep=fire_maxstep, nsteps=1e6, tol=tol)
        if res2.nfev > 1:
            print 'quench failed (structure changed at second minimisation)'
            return False

        # asserts that none of the hard sphere is overlapping
        no_overlap = self._check_no_overlaps()
        if not no_overlap:
            print 'overlap found'
            return False

        return self._find_rattlers()

def run_hsgp(hsgp):
    hsgp.run()

class FindJ(object):
    def __init__(self, nparticles, workspace=None, method='quench', bdim=3, boxv=None,
                 ss_packing_frac=[0.84], sca=0.1212238211627763, hs_radii=None, mu=1, sig=0.05,
                 new_poly=False, hsf_niter=1e6, hsf_stepsize=1e-3, max_iter=10, tol=1e-9,
                 use_cell_lists=True, single=True, seeds=None, opt_pot_str='hs_wca',
                 start_iteration=0, ncores=2):

        self.rng = RandomState()
        self.ncores = ncores
        self.hsgp = []
        if hs_radii is None:
            hs_radii = self.rng.normal(mu, sig, nparticles)
        for phi in ss_packing_frac:
            hsgp_ = GeneratePackingFindJ(nparticles, workspace=workspace, method=method,
                                         bdim=bdim, boxv=boxv, ss_packing_frac=phi,
                                         sca=sca, hs_radii=hs_radii, mu=mu,
                                         sig=sig, new_poly=new_poly, hsf_niter=hsf_niter,
                                         hsf_stepsize=hsf_stepsize, max_iter=max_iter, tol=tol,
                                         use_cell_lists=use_cell_lists, single=single,
                                         seeds=seeds, opt_pot_str=opt_pot_str, start_iteration=start_iteration)
            hs_radii = hsgp_.hs_radii
            self.hsgp.append(hsgp_)

    def run(self):
        results = Parallel(n_jobs=max(1, self.ncores))(
            delayed(run_hsgp)(hsgp_) for hsgp_ in self.hsgp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="generate 2/3-D hard disks/spheres packings")
    parser.add_argument("nparticles", type=int, help="number of particles")
    parser.add_argument("-n","--npackings", type=int, help="number of packings to produce",default=1)
    parser.add_argument("-d","--boxdim", type=int, help="box dimensions",default=3)
    parser.add_argument("-x", "--ncores", type=int, help="number of cores", default=2)
    # parser.add_argument("-p","--density", type=float, help="target packing fraction",default=0.86)
    parser.add_argument("-a", "--sca", type=float, help="1+a = r_ss/r_hs", default=0.1212238211627763)
    parser.add_argument("-u","--rmean", type=float, help="mean particle radius",default=1.0)
    parser.add_argument("-s","--rsigma", type=float, help="percent standard deviation",default=0.1)
    parser.add_argument("-m","--hsfniter", type=int, help="number of hard sphere fluid MC steps between 2 samples",default=1e6)
    parser.add_argument("-t","--hsfstep", type=float, help="stepsize for hard sphere fluid MC simulation",default=1e-3)
    parser.add_argument("-i", "--start-iter", type=int, help="starting label iteration, default=0", default=0)
    parser.add_argument("--newpoly", action='store_true', help="resample polidispersity at each iteration, default: False",default=False)
    parser.add_argument("--dpath", type=str, help="path to xy(z)d path from where to import diameters",default=None)
    parser.add_argument("--nocell", action='store_true', help="don't use cell lists, default: False",default=False)
    parser.add_argument("--moveall", action='store_true', help="move all particles at each step, default: False",default=False)
    parser.add_argument("--method", type=str, help="protocol to generate packings", default="quench")
    parser.add_argument("--tol", type=float, help="minimizer rms tolerance",default=1e-8)
    parser.add_argument("--phimin", type=float, help="smallest density to run", default=0.83)
    parser.add_argument("--phimax", type=float, help="largest density to run", default=0.87)
    parser.add_argument("--nphi", type=int, help="number of densities to run", default=32)

    args = parser.parse_args()
    print args
    single = not args.moveall

    #import radii from other configuration file
    dpath = args.dpath
    hs_radii = None
    if dpath:
        if not os.path.isabs(args.dpath):
            dpath = os.path.abspath(dpath)
        if args.boxdim == 2:
            coords, hs_diameters = read_xyd(dpath)
        else:
            coords, hs_diameters = read_xyzd(dpath)
        hs_radii = hs_diameters/2

    # density = args.density
    density = np.logspace(np.log10(args.phimin), np.log10(args.phimax), args.nphi)

    sim = FindJ(args.nparticles, method=args.method, bdim=args.boxdim, ss_packing_frac=density,
                sca=args.sca, hs_radii=hs_radii, mu = args.rmean, sig = args.rsigma, new_poly=args.newpoly,
                hsf_niter=args.hsfniter, hsf_stepsize = args.hsfstep, max_iter =args.npackings,
                use_cell_lists=not args.nocell, single=single, ncores=args.ncores, tol=args.tol,
                start_iteration=args.start_iter)

    sim.run()
