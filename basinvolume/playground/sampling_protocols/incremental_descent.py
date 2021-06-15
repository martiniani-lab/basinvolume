from __future__ import division
from __future__ import print_function
from builtins import range
import numpy as np
import os
from pele.potentials import HS_WCA
from pele.optimize._quench import modifiedfire_cpp
from pele.distance import Distance
from basinvolume.utils import import_packing
from basinvolume.spheres import HS_Generate_Jammed_Packing
import re
import argparse
try:
    import pylab
except:
    pass

class Incremental_Generate_Jammed_Packing(HS_Generate_Jammed_Packing):
    """
    *this class generates packings and identifies rattlers by computing the hessian eigenvalues for each particle
    *in the equilibrium jammed structure. A .xyzdr file is produced that contains the 3 system coordinates, the particle
    * diameter and if not it's a rattler (0 if a rattler, 1 otherwise)
    *PARAMETERS
    *hs_radii: array with the radii of the particles, if none sample particle sizes from a normal distribution
    *mu: average particle size, passable to normal distribution
    *sig: standard deviaton of normal distribution from which to sample particles
    *sca: determines % by which the hs is inflated
    *eps: LJ interaction energy of WCA part of the HS potential
    """
    def __init__(self, target_packing_frac=0.7, nincrements=1, rattler_eval_tol=1.,packings_dir='packings', use_cell_lists=False, show=False):
        super(Incremental_Generate_Jammed_Packing,self).__init__(
            target_packing_frac=target_packing_frac, rattler_eval_tol=rattler_eval_tol,
            packings_dir=packings_dir, use_cell_lists=use_cell_lists, show=show)
        self.nincrements = nincrements
        print("nincrements", self.nincrements)

    def one_iteration(self, fname, rd=3.0):
        """perform one iteration
        """
        self._import_single_packing_config_file(fname)
        self.rattlers = np.empty(self.nparticles,dtype='d')
        self.rattlers_draw = np.empty(self.nparticles,dtype='d')

        self._import_packing_configuration(fname)
        self.max_nrattlers = int(self.nparticles*0.1)

        #assert that largest soft particle is not > 1/2 of smallest box size
        if np.amax(self.hs_radii) * 2 * (1 + self.sca) >= np.amin(self.boxv) / 2:
            print("WARNING: max soft diameter >= 1/2 box side!")
        if np.amax(self.hs_radii) * 2 * (1 + self.sca) >= np.amin(self.boxv):
            raise Exception("WARNING: particle does not fit the box")

        #test change radius
        self.hs_radii /= rd #test half the radius
        self.packing_frac = self._get_particles_volume()/np.prod(self.boxv)

        ###potential needs to be called because self.coords is an input argument of HS_WCAPeriodicCellLists
        phi_increments = np.linspace(self.packing_frac, self.target_packing_frac, self.nincrements+1)[1:]
        for i, phi in enumerate(phi_increments):
            new_phi = phi
            self._compute_sca(phi)
            rcut = np.amax(self.hs_radii) * 2.0 * (1.0 + self.sca) #rcut set to largest particle diameter
            if self.use_cell_lists:
                if np.amin(self.boxv) // rcut <= 3:
                    self.use_cell_lists = False
            if self.use_cell_lists:
                self.potential = HS_WCA(distance_method=Distance.PERIODIC, use_cell_lists=True,
                                        eps=self.eps, sca=self.sca, radii=self.hs_radii,
                                        boxvec=self.boxv, reference_coords=self.coords,
                                        ndim=self.bdim, ncellx_scale=1.0)
            else:
                self.potential = HS_WCA(distance_method=Distance.PERIODIC, eps=self.eps, sca=self.sca,
                                        radii=self.hs_radii, boxvec=self.boxv, ndim=self.bdim)

            success = self._generate_packing_coords(i) #returns false if saddle
            if not success:
                break

        #test change radius back
        self.hs_radii *= rd #test half the radius
        self.packing_frac = self._get_particles_volume()/np.prod(self.boxv)
        self._compute_sca(new_phi)
        success = self._generate_packing_coords(0) #returns false if saddle

        if success:
            self._find_rattlers()
            #strips the integer unique identifier out of fname
            n = int(re.search(r'\d+',fname).group())
            self._print(n)

        self.iteration+=1

    def _import_packing_configuration(self, fname):
        path = os.path.join(self.packings_dir, fname)
        packing = import_packing(path, False, self.bdim)
        self.coords = packing['coords']
        self.hs_radii = packing['hs_radii']

    def _compute_sca(self, packing_frac):
        ##test##
        vol_part = self._get_particles_volume()
        vol_box = np.prod(self.boxv)
        phi = vol_part/vol_box #instanteneous pack frac
        assert(phi - self.packing_frac < 1e-4)
        ##endtest##
        ###r_soft = r_hs*(1+sca)
        self.sca = np.power(packing_frac/self.packing_frac,1./self.bdim) - 1

    def _generate_packing_coords(self, phi_iteration):
        """
        perform quench and run tests
        """
        success = self._generate_packing_coords_iteration(tol=1e-7, analyse=(phi_iteration==self.nincrements-1))
        return success

    def _generate_packing_coords_iteration(self, tol=1e-7, iprint=-1, analyse=False):
        """quenches the imported structure using FIRE"""
        fire_maxstep = np.amin(self.hs_radii)*self.sca
        res = modifiedfire_cpp(self.coords, self.potential, maxstep=fire_maxstep, nsteps=1e6, tol=tol, iprint=iprint)
        if not res.success:
            print('quench failed')
            return False

        self.coords = res.coords
        self.energy = res.energy

        #asserts that none of the hard sphere is overlapping
        no_overlap = self._check_no_overlaps()
        if not no_overlap:
            print('overlap found')
            return False

        if analyse:
            #test that on ri-minimisation the structure does not change
            res2 = modifiedfire_cpp(self.coords, self.potential, maxstep=fire_maxstep, nsteps=1e6, tol=tol)
            if res2.nfev > 1:
                print('quench failed (structure changed at second minimisation)')
                return False

            #analyse packing, assert that the whole system has only 3 0'evalues + a 0 evalue for each rattler 0 evalue
            hess = self.potential.getHessian(self.coords)
            ratt0evals= []
            nratls = 0
            for i in range(self.nparticles):
                i1 = self.bdim*i
                hess_block = hess[i1:i1+self.bdim,i1:i1+self.bdim]
                w, v = np.linalg.eig(hess_block)
                w = np.real(w)
                if np.any(w < self.rattler_eval_tol):
                    nratls += 1
                ratt0evals.extend([x for x in w if abs(x) < self.rattler_eval_tol]) #append to array of zero evalues due to rattlers
                self.block_evalues.extend(w)

            w, v = np.linalg.eig(hess)
            w = np.real(w)
            full0evals = [x for x in w if abs(x) < 1e-6]
            if len(full0evals) - len(ratt0evals) > self.bdim:
                print('hessian 0s mismatch rattlers 0s')
                return False
            self.whole_evalues.extend(w)

            print("nrattlers: {}".format(nratls))
            if nratls > self.max_nrattlers:
                print('{} rattlers constitute more than 10% of the system'.format(nratls))
                return False

        return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="generate 2/3-D hard disks/spheres packings")
    parser.add_argument("-p","--density", type=float, help="target packing fraction",default=0.7)
    parser.add_argument("-e","--etol", type=float, help="tolerance on particles eigenvalues, if eval < etol particle will be considered a rattler",default=1.0)
    parser.add_argument("--nocell", action='store_true', help="don't use cell lists, default: False",default=False)
    parser.add_argument("--packingsdir", type=str, help="name of directory with packings, must be in cwd", default="packings")
    parser.add_argument("--show", action='store_true', help="show histograms", default=False)

    args = parser.parse_args()
    print(args)

    sim = Incremental_Generate_Jammed_Packing(target_packing_frac=args.density, rattler_eval_tol=args.etol, packings_dir=args.packingsdir,
                                              use_cell_lists=not args.nocell, show=args.show)
    sim.run()
