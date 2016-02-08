from __future__ import division

import numpy as np
import os

from pele.optimize import ModifiedFireCPP
from pele.potential import LJCut

from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import UniformCubicSampling
from mcpele.monte_carlo import NullPotential

from basinvolume.monte_carlo import CheckSameMinimum

#    def _initialise_coords_hcp_lattice_3d(self):
#        """
#        Put spheres in FCC lattice.
#        See e.g. here: Frenkel and Smit: Understanding Molecular Simulation, page 252
#        http://micro.stanford.edu/wiki/M02_Making_a_Perfect_Crystal
#        """
#        n = int(np.power(self.nparticles/4,1./self.bdim))
#        assert ( n - np.power(int(n),self.bdim)) < 1e-8, "Nparticles is not N^3/4"
#        #assuming that box is cubic
#        L_cube = int((self.nparticles/4) ** (1/3))
#        NX = L_cube
#        NY = L_cube
#        NZ = L_cube
#        print L_cube
#        dx = self.boxv[0] / NX
#        dy = self.boxv[1] / NY
#        dz = self.boxv[2] / NZ
#        d = [dx, dy, dz]
#        if np.amax(self.hs_radii) > np.amax(d):
#            raise Exception("_generate_packing_coords_lattice_3d: spheres can not be placed on lattice")
#        coords=[]
#        for iz in xrange(NZ):
#            for iy in xrange(NY):
#                for ix in xrange(NX):
#                    coords.extend([ix*d[0],iy*d[1],iz*d[2]])
#                    coords.extend([(ix+0.5)*d[0],(iy+0.5)*d[1],iz*d[2]])
#                    coords.extend([(ix+0.5),(iy+1./6)*d[1],(iz+0.5)*d[2]])
#                    coords.extend([ix*d[0],(iy+2/3)*d[1],(iz+0.5)*d[2]])
#        self.coords = np.array(coords)

#https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres
#    def _initialise_coords_hcp_lattice_3d(self):
#        L_cube = int((self.nparticles/4) ** (1/3))
#        NX = L_cube
#        NY = L_cube
#        NZ = L_cube
#        print L_cube
#        a1 = (np.prod(self.boxv) / (NX * NY * NZ)) ** (1/3)
#        for iz in xrange(NZ):
#            for iy in xrange(NY):
#                for ix in xrange(NX):
#                    i = (ix + iy*NX + iz*NX*NY)*self.bdim 
#                    self.coords[i] = (2*ix+((iy+iz)%2))*a1
#                    self.coords[i + 1] = (np.sqrt(3)*(iy+(iz%2)/3))*a1
#                    self.coords[i + 2] = (2*np.sqrt(6)*iz/3)*a1

class MC(_BaseMCRunner):
    def set_control(self, tmp):
        self.set_temperature(tmp)
            
class BruteComptuer(object):
    def __init__(self, common_pars):
        self.common_pars = common_pars
        self.boxvec = np.asarray([1, 1, 1])
        self.optimizer_potential = LJCut(boxvec=self.boxvec)
        self.optimizer = 
        self.conftest_check_minimum_is_hcp = 
        self.mc_potential = NullPotential()
        self.mc = 
        self.step = 
        self.mc.set_takestep(self.step)
        self.mc.add_conf_test(self.conftest_check_same_minimum)
        
    def run_bv(self):
        self.mc.run()
        p = self.mc.get_accepted_fraction()
        self.volume = np.exp(np.log(p) + self.common_pars["log_accessible_volume"])

if __name__ == "__main__":
    common_pars = dict([("nr_samples", int(1e5)),
        ("nr_particles", 16), ("log_accessible_volume", 42 )])
    c = BruteComputer(common_pars)
    c.run_bv()
    print("common_pars", common_pars)
    print("volume", c.volume)
