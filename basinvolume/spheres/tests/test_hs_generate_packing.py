from __future__ import division
import numpy as np
from basinvolume.spheres import HS_Generate_Packing
import unittest
import logging

class Test_HS_Generate_Packing(unittest.TestCase):
    
    def setUp(self):
        self.seeds = dict(seed_takestep=42, seed_generate_packing=43)
        self.nparticles = 10
        self.bdim = 3
        self.packing_frac=0.4
        self.sig = 0.2
        self.hsf_niter = 1e5
        self.hs_radii = None #np.random.normal(1,self.sig,self.nparticles)
        self.gp_nocell = HS_Generate_Packing(self.nparticles, method='quench', bdim=self.bdim, boxv=None, 
                                             packing_frac=self.packing_frac, hs_radii=self.hs_radii, mu = 1, sig = self.sig, 
                                             hsf_niter=self.hsf_niter, hsf_stepsize = 1e-3,
                                             max_iter = 10, use_cell_lists=False, seeds=self.seeds)
        
        self.gp_cell = HS_Generate_Packing(self.nparticles, method='quench', bdim=self.bdim, boxv=None, 
                                           packing_frac=self.packing_frac, hs_radii=self.hs_radii, mu = 1, sig = self.sig, 
                                           hsf_niter=self.hsf_niter, hsf_stepsize = 1e-3, 
                                           max_iter = 10, use_cell_lists=True, seeds=self.seeds)
    
    def test_seed_initialise_nocell(self):
        """
        test that seeding works correctly and gives identical results up to initialisation without cell lists
        """    
        gp = HS_Generate_Packing(self.nparticles, method='quench', bdim=self.bdim, boxv=None, 
                                         packing_frac=self.packing_frac, hs_radii=self.hs_radii, mu = 1, sig = self.sig, 
                                         hsf_niter=self.hsf_niter, hsf_stepsize = 1e-3, 
                                         max_iter = 1, use_cell_lists=False, seeds=self.seeds)
        gp._initialise()
        coords_nocell1 = gp.coords  
        gp = HS_Generate_Packing(self.nparticles, method='quench', bdim=self.bdim, boxv=None, 
                                         packing_frac=self.packing_frac, hs_radii=self.hs_radii, mu = 1, sig = self.sig, 
                                         hsf_niter=self.hsf_niter, hsf_stepsize = 1e-3, 
                                         max_iter = 1, use_cell_lists=False, seeds=self.seeds)
        gp._initialise()
        coords_nocell2 = gp.coords
        self.assertTrue(np.array_equal(coords_nocell1, coords_nocell2))
    
    def test_seed_initialise_cell(self):    
        """
        test that seeding works correctly and gives identical results up to initialisation with cell lists
        """
        gp = HS_Generate_Packing(self.nparticles, method='quench', bdim=self.bdim, boxv=None, 
                                         packing_frac=self.packing_frac, hs_radii=self.hs_radii, mu = 1, sig = self.sig, 
                                         hsf_niter=self.hsf_niter, hsf_stepsize = 1e-3, 
                                         max_iter = 1, use_cell_lists=True, seeds=self.seeds)
        gp._initialise()
        coords_cell1 = gp.coords  
        gp = HS_Generate_Packing(self.nparticles, method='quench', bdim=self.bdim, boxv=None, 
                                         packing_frac=self.packing_frac, hs_radii=self.hs_radii, mu = 1, sig = self.sig, 
                                         hsf_niter=self.hsf_niter, hsf_stepsize = 1e-3, 
                                         max_iter = 1, use_cell_lists=True, seeds=self.seeds)
        gp._initialise()
        coords_cell2 = gp.coords
        self.assertTrue(np.array_equal(coords_cell1, coords_cell2))
    
    def test_seed_initialise(self):
        """
        test that seeding works correctly and gives identical results up to initialisation without and without cell lists
        """
        self.gp_nocell._initialise()
        self.gp_cell._initialise()
        self.assertTrue(np.array_equal(self.gp_nocell.coords, self.gp_cell.coords))
    
    def test_seed_generate_packing_nocell(self):
        """
        test that seeding works correctly and HS_Generate_Packings gives identical packings without cell lists
        """
        gp = HS_Generate_Packing(self.nparticles, method='quench', bdim=self.bdim, boxv=None, 
                                         packing_frac=self.packing_frac, hs_radii=self.hs_radii, mu = 1, sig = self.sig, 
                                         hsf_niter=self.hsf_niter, hsf_stepsize = 1e-3, 
                                         max_iter = 1, use_cell_lists=False, seeds=self.seeds)
        gp._initialise()
        gp._generate_packing_coords()
        coords_nocell1 = gp.coords  
        gp = HS_Generate_Packing(self.nparticles, method='quench', bdim=self.bdim, boxv=None, 
                                         packing_frac=self.packing_frac, hs_radii=self.hs_radii, mu = 1, sig = self.sig, 
                                         hsf_niter=self.hsf_niter, hsf_stepsize = 1e-3, 
                                         max_iter = 1, use_cell_lists=False, seeds=self.seeds)
        gp._initialise()
        gp._generate_packing_coords()
        coords_nocell2 = gp.coords
        self.assertTrue(np.array_equal(coords_nocell1, coords_nocell2))
    
    def test_seed_generate_packing_cell(self):
        """
        test that seeding works correctly and HS_Generate_Packings gives identical packings with cell lists
        """  
        gp = HS_Generate_Packing(self.nparticles, method='quench', bdim=self.bdim, boxv=None, 
                                         packing_frac=self.packing_frac, hs_radii=self.hs_radii, mu = 1, sig = self.sig, 
                                         hsf_niter=self.hsf_niter, hsf_stepsize = 1e-3, 
                                         max_iter = 1, use_cell_lists=True, seeds=self.seeds)
        gp._initialise()
        gp._generate_packing_coords()
        coords_nocell1 = gp.coords  
        gp = HS_Generate_Packing(self.nparticles, method='quench', bdim=self.bdim, boxv=None, 
                                         packing_frac=self.packing_frac, hs_radii=self.hs_radii, mu = 1, sig = self.sig, 
                                         hsf_niter=self.hsf_niter, hsf_stepsize = 1e-3, 
                                         max_iter = 1, use_cell_lists=True, seeds=self.seeds)
        gp._initialise()
        gp._generate_packing_coords()
        coords_nocell2 = gp.coords
        self.assertTrue(np.array_equal(coords_nocell1, coords_nocell2))
    
    def test_cell_iter(self):
        """
        test that HS_Generate_Packings gives identical packings with and without cell lists
        """
        self.gp_nocell._initialise()
        self.gp_nocell._generate_packing_coords()
        
        self.gp_cell._initialise()
        self.gp_cell._generate_packing_coords()
        
        self.assertTrue(np.array_equal(self.gp_nocell.coords, self.gp_cell.coords))
    
    def test_cell_run(self):
        """
        test that HS_Generate_Packings run gives identical packings with and without cell lists
        """
        self.gp_cell.run()
        self.gp_nocell.run()
        self.assertTrue(np.array_equal(self.gp_nocell.coords, self.gp_cell.coords))
        
if __name__ == "__main__":
    logging.basicConfig(filename='Test_HS_Generate_Packing.log',level=logging.DEBUG)
    unittest.main()
    