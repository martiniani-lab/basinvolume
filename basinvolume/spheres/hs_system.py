from pele.optimize import ModifiedFireCPP
from pele.potentials import Harmonic, HS_WCA
from pele.systems import BaseSystem
import numpy as np



class HSWCASystem(BaseSystem):
    """
    etol: tolerance to calssify eigenvalues, if e<etol the it's a rattler 
    """    
    def __init__(self, eps, sca, hs_radii, boxv, etol=1, bdim=3):
        super(HSWCASystem, self).__init__()
        self.potential = HS_WCA(eps, sca, hs_radii, boxvec=boxv)
        self.bdim=bdim
        self.radii = hs_radii * (1. + sca)
        self.natoms = len(self.radii)
        self.etol=etol
            
    def get_potential(self):
        return self.potential
    
    def get_random_configuration(self):
        return np.random.uniform(-1,1,3*10)
    
#    def get_system_properties(self):

    def get_pgorder(self, coords):
        return 1
    
    def get_metric_tensor(self, coords):
        return None
    
    def get_nzero_modes(self):
        return 3
    
    def get_orthogonalize_to_zero_eigenvectors(self):
        return None
    
    def find_rattlers(self, coords):
        hess = self.potential.getHessian(coords)
        rattlers = np.ones(self.natoms)
        for i in xrange(self.natoms):
            i1 = self.bdim*i
            hess_block = hess[i1:i1+self.bdim,i1:i1+self.bdim]
            w, v = np.linalg.eig(hess_block)
            w = np.real(w)
            if np.any(np.absolute(w) < self.etol):
                rattlers[i] = 0
        return rattlers
                
    
    def draw(self, coordslinear, index):
        from pele.systems._opengl_tools import draw_atomic_binary_polydisperse
#        m = self.database.findMinimum(self.potential.getEnergy(coordslinear), coordslinear)
#        rattlers = m.user_data["rattlers"]
        draw_atomic_binary_polydisperse(coordslinear, index, bdim=self.bdim, subtract_com=True, 
                                        radii=self.radii, Batoms=self.find_rattlers(coordslinear))
        
    
#    def draw(self, coordslinear, index):
#        from pele.systems._opengl_tools import draw_atomic_single_atomtype
#        draw_atomic_single_atomtype(coordslinear, index, subtract_com=True)

def test():
    coords = np.random.uniform(-1,1,3*10)
    system = HSWCASystem()
    db = system.create_database()
    pot = system.get_potential()
    db.addMinimum(pot.getEnergy(coords), coords)
    
    from pele.gui import run_gui
    run_gui(system, db)
    

if __name__ == "__main__":
    test()