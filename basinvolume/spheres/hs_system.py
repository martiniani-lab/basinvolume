from pele.optimize import ModifiedFireCPP
from pele.potentials import Harmonic, HS_WCA
from pele.systems import BaseSystem
import numpy as np



class HSWCASystem(BaseSystem):
    
    def __init__(self, eps, sca, hs_radii, boxv):
        super(HSWCASystem, self).__init__()
        self.potential = HS_WCA(eps, sca, hs_radii, boxvec=boxv)
        
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
    
    def draw(self, coordslinear, index):
        from pele.systems._opengl_tools import draw_atomic_single_atomtype
        draw_atomic_single_atomtype(coordslinear, index, subtract_com=True)

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