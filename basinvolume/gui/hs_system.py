from pele.systems import BaseSystem
from pele.landscape import smoothPath
from basinvolume.utils import put_in_box
from _hs_wca_smooth_cpp import HS_WCA_Smooth
import numpy as np

class HSWCASystem(BaseSystem):
    """
    etol: tolerance to classify eigenvalues, if e<etol the it's a rattler 
    dtol: rms tolerance on distance between two structures
    """    
    def __init__(self, eps, sca, hs_radii, boxv, dtol=1e-3, etol=1, bdim=3):
        super(HSWCASystem, self).__init__()
        self.potential = HS_WCA_Smooth(eps, sca, hs_radii, boxvec=boxv)
        self.bdim=bdim
        self.eps = eps
        self.sca = sca
        self.boxv = boxv
        self.radii = hs_radii
        self.natoms = len(self.radii)
        self.etol=etol
        self.dtol = dtol
            
        self.set_params(self.params)
    
    def set_params(self, params):
        nebparams = params.double_ended_connect.local_connect_params.NEBparams
        nebparams.adjustk_freq = 10
        nebparams.k = 100000
        nebparams.adaptive_nimages = True
        nebparams.adaptive_niter = True
        nebparams.iter_density = 25
    
    def get_system_properties(self):
        return dict(potential = 'HS WCA smooth',
                    bdim = self.bdim,
                    eps = self.eps,
                    sca = self.sca,
                    boxv = self.boxv,
                    radii = self.radii,
                    etol = self.etol,
                    dtol = self.dtol
                    )
            
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
    
    def get_mindist(self):
        """
        align wrt one particle that is not a rattler in both configurations
        then compute the distance ignoring the rattlers of structure 1
        """
        def mindist(x1, x2):
            rattlers1 = self.find_rattlers(x1)
            rattlers2 = self.find_rattlers(x2)
            #build arrays of non rattlers indices
            rindex1 = np.flatnonzero(rattlers1)
            rindex2 = np.flatnonzero(rattlers2)
            #indeces in rindex1 that are also in rindex2
            mask = np.in1d(rindex1, rindex2)
            #index of first element that is not a rattler in neither configurations
            i = next((i for i, e in enumerate(mask) if e==True), None)
            i1 = rindex1[i]
            #align x2
            dx = x2[i1*self.bdim:(i1+1)*self.bdim] - x1[i1*self.bdim:(i1+1)*self.bdim]
            alg_x2 = np.subtract(np.reshape(x2, (-1,self.bdim)), dx).flatten()
            #compute the distance ignoring the rattlers of structure 1
            dist = np.reshape((x1-alg_x2),(-1,self.bdim)) * np.reshape(rattlers1,(-1,1))
            dist = np.linalg.norm(dist.flatten())
            return dist, x1, alg_x2
        return mindist
    
    def get_compare_exact(self, **kwargs):
        """this function quickly determines whether two clusters are identical
        given translational symmetries
        """
        mindist = self.get_mindist()
        return lambda x1, x2: mindist(x1, x2)[0]/np.sqrt(self.natoms) < self.dtol

    def smooth_path(self, path, **kwargs):
        mindist = self.get_mindist()
        return smoothPath(path, mindist, **kwargs)
        

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
        #put_in_box(coordslinear, self.boxv)
        draw_atomic_binary_polydisperse(coordslinear, index, bdim=self.bdim, subtract_com=True, 
                                        radii=self.radii*(1+self.sca), Batoms=self.find_rattlers(coordslinear))
        
    
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