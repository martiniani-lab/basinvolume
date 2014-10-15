from __future__ import division
import numpy as np
import os
from basinvolume.spheres import _Generate_Packing
from basinvolume.utils import trymakedir, volume_nball, get_git_version, get_cython_version, get_python_version
from basinvolume.experiment_2d import Experimental_Packing
import pyvoro
import argparse
            
class HS_Exp_Generate_Packing(_Generate_Packing):
    """
    *DESCRIPTION
    this class imports packings coordinated from images and prints a .xyrf file and a config file for
    each packing after computing the packing fraction
    *PARAMETERS
    *max_iter is the number of desired packings    
    *sca: determines % by which the hs is inflated
    *eps: LJ interaction energy of WCA part of the HS potential, here irrelevant because 'sca' is set to 0
    *hsf stands for hard sphere fluid
    *set seed to something other than none to remove randomness between instances of the class
    *data_set_index: selects the experimental image, it's the number at the end of the data_set image file
    *small packings is a list of Small_Packing_Information objects
    *deflation: factor by which the radius of the hard cores should be deflated
    """    
    def __init__(self, nparticles, bdim=2, boxv=None, max_iter=1, data_set_index=1,
                 distance_from_boundary_x=0.04, distance_from_boundary_y=0.04, frozen_shell_thickness=2, 
                 grid_version=0, grid_all=True, deflation=1, data_file_name="PackingsData_", 
                 data_dir="packingsData"):
        super(HS_Exp_Generate_Packing,self).__init__(nparticles, bdim=bdim, boxv = boxv, 
                                                 packing_frac=0, max_iter = max_iter, 
                                                 use_cell_lists = False)
        
        self.sca = 0. #this must be 0 for hard spheres
        self.deflation = deflation #factor by which diameters should be contracted
        self.mu = 0.
        self.sig = 0.
        assert(self.sca == 0.)
        
        if nparticles <= 0:
            raise Exception('expect finite number of particles')
        if self.max_iter <= 0:
            raise Exception('expect finite number of packings')
        self.data_set_index = data_set_index
        self.nparticles = nparticles
        self.max_iter = max_iter
        self.data_file_name = data_file_name + str(self.data_set_index) + ".dat"
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(os.getcwd(),data_dir)
        self.path_to_datafile = os.path.join(data_dir, self.data_file_name)
        self.distance_from_boundary_x = distance_from_boundary_x
        self.distance_from_boundary_y = distance_from_boundary_y
        self.frozen_shell_thickness = frozen_shell_thickness
        self.grid_version = grid_version
        self.grid_all = grid_all
        self.small_packings = []
        #read experimental data
        self.all_particles = Experimental_Packing(self.path_to_datafile, self.nparticles, 
                                                  self.distance_from_boundary_x, self.distance_from_boundary_y, 
                                                  self.frozen_shell_thickness, self.grid_version, self.grid_all)
        if self.all_particles.grid.nr_of_cells < self.max_iter:
            self.max_iter = self.all_particles.grid.nr_of_cells
        
    def _initialise(self):
        """
        this function needs to import the data and initialise the output
        """
        self._split_packings()
        self._print_initialise()
        self.initialised = True     
    
    def _split_packings(self):
        """
        build list of Small_Packing_Information objects
        """
        for i in xrange(self.max_iter):
            self._find_one_small_packing(i)
            print("found packing %d of %d" % (i+1, self.max_iter))
    
    def _find_one_small_packing(self, index):
        self.small_packings.append(self.all_particles.extract_small_packing(index))
        self.mobile_particle_radius = self.all_particles.mobile_particle_radius
        self.frozen_particle_radius = self.all_particles.frozen_particle_radius
        for i in xrange(self.bdim):
            self.boxv[i] = self.frozen_particle_radius*2.5 # extra 0.5 because the box must fit the whole particle for voro
        print "boxv", self.boxv
    
    def _set_packing_fraction(self):
        """
        compute volume fraction of mobile particles
        """
        vparticle = self._get_particles_volume()
        vcavity = self._get_voronoi_mobile_area()
        assert(0 < vparticle < vcavity)
        self.packing_frac = vparticle/vcavity
        print "phi ",self.packing_frac
    
    def _get_particles_volume(self):
        """returns volume of n=self.bdim dimensional sphere for mobile particles """
        volume = 0.
        for i,frozen in enumerate(self.frozen_idx):
            if not frozen:
                volume += volume_nball(self.hs_radii[i],self.bdim)
        return volume
    
    def _get_voronoi_mobile_area(self):
        """
        Voronoi tesselates the packing and adds up the areas of the mobile particles. This should
        give some decent estimate of the volume fraction for the current packing
        """
        #get coordinates
        coords = self.coords.reshape(-1,self.bdim).tolist()
        #get box limits
        limits = []
        for i in xrange(self.bdim):
            limits.append([-self.boxv[i]/2,self.boxv[i]/2])
        #compute dispersion (max distance between two points that might be adjacent)
        dispersion = np.amax(self.hs_radii) * 2
        #get radii and compute mean and standard deviation
        radii = self.hs_radii.tolist()
        self.mu = np.mean(radii)
        self.sig = np.std(radii)
        #tesselate packing
        if self.bdim == 2:
            cells = pyvoro.compute_2d_voronoi(coords,limits, dispersion, radii=radii)
        elif self.bdim == 3:
            cells = pyvoro.compute_voronoi(coords,limits, dispersion, radii=radii)
        else:
            raise Exception('number of dimensions not allowed')
        assert(len(cells) == int(len(self.coords)/self.bdim))
        #compute free volume
        vcavity = 0.
        vtot = 0.
        for i,cell in enumerate(cells):
            assert(cell['original'] == coords[i])
            vtot += cell['volume']
            if not self.frozen_idx[i]:
                vcavity += cell['volume']
        #test that sum of voronoi areas is within some precision from the exact area
        assert(abs(vtot - np.product(self.boxv)) < 1e-3)
        assert(0 < vcavity < vtot)
        return vcavity
    
#    def _rescale_radii(self):
#        """rescale radii to meet target packing fraction"""
#        vol_box = np.power(self.boxl,self.bdim)
#        vol_part = np.sum(volume_nball(self.hs_radii,self.bdim))
#        phi = vol_part/vol_box
#        self.hs_radii *= np.power(self.packing_frac/phi,1/self.bdim)
#        #test
#        vol_part = np.sum(volume_nball(self.hs_radii,self.bdim))
#        phi = vol_part/vol_box
#        assert(phi - self.packing_frac < 1e-4)
#        #endtest
    
    def _check_overlaps(self):
        """check that no two particles are overlapping (using nearest image convention)"""
        no_overlap = True
        for i in xrange(self.nparticles):
            if no_overlap == True:
                for j in xrange(i, self.nparticles):
                    dij = 0
                    for k in xrange(self.bdim):
                        #use distances to nearest image convention
                        dij += np.square(self.coords[i*self.bdim+k] - self.coords[j*self.bdim+k])
                    if i != j:
                        dij = np.sqrt(dij)
                        dmin = self.hs_radii[i]+self.hs_radii[j]
                        if dij - dmin <= 0:
                            print 'invalid configuration'
                            print 'atoms {} {} are overlapping'.format(i,j)
                            print 'real distance {}'.format(dij)
                            print 'min distance {}'.format(dmin)
                            no_overlap = False
                            break
            else:
                break
        return no_overlap
    
    def _generate_packing_coords(self):
        """
        pick the next packing
        """
        self.coords = []
        packing = self.small_packings[self.iteration]
        self.hs_radii = np.array(packing.d) / (2 * self.deflation);
        self.frozen_idx = np.array(packing.f)
        #align the cell centre to origin
        packing.x -= np.mean(packing.x)
        packing.y -= np.mean(packing.y)
        if self.bdim == 2:
            for particle in zip(packing.x, packing.y):
                self.coords.extend(particle)
        else:
            packing.z -= np.mean(packing.z)
            for particle in zip(packing.x, packing.y, packing.z):
                self.coords.extend(particle)
        self.coords = np.array(self.coords)
        #raise warning if there's an overlap
        if not self._check_overlaps():
            return False
        #compute packing fraction
        self._set_packing_fraction()
        return True
    
    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)    
    
    def _print(self):
        """dump configuration and opengl input to packings directory"""
        self._print_parameters()
        self._dump_configuration()
        self._write_opengl_input()
    
    def _dump_configuration(self):
        """write coordinates to file .xyzdf"""
        coords = self.coords
        directory = self.base_directory
        nparticles = len(self.hs_radii)
        if self.bdim == 2:
            fname = "{0}/packing{1}.xydf".format(directory,self.iteration)
            f = open(fname,'w')
            for i in xrange(nparticles):
                f.write('{:.16f}\t{:.16f}\t{:.16f}\t{}\n'.format(coords[i*self.bdim],coords[i*self.bdim+1],
                                                                  self.hs_radii[i]*2, int(self.frozen_idx[i])))
        else:
            fname = "{0}/packing{1}.xyzdf".format(directory,self.iteration)
            f = open(fname,'w')
            for i in xrange(nparticles):
                f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{}\n'.format(coords[i*self.bdim],coords[i*self.bdim+1],
                                                               coords[i*self.bdim+2],self.hs_radii[i]*2, int(self.frozen_idx[i])))
        f.close()
    
    def _write_opengl_input(self):
        """write opengl input file"""
        coords = self.coords
        boxv = self.boxv
        colour = 13
        directory = self.base_directory
        nparticles = len(self.hs_radii)
        fname = "{0}/packing{1}.dat".format(directory,self.iteration)
        f = open(fname,'w')
        f.write('{}\n'.format(nparticles))
        
        if self.bdim == 2:
            f.write('{} {} {}\n'.format(-boxv[0]/2,-boxv[1]/2, -np.amax(self.hs_radii)))
            f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
            f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
            f.write('0.0 \t 0.0 \t {}\n'.format(np.amax(self.hs_radii)*2))
            for i in xrange(nparticles):
                for j in xrange(self.bdim):
                    f.write('{}\t'.format(coords[i*self.bdim+j]))
                f.write('{}\t'.format(0))
                f.write('{}\t'.format(self.hs_radii[i]*2))
                f.write('{}\n'.format(colour-self.frozen_idx[i]))
        else:
            f.write('{} {} {}\n'.format(-boxv[0]/2,-boxv[1]/2,-boxv[2]/2))
            f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
            f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
            f.write('0.0 \t 0.0 \t {}\n'.format(boxv[2]))
            for i in xrange(nparticles):
                for j in xrange(self.bdim):
                    f.write('{}\t'.format(coords[i*self.bdim+j]))
                f.write('{}\t'.format(self.hs_radii[i]*2))
                f.write('{}\n'.format(colour-self.frozen_idx[i]))
        f.close()
        
    def _print_parameters(self):
        """writes the simulation parameters"""
        fname = '{}/packing{}.config'.format(self.base_directory, self.iteration)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Generate_Packings base class input parameters\n')
        f.write('[PACKING]\n')
        f.write('packing_number: {}\n'.format(self.iteration))
        f.write('method: {}\n'.format("experimental"))
        f.write('nparticles: {}\n'.format(self.nparticles))
        f.write('packing_fraction: {}\n'.format(self.packing_frac))
        f.write('deflation: {}\n'.format(self.deflation))
        f.write('boxdim: {}\n'.format(self.bdim))
        f.write('ndim: {}\n'.format(self.ndof))
        f.write('radii_mean: {}\n'.format(self.mu))
        f.write('radii_stdev: {}\n'.format(self.sig))
        f.write('boxv: ')
        for val in self.boxv:
            f.write('{:.16f} '.format(val))
        f.write('\n')
        f.write('[EXPERIMENTAL_DATA_EXTRACTION]\n')
        f.write('path_data: {}\n'.format(self.path_to_datafile))
        f.write('distance_from_boundary_x: {}\n'.format(self.distance_from_boundary_x))
        f.write('distance_from_boundary_y: {}\n'.format(self.distance_from_boundary_y))
        f.write('frozen_shell_thickness: {}\n'.format(self.frozen_shell_thickness))
        f.write('grid_version: {}\n'.format(self.grid_version))
        f.write('grid_all: {}\n'.format(self.grid_all))
        #print software version
        f.write('[CODEVERSION]\n')
        f.write('basinvolume_version: {}\n'.format(get_git_version('basinvolume')))
        f.write('mcpele_version: {}\n'.format(get_git_version('mcpele')))
        f.write('pele_version: {}\n'.format(get_git_version('pele')))
        f.write('python_version: {}\n'.format(get_python_version()))
        f.write('cython_version: {}\n'.format(get_cython_version()))
        f.close()        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="generate 2/3-D hard disks/spheres packings")
    parser.add_argument("nparticles", type=int, help="number of particles")
    parser.add_argument("-n","--npackings", type=int, default=1, help="number of packings to produce")
    parser.add_argument("-d","--boxdim", type=int, default=3, help="box dimensions")
    parser.add_argument('--deflation', type=float, default=1.12, help='fraction by which particles are deflated')
    parser.add_argument('--dataset_index', type=int, nargs='?', default=1, help='selects experimental dataset')
    parser.add_argument('--datadir',type=str, nargs='?', default="packingsData", help='path to data files')
    parser.add_argument('--distance_from_boundary_x',type=float, nargs='?', default=0.04, help='discarded margins left and right, per-cent')
    parser.add_argument('--distance_from_boundary_y',type=float, nargs='?', default=0.04, help='discarded margins bottom and top, per-cent')
    parser.add_argument('--frozen_shell_thickness',type=float, nargs='?', default=2, help='number of average particle diameters in frozen shell')
    parser.add_argument('--grid_version',type=int, nargs='?', default=0, help='selects type of grid for splitting')
    parser.add_argument('--all', action='store_true', help='extract maximum number of packings')
    parser.add_argument("--datafname", type=str, default="PackingsData_", help="protocol to generate packings")
    args = parser.parse_args()
    print args
        
    sim = HS_Exp_Generate_Packing(args.nparticles, bdim=args.boxdim, max_iter=args.npackings, data_set_index=args.dataset_index,
                                  distance_from_boundary_x=args.distance_from_boundary_x, distance_from_boundary_y=args.distance_from_boundary_y, 
                                  frozen_shell_thickness=args.frozen_shell_thickness, grid_version=args.grid_version, 
                                  grid_all=args.all, deflation=args.deflation, data_file_name=args.datafname, data_dir=args.datadir)
    sim.run()    
                
            
              
                
                
                
