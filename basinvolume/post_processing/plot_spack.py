import numpy as np
import os
from spack import Packing
from basinvolume.post_processing import StructuralAnalysis
from basinvolume.spheres.generate_jammed_packing import import_jammed_packing_config

class GenerateSpackPlot(StructuralAnalysis):
    """
    fname jammed_packing1.xydr
    workspace n32_phi...
    """
    def __init__(self, workspace, packings_dir='packings', jammed_packings_dir='jammed_packings'):
        super(GenerateSpackPlot, self).__init__(workspace, packings_dir=packings_dir, jammed_packings_dir=jammed_packings_dir,
                                                    analysis_dir=None)

    def create_image(self, fname, size=1000, cmap=None, glass=True):
        """
        """
        import matplotlib.pyplot as plt
        assert 'xyzd' in fname or 'xyd' in fname
        dname = self._get_dname(fname)
        jammed_packing_configpath = os.path.join(self.jammed_packings_dir, dname + '.config')
        print jammed_packing_configpath
        import_jammed_packing_config(self, jammed_packing_configpath)
        hs_coords, hs_radii, ss_radii, rattlers = self._import_packing_configuration(fname)
        hs_coords = np.reshape(hs_coords, (-1,self.bdim))

        # #packing
        pack = Packing(hs_coords, hs_radii*2, L=self.boxv[0])
        # sc = pack.scene(rot=np.pi/2, camera_dist=4, camera_height=0, cmap=cmap, lightstrength=1.1)
        # print os.path.join(os.getcwd(),'{}.png'.format(dname[7:]+'marble'))
        # sc.render(os.path.join(os.getcwd(),'{}.png'.format(dname[7:]+'marble')), width=size, height=size, antialiasing=0.000001)
        pack.plot_disks()

        #jammed packing
        pack = Packing(hs_coords, ss_radii*2, L=self.boxv[0])
        # sc = pack.scene(rot=np.pi/2, camera_dist=1000, camera_height=0, cmap=cmap,
        #                 lightstrength=1.1)
        # print os.path.join(os.getcwd(),'{}.png'.format(dname+'marble'))
        # sc.render(os.path.join(os.getcwd(),'{}.png'.format(dname+'marble')), width=size, height=size, antialiasing=0.000001)
        pack.plot_disks()

        # pack.plot_contacts(reshape=True, tol=1e-9)
        plt.savefig('packing_2d.pdf')


if __name__ == "__main__":
    gsp = GenerateSpackPlot(os.getcwd())
    gsp.create_image('jammed_packing564.xydr')
