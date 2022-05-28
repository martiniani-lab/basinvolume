import numpy as np
from basinvolume.mbar_spheres import mbar_compute_dos
import os
import ConfigParser
import argparse
from basinvolume.utils import import_pt_time_series

class hyperelem_mbar_compute_dos(mbar_compute_dos):
    """
    this is a class that implements _mbar_compute_dos class 
    """
    def __init__(self, nbins=1000, bootstrap=False, kde=True, plot_dos_data=True, ncores=7):
        super(hyperelem_mbar_compute_dos, self).__init__(nbins=nbins, bootstrap=bootstrap,
                                                   kde=kde, plot_dos_data=plot_dos_data, 
                                                   ncores=ncores)
    
    def __call__(self, explore_dir, base_dir='analysis', show=False, verbose=True):
        assert 'hyper' in explore_dir
        if not os.path.isabs(explore_dir):
            self.explore_dir = os.path.join(os.getcwd(), explore_dir)
        else:
            self.explore_dir = explore_dir
        self.base_directory = os.path.join(self.explore_dir, base_dir)

        base_name = os.path.basename(os.path.normpath(self.explore_dir))
        dname = str(base_name).replace('explore_bv_', '')
        self.pt_configpath = os.path.join(self.explore_dir, 'explore_' + dname + '.config')
        assert os.path.isfile(self.pt_configpath)
        self.findk_configpath = os.path.join(self.explore_dir,'findk_'+dname+'.config')
        assert os.path.isfile(self.findk_configpath)
        self.kmin_configpath = os.path.join(self.explore_dir,'kmin_'+dname+'.config')
        assert os.path.isfile(self.kmin_configpath)
        self.innersphere_configpath = os.path.join(self.explore_dir, 'innersphere_' + dname + '.config')
        assert os.path.isfile(self.innersphere_configpath)
        
        self.show = show
        self.verbose = verbose
        self._import_config_files()
        if not self.bootstrap:
            self.run()
        else:
            self.run_bs()
    
    def _import_config_files(self):
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.pt_configpath))
        self.report_steps = configf.getfloat('MCRUNNER', 'report_steps')
        configf.read(str(self.findk_configpath))
        self.kmax = configf.getfloat('FINDK', 'kmax')
        self.prob_kmax = configf.getfloat('FINDK', 'prob') 
        configf.read(str(self.innersphere_configpath))
        self.k_innersphere = configf.getfloat('INNERSPHERE_MCRUNNER', 'k')
        self.ndof = configf.getfloat('INNERSPHERE_HYPERELEM', 'ndof')
        self.geometry = configf.get('INNERSPHERE_HYPERELEM', 'geometry')
        geom_params = configf.get('INNERSPHERE_HYPERELEM', 'geom_params')
        self.geom_params = np.array([float(x) for x in geom_params.split()])
        self.nparticles = 1
        self.vcavity = 1
        self.ref_radius = configf.getfloat('INNERSPHERE_BALLPICK_MCRUNNER_STATUS', 'stepsize')
        self.ref_acceptance = configf.getfloat('INNERSPHERE_BALLPICK_MCRUNNER_STATUS', 'acc_frac')
    
    def _import_pt_time_series(self):
        self.timeseries = import_pt_time_series(self.explore_dir, self.report_steps,
                                                max_series_size=int(1e5), ncores=self.ncores, 
                                                crop_report_steps=True, del_raw=False)
    
    def _compute_hs_fluid_volume(self, numerical_moments=False):
        self.F0_acc = 0
        self.ideal_gas_F_acc = 0

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="analyze PT data from thermodynamic integration")
    parser.add_argument("explore_dir", type=str, help="explore_dir")
    parser.add_argument("--show", action='store_true', help="show plots, default: False", default=False)
    parser.add_argument("--bootstrap", action='store_true', help="run bootstrap (slow!), default: False", default=False)
    parser.add_argument("--kde", action='store_true', help="use kernel density estimate, default: False", default=False)
    parser.add_argument("-j", "--ncores", type=int, help="number of cores", default=7)
    args = parser.parse_args()
    print args
    
    sim = hyperelem_mbar_compute_dos(bootstrap=args.bootstrap, kde=args.kde, plot_dos_data=True, ncores=args.ncores)
    
    sim(args.explore_dir, show=args.show)
#    else :
#        for subdir, dirs, files in os.walk(wdir):
#            for dir in dirs:
#                if dir is not 'packings' and dir is not 'jammed_packings' and dir is not 'analysis':
#                    path = os.path.join(wdir, dir)
#                    sim(explore_dir=path, frozen=args.frozen)
        