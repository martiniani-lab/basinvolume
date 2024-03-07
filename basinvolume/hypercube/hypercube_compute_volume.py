from __future__ import print_function
from future import standard_library
from numpy import inner, loadtxt

standard_library.install_aliases()
from builtins import str
import os
import configparser
import argparse
import glob
import logging
os.environ["JAX_ENABLE_X64"] = "True"
from basinvolume.utils import import_pt_time_series
from basinvolume.mbar_spheres.mbar_compute_volume import mbar_compute_dos

class hypercube_mbar_compute_dos(mbar_compute_dos):
    """
    this is a class that implements _mbar_compute_dos class
    """

    def __init__(
        self,
        nbins=1000,
        bootstrap=False,
        kde=True,
        plot_dos_data=True,
        ncores=7,
        bias = "harmonic",
        method = "mbar",
        bypass_ballpicking_data = False,
        truncate_inner_gaussian = False,
        include_ballpicking_in_plots = False, # Option to include the ballpicking data in plots. Breaks naïve histogram reconstruction
        use_inner_gaussian = True
    ):
        super(hypercube_mbar_compute_dos, self).__init__(
            nbins=nbins,
            bootstrap=bootstrap,
            kde=kde,
            plot_dos_data=plot_dos_data,
            ncores=ncores,
            bias = bias,
            method = method,
            bypass_ballpicking_data=bypass_ballpicking_data,
            truncate_inner_gaussian = truncate_inner_gaussian,
            include_ballpicking_in_plots = include_ballpicking_in_plots,
            use_inner_gaussian = use_inner_gaussian
        )

    def __call__(self, explore_dir, base_dir="analysis", show=False, verbose=True):
        if not os.path.isabs(explore_dir):
            self.explore_dir = os.path.join(os.getcwd(), explore_dir)
        else:
            self.explore_dir = explore_dir
        if self.method == "emus":
            base_dir += "_emus"
        self.base_directory = os.path.join(self.explore_dir, base_dir)

        dlist = explore_dir.split("_")
        assert dlist[2] == "hypercube" or dlist[2] == "hyperball"
        dname = dlist[2] + "_" + dlist[3] + "_" + dlist[4]
        self.pt_configpath = os.path.join(self.explore_dir, "explore_" + dname + ".config")
        assert os.path.isfile(self.pt_configpath)
        self.findk_configpath = os.path.join(self.explore_dir, "findk_" + dname + ".config")
        assert os.path.isfile(self.findk_configpath)
        self.kmax_statuspath = os.path.join(self.explore_dir,"0/status")
        assert os.path.isfile(self.kmax_statuspath)
        self.kmin_configpath = os.path.join(self.explore_dir, "kmin_" + dname + ".config")
        assert os.path.isfile(self.kmin_configpath)
        # There can be several innersphere runs, each with a config path
        self.ballpicking_timeseries_available = False
        self.innersphere_configpaths = []
        self.innersphere_timeseries_paths = (
            []
        )  # It's actually convenient to write down the time series paths as well right here
        innersphere_dir_list = glob.glob(self.explore_dir + "/innersphere_*[!config]")
        if (
            len(innersphere_dir_list) == 0
        ):  # Make this implementation safe to use with the older runs
            self.number_nested_spheres = 1
            innersphere_configpath = os.path.join(
                self.explore_dir, "innersphere_" + dname + ".config"
            )
            assert os.path.isfile(innersphere_configpath)
            self.innersphere_configpaths.append(innersphere_configpath)
            innersphere_timeseries_path = os.path.join(self.explore_dir, "inner_sphere.timeseries")
            assert os.path.isfile(innersphere_timeseries_path)
            self.innersphere_timeseries_paths.append(innersphere_timeseries_path)
            # Also check whether the ballpicking part of innersphere was saved
            ballpicking_timeseries_path = os.path.join(self.explore_dir, "inner_sphere_ballpick.timeseries")
            if os.path.isfile(ballpicking_timeseries_path) and not self.bypass_ballpicking_data:
                self.ballpicking_timeseries_path = ballpicking_timeseries_path
                self.ballpicking_timeseries_available = True
        else:  # If there are actually several innerspheres, go to each directory to extract the path to the config file
            self.number_nested_spheres = len(innersphere_dir_list)
            innersphere_dir_list = sorted(
                innersphere_dir_list, key=lambda x: int(x.split("_")[-1])
            )
            for dir in innersphere_dir_list:
                innersphere_configpath = dir + "/innersphere_" + dname + ".config"
                assert os.path.isfile(innersphere_configpath)
                self.innersphere_configpaths.append(innersphere_configpath)
                innersphere_timeseries_path = dir + "/inner_sphere.timeseries"
                assert os.path.isfile(innersphere_timeseries_path)
                self.innersphere_timeseries_paths.append(innersphere_timeseries_path)
                # Also check whether the ballpicking part of innersphere was saved
                ballpicking_timeseries_path = dir + "/inner_sphere_ballpick.timeseries"
                if os.path.isfile(ballpicking_timeseries_path) and dir == innersphere_dir_list[0] and not self.bypass_ballpicking_data:
                    self.ballpicking_timeseries_path = ballpicking_timeseries_path
                    self.ballpicking_timeseries_available = True

        if not self.use_inner_gaussian:
            # Just set this here so that ballpick can be used even in this case
            self.number_nested_spheres = 0 

        self.show = show
        self.verbose = verbose
        self._import_config_files()
        if not self.bootstrap:
            self.run()
        else:
            self.run_bs()

    def _import_config_files(self):
        configf = configparser.ConfigParser()
        configf.read(str(self.pt_configpath))
        self.adjustf_niter = configf.getint("MCRUNNER", "adjustf_niter")
        configf.read(str(self.findk_configpath))
        self.kmax = configf.getfloat("FINDK", "kmax")
        self.prob_kmax = configf.getfloat("FINDK", "prob")
        # import acceptance of actual kmax run
        file = loadtxt(self.kmax_statuspath)
        self.first_run_acceptance = 1.0 - file[3]
        # There can be several inner spheres: each can come with its own k, radius and acceptance
        self.ks_innersphere = []
        self.inner_gaussian_acceptances = []
        self.ref_radii = []
        self.ref_acceptances = []
        for innersphere_configpath in self.innersphere_configpaths:
            configf.read(str(innersphere_configpath))
            k_innersphere = configf.getfloat("INNERSPHERE_MCRUNNER", "k")
            self.ndof = configf.getfloat("INNERSPHERE_HYPERCUBE", "ndof")
            self.sidelength = configf.getfloat("INNERSPHERE_HYPERCUBE", "sidelength")
            innergaussian_acceptance = configf.getfloat("INNERSPHERE_MCRUNNER_STATUS", "acc_frac")
            ref_radius = configf.getfloat("INNERSPHERE_BALLPICK_MCRUNNER_STATUS", "stepsize")
            ref_acceptance = configf.getfloat("INNERSPHERE_BALLPICK_MCRUNNER_STATUS", "acc_frac")
            self.ks_innersphere.append(k_innersphere)
            self.inner_gaussian_acceptances.append(innergaussian_acceptance)
            self.ref_radii.append(ref_radius)
            self.ref_acceptances.append(ref_acceptance)
        self.nparticles = 1
        self.vcavity = 1

    def _import_pt_time_series(self):
        self.timeseries = import_pt_time_series(
            self.explore_dir,
            self.adjustf_niter,
            max_series_size=int(1e5),
            ncores=self.ncores,
            crop_adjustf_niter=True,
            del_raw=False,
        )

    def _compute_hs_fluid_volume(self, numerical_moments=False):
        self.F0_acc = 0
        self.ideal_gas_F_acc = 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="analyze PT data from thermodynamic integration")
    parser.add_argument("explore_dir", type=str, help="explore_dir")
    parser.add_argument(
        "--show",
        action="store_true",
        help="show plots, default: False",
        default=False,
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="run bootstrap (slow!), default: False",
        default=False,
    )
    parser.add_argument(
        "--kde",
        action="store_true",
        help="use kernel density estimate, default: False",
        default=False,
    )
    parser.add_argument(
        "--ncores",
        action="store_true",
        help="number of cores to use for the calculation",
        default=1,
    )
    parser.add_argument(
        "--bias",
        help = "Biasing potential used in the PT",
        default = "harmonic"
    )
    parser.add_argument(
        "--method",
        help = "Solving method to recombine samples from umbrella sampling, \
            options = mbar, emus; default = mbar",
        default = "mbar"
    )
    parser.add_argument(
        "--bypass_ballpicking_data",
        action="store_true",
        help="Ignore ballpicking timeseries even if it is there\
        used to compare strategies",
        default = False
    )
    parser.add_argument(
        "-t",
        "--truncate_inner_gaussian",
        action="store_true",
        help="Truncate inner gaussian to avoid overflows in the unbiasing\
        used to compare strategies",
        default = False
    )
    parser.add_argument(
        "--discard_inner_gaussian",
        action = "store_true",
        help = "Do not use the inner gaussian run from innersphere at all",
        default = False
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.info(args)
    

    sim = hypercube_mbar_compute_dos(
        bootstrap=args.bootstrap,
        kde=args.kde,
        plot_dos_data=True,
        ncores=1,
        bias = args.bias,
        method = args.method,
        bypass_ballpicking_data = args.bypass_ballpicking_data,
        truncate_inner_gaussian=args.truncate_inner_gaussian,
        use_inner_gaussian = not args.discard_inner_gaussian
    )

    sim(args.explore_dir, show=args.show)
#    else :
#        for subdir, dirs, files in os.walk(wdir):
#            for dir in dirs:
#                if dir is not 'packings' and dir is not 'jammed_packings' and dir is not 'analysis':
#                    path = os.path.join(wdir, dir)
#                    sim(explore_dir=path, frozen=args.frozen)
