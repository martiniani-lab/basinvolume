"""
First part of basin volume analysis.
Compute the volumes of all basins present in the folder.
Do not compute entropy or any further information; this is done in compute_entropy.py.
Usage:
------
To compute volumes for packings in a folder, say, ./n32_phi88_2D
run
python ~/PathToBasinvolume/basinvolume/post_processing/compute_volumes.py -d n32_phi88_2D -m TINT
"""

from __future__ import division
try:
    import numpy as np
    import argparse
    import ConfigParser
    import os
    import re
    import matplotlib.pyplot as plt
    import traceback
    import copy
    import multiprocessing as mp
    import pele.utils.fix_multiprocessing
    from scipy import integrate
    from basinvolume.utils import to_string
    from basinvolume.utils import ResultsFile
    from basinvolume.utils import MomentsAcc, trymakedir
    from basinvolume.post_processing import PTFailures, assert_pt_success
    from basinvolume.spheres import _collect_u2_vs_k
except ImportError as err:
    print err

class ComputeVolumesCommon(object):
    """
    Contains common functionality of volume computation which is
    independent on config file layout.
    """
    def __init__(self, packings_dir, nr_volume_points, force_run, method):
        # Begin: store input parameters.
        self.packings_dir = packings_dir
        self.nr_volume_points = nr_volume_points
        self.force_run = force_run
        self.method = method
        # End: store input parameters.
        self.experimental = "exp" in packings_dir
        self.set_up_directories()
    def run_analysis(self):
        self.compute_integrals_for_F0()
    def set_up_directories(self):
        self.explore_dirs = [os.path.join(self.packings_dir, f) for f in os.listdir(self.packings_dir) if f.startswith("explore_bv_jammed_packing")]
        if self.nr_volume_points != -1:
            print "removing volume points"
            nr_to_kill = len(self.explore_dirs) - self.nr_volume_points
            for _ in xrange(nr_to_kill):
                self.explore_dirs = np.delete(self.explore_dirs, np.random.randint(0, len(self.explore_dirs)))
            assert(len(self.explore_dirs) == self.nr_volume_points)
        self.packing_strings = ["jammed_" + (s.split("/")[-1]).split("_")[3] for s in self.explore_dirs]
    def compute_integrals_for_F0(self):
        series_collector = _collect_u2_vs_k()
        self.pt_failures = PTFailures()
        for (path, fname) in zip(self.explore_dirs, self.packing_strings):
            if not (assert_pt_success(path, fname)):
                # PT runs failed.
                self.pt_failures.add_failure(fname)
            else:
                # PT runs successful.
                self.pt_failures.add_success()
                try:
                    if not self.force_run and os.path.isfile(os.path.join(path, "analysis/volume_data")):
                        try:
                            volf = ConfigParser.ConfigParser()
                            volf.read(str(path + "/analysis/volume_data"))
                            F0 = volf.getfloat('VOLUME_FULL_PT', 'F0')
                        except Exception, e:
                            print "Exception: ", e
                            series_collector(frozen=self.experimental,
                                             fname=fname,
                                             explore_dir=path,
                                             jammed_packings_dir=os.path.abspath(os.path.join(self.packings_dir, "jammed_packings")),
                                             packings_dir=os.path.abspath(os.path.join(self.packings_dir, "packings")),
                                             plot_ts_integrand_data=False)
                    else:
                        series_collector(frozen=self.experimental,
                                             fname=fname,
                                             explore_dir=path,
                                             jammed_packings_dir=os.path.abspath(os.path.join(self.packings_dir, "jammed_packings")),
                                             packings_dir=os.path.abspath(os.path.join(self.packings_dir, "packings")),
                                             plot_ts_integrand_data=False)
                except Exception, e:
                    print "Exception: ", e
                    print "failed packing!"
                    print "name: ", fname
                    print "path:", path
        self.pt_failures.print_failure_info()
    def collect_computed_data_F0(self):
        self.volume_files = [os.path.join(f, "analysis/volume_data") for f in self.explore_dirs]
        self.F0_actually_imported_files = []
        self.F0 = []
        self.unit_box_F0 = []
        self.sigF0 = []
        for vf in self.volume_files:
            self.read_from_volume_file(vf)
    def read_from_volume_file(self, vf):
        volf = ConfigParser.ConfigParser()
        volf.read(str(vf))
        try:
            self.F0.append(volf.getfloat('VOLUME_FULL_PT', 'F0'))
            self.unit_box_F0.append(volf.getfloat('VOLUME_FULL_PT', 'unit_box_F0'))
            self.sigF0.append(volf.getfloat('VOLUME_FULL_PT', 'sigF0'))
            self.F0_actually_imported_files.append(vf)
            try:
                packing_configpath = self.get_packing_configpath(vf)
                print "packing_configpath", packing_configpath
                volume_sanity_check = VolumeSanityCheck(packing_configpath, numerical_moments=self.numerical_moments)
                self.best_integration_selection.check_next_F0(volume_sanity_check, self, vf)
            except Exception, e:
                print "Exception: ", e
                print "integration selection failed"
                print "location:", vf
        except Exception, e:
            print "Exception: ", e
            print "insufficient data available"
            print "location:", vf
    def select_final_dataset(self):
        self.best_integration_selection.print_fail_information(self.packings_dir)
        self.F0_final_integration_selection = self.best_integration_selection.F0_final
        self.outlier_detection = OutlierDetection(self.F0_final_integration_selection, p=0.5, D=3*np.std(self.F0_final_integration_selection), verbose=True)
        self.F0_wo_outliers = np.asarray(self.outlier_detection.non_outliers)
    def process_final_dataset_for_plots(self):
        try:
            self.print_histogram_and_data(self.F0_final_integration_selection, "/volume_histogram_F0_final")
            self.print_histogram_and_data(self.F0_wo_outliers, "/volume_histogram_F0_final_removed_outliers")
        except Exception as err:
            print err

class ComputeVolumesTINTMultiConfigFile(ComputeVolumesCommon):
    """
    Used for packings with one config file for each packing.
    """
    def __init__(self, packings_dir, nr_volume_points, force_run, method):
        super(ComputeVolumesTINTMultiConfigFile, self).__init__(packings_dir, nr_volume_points, force_run, method)
    def get_packing_configpath(self, volume_file):
        tmp = os.path.split(os.path.split(volume_file)[0])[0]
        only_number = int(re.findall('\d+', volume_file)[0])
        return os.path.join(self.packings_dir, "packings", "packing" + only_number + ".config")

class ComputeVolumes(object):
    def __init__(self, packings_dir, nr_volume_points=-1,
                 force_run=False, method="MBAR"):
        self.method = method
        self.experimental = "exp" in packings_dir
        if self.method == "MBAR":
            print("using MBAR method")
        elif self.method == "TINT":
            print("using thermodynamic integration method")
            self.computer = ComputeVolumesTINTMultiConfigFile(packings_dir,
                                nr_volume_points, force_run, method)
            self.computer.run_analysis()
        else:
            raise Exception("ComputeVolumes: illegal choice of method, should be MBAR or TINT")

def worker(packings_dir, kwargs):
    try:
        ComputeVolumes(packings_dir, **kwargs)
    except:
        print('find_k worker: %s' % (traceback.format_exc()))

def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute volumes from PT data, use either MBAR or TINT methods")
    parser.add_argument("-d", "--packings_dir", type=str, help="top-level dir containing the packings, e.g. n32_phi88_2D")
    parser.add_argument("--all", action='store_true', help="run for all packing subdirectories", default=False)
    parser.add_argument("--nr_vpoints", type=int, default=-1, help="number of volume points, by default all otherwise select n at random")
    parser.add_argument("--force", action='store_true', help="force to recompute volumes for already computed ones", default=False)
    parser.add_argument("-j","--ncores", type=int, help="threads for prallel execution", default=4)
    parser.add_argument("-m", "--method", type=str, help="volume computation method", default="TINT")
    args = parser.parse_args()
    
    ncores = args.ncores
    kwargs = dict(nr_volume_points=args.nr_vpoints,
                  force_run=args.force,
                  method=args.method)
    
    if not args.all:
        packings_dir = os.path.abspath(args.packings_dir)
        worker(packings_dir, kwargs)
    else:
        mypool = mp.Pool(ncores)
        subdirs = get_immediate_subdirectories(os.getcwd())
        try:
            for folder in subdirs:
                if folder[1].isdigit() and folder[-1] == "D":
                    mypool.apply_async(worker, args=(os.path.abspath(folder),kwargs,))
        except:
            mypool.terminate()
            mypool.join()
            raise
                    
        mypool.close()
        mypool.join()
