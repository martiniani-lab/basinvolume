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
from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import zip
from builtins import range
from builtins import object
from future.utils import with_metaclass

try:
    import numpy as np
    import argparse
    import configparser
    import os
    import re
    import logging
    import matplotlib

    matplotlib.use("Agg")  # matplotlib.use('Agg', warn=False)
    import matplotlib.pyplot as plt
    import traceback
    import copy
    import abc
    import glob
    import multiprocessing as mp
    import pele.utils.fix_multiprocessing
    import gc
    from scipy import integrate
    from basinvolume.utils import (
        to_string,
        ResultsFile,
        MomentsAcc,
        trymakedir,
        OutlierDetection,
    )
    from basinvolume.post_processing import (
        PTFailures,
        assert_pt_success,
        VolumeSanityCheck,
    )
    from basinvolume.spheres import _collect_u2_vs_k
    from basinvolume.mbar_spheres import mbar_compute_dos
except ImportError as err:
    print(err)


class ComputeVolumesCommon(with_metaclass(abc.ABCMeta, object)):
    """
    Contains common functionality of volume computation which is
    independent on config file layout.
    """

    def __init__(
        self,
        workspace_dir,
        nr_volume_points,
        force_run,
        method,
        is_it_gausslobato,
        explore_bv_dir="explore_bv_jammed_packing",
        analysis_dir="analysis",
        volume_file="volume_data",
        packings_dir="packings",
        jammed_packings_dir="jammed_packings",
        volume_title="VOLUME_FULL_PT",
        set_explore_dir=None,
    ):

        # Begin: store input parameters.
        self.workspace_dir = workspace_dir
        self.force_run = force_run
        self.method = method
        self.is_it_gausslobato = is_it_gausslobato
        # End: store input parameters.
        self.experimental = "exp" in workspace_dir  # this is weak, fix it
        self.nr_volume_points = nr_volume_points
        self.explore_bv_dir = explore_bv_dir
        self.analysis_dir = analysis_dir
        self.packings_dir = packings_dir
        self.jammed_packings_dir = jammed_packings_dir
        self.volume_file = volume_file
        self.volume_title = volume_title
        self.set_explore_dir = set_explore_dir
        self.set_up_directories()

    @abc.abstractmethod
    def _compute_volume(self, fname, explore_dir):
        """
        define method to compute volume
        """

    def run_analysis(self):
        """
        replace with approapriate method
        """
        self.pt_failures = PTFailures()
        for (path, fname) in zip(self.explore_dirs, self.packing_strings):
            if not (assert_pt_success(path, fname)):
                # PT runs failed.
                self.pt_failures.add_failure(fname)
            else:
                # PT runs successful.
                self.pt_failures.add_success()
                try:
                    if not self.force_run and os.path.isfile(
                        os.path.join(path, self.analysis_dir, self.volume_file)
                    ):
                        try:
                            volf = configparser.ConfigParser()
                            volf.read(os.path.join(path, self.analysis_dir, self.volume_file))
                            F0 = volf.getfloat(self.volume_title, "F0")
                        except Exception as e:
                            logging.info("run_analysis Exception: {}".format(e))
                            self._compute_volume(fname, path)
                    else:
                        self._compute_volume(fname, path)
                except Exception as e:
                    logging.info("Exception: {}".format(e))
                    logging.info(traceback.format_exc())
                    logging.info("failed packing!")
                    logging.info("name: {}".format(fname))
                    logging.info("path: {}".format(path))
        self.pt_failures.print_failure_info()

    def set_up_directories(self):
        if self.set_explore_dir is not None:
            self.explore_dirs = [self.set_explore_dir]
            self.packing_strings = os.path.basename(self.set_explore_dir)
            # remove the explore_bv_ part
            self.packing_strings = [self.packing_strings[len("explore_bv_") :]]
            return

        self.explore_dirs = [
            os.path.join(self.workspace_dir, f)
            for f in os.listdir(self.workspace_dir)
            if (
                f.startswith(self.explore_bv_dir)
                and os.path.isfile(os.path.join(self.workspace_dir, f, "inner_sphere.timeseries"))
            )
        ]
        if self.nr_volume_points != -1:
            logging.info("removing volume points")
            nr_to_kill = len(self.explore_dirs) - self.nr_volume_points
            for _ in range(nr_to_kill):
                self.explore_dirs = np.delete(
                    self.explore_dirs,
                    np.random.randint(0, len(self.explore_dirs)),
                )
            assert len(self.explore_dirs) == self.nr_volume_points
        self.packing_strings = [
            "jammed_" + (s.split("/")[-1]).split("_")[3] for s in self.explore_dirs
        ]


class ComputeVolumesTINTMultiConfigFile(ComputeVolumesCommon):
    """
    Used for packings with one config file for each packing.
    """

    def __init__(
        self,
        workspace_dir,
        nr_volume_points,
        force_run,
        method,
        is_it_gausslobato,
        volume_file="volume_data",
        volume_title="VOLUME_FULL_PT",
        explore_bv_dir="explore_bv_jammed_packing",
        packings_dir="packings",
        jammed_packings_dir="jammed_packings",
        set_explore_dir=None,
    ):
        super(ComputeVolumesTINTMultiConfigFile, self).__init__(
            workspace_dir,
            nr_volume_points,
            force_run,
            method,
            is_it_gausslobato,
            volume_file=volume_file,
            volume_title=volume_title,
            explore_bv_dir=explore_bv_dir,
            packings_dir=packings_dir,
            jammed_packings_dir=jammed_packings_dir,
            set_explore_dir=set_explore_dir,
        )
        self.series_collector = _collect_u2_vs_k()

    def _compute_volume(self, fname, explore_dir):
        jammed_packings_path = os.path.abspath(
            os.path.join(self.workspace_dir, self.jammed_packings_dir)
        )
        packings_path = os.path.abspath(os.path.join(self.workspace_dir, self.packings_dir))
        print(explore_dir)
        print("jammed_packings_path", jammed_packings_path)
        print("packings_dir", packings_path)
        print(explore_dir)
        self.series_collector(
            frozen=self.experimental,
            fname=fname,
            explore_dir=explore_dir,
            jammed_packings_dir=jammed_packings_path,
            packings_dir=packings_path,
            plot_ts_integrand_data=False,
            simple_integrator=not self.is_it_gausslobato,
        )


class ComputeVolumesMBARMultiConfigFile(ComputeVolumesCommon):
    """
    Used for packings with one config file for each packing.
    """

    def __init__(
        self,
        workspace_dir,
        nr_volume_points,
        force_run,
        method,
        volume_file="mbar_volume_data",
        volume_title="VOLUME_MBAR",
        explore_bv_dir="explore_bv_jammed_packing",
        packings_dir="packings",
        jammed_packings_dir="jammed_packings",
        set_explore_dir=None,
    ):
        super(ComputeVolumesMBARMultiConfigFile, self).__init__(
            workspace_dir,
            nr_volume_points,
            force_run,
            method,
            volume_file=volume_file,
            volume_title=volume_title,
            explore_bv_dir=explore_bv_dir,
            packings_dir=packings_dir,
            jammed_packings_dir=jammed_packings_dir,
            set_explore_dir=set_explore_dir,
        )
        self.plot_dos_data = False
        self.series_collector = mbar_compute_dos(
            nbins=1000,
            bootstrap=True,
            kde=True,
            plot_dos_data=self.plot_dos_data,
            ncores=4,
        )

    def _compute_volume(self, fname, explore_dir):
        jammed_packings_path = os.path.abspath(
            os.path.join(self.workspace_dir, self.jammed_packings_dir)
        )
        packings_path = os.path.abspath(os.path.join(self.workspace_dir, self.packings_dir))
        self.series_collector(
            fname=fname,
            explore_dir=explore_dir,
            packings_dir=packings_path,
            jammed_packings_dir=jammed_packings_path,
            base_dir="analysis",
            frozen=self.experimental,
            show=False,
            verbose=False,
        )

    def run_analysis(self):
        """
        replace with appropriate method (this tests also if the dos calculation was succesfull)
        """
        self.pt_failures = PTFailures()
        for (path, fname) in zip(self.explore_dirs, self.packing_strings):
            if not (assert_pt_success(path, fname)):
                # PT runs failed.
                self.pt_failures.add_failure(fname)
            else:
                # PT runs successful.
                self.pt_failures.add_success()
                try:
                    log_gr_ratio_file = "log_gr_ratio.csv"
                    if (
                        not self.force_run
                        and os.path.isfile(os.path.join(path, self.analysis_dir, self.volume_file))
                        and (
                            not self.plot_dos_data
                            or os.path.isfile(
                                os.path.join(path, self.analysis_dir, log_gr_ratio_file)
                            )
                        )
                    ):
                        try:
                            volf = configparser.ConfigParser()
                            volf.read(os.path.join(path, self.analysis_dir, self.volume_file))
                            F0 = volf.getfloat(self.volume_title, "F0")
                        except Exception as e:
                            logging.info("run_analysis Exception: {}".format(e))
                            self._compute_volume(fname, path)
                    else:
                        self._compute_volume(fname, path)
                except Exception as e:
                    logging.info("Exception: {}".format(e))
                    logging.info(traceback.format_exc())
                    logging.info("failed packing!")
                    logging.info("name: {}".format(fname))
                    logging.info("path: {}".format(path))
        self.pt_failures.print_failure_info()


class ComputeVolumes(object):
    def __init__(
        self,
        workspace_dir,
        nr_volume_points=-1,
        force_run=False,
        method="mbar",
        is_it_gausslobato=False,
        explore_bv_dir="explore_bv_jammed_packing",
        packings_dir="packings",
        jammed_packings_dir="jammed_packings",
        set_explore_dir=None,
    ):
        self.method = method
        self.experimental = "exp" in workspace_dir  # THIS SHOULD BE IMPROVED
        if self.method == "mbar":
            logging.info("Using MBAR method")
            self.computer = ComputeVolumesMBARMultiConfigFile(
                workspace_dir,
                nr_volume_points,
                force_run,
                method,
                explore_bv_dir=explore_bv_dir,
                packings_dir=packings_dir,
                jammed_packings_dir=jammed_packings_dir,
                set_explore_dir=set_explore_dir,
            )
        elif self.method == "tint":
            logging.info("Using thermodynamic integration method")
            if is_it_gausslobato:
                logging.info("Using Gauss-Lobato quadrature")
            else:
                logging.info("Using brute-force trapeze method")
            self.computer = ComputeVolumesTINTMultiConfigFile(
                workspace_dir,
                nr_volume_points,
                force_run,
                method,
                is_it_gausslobato,
                explore_bv_dir=explore_bv_dir,
                packings_dir=packings_dir,
                jammed_packings_dir=jammed_packings_dir,
                set_explore_dir=set_explore_dir,
            )
        else:
            raise Exception("ComputeVolumes: illegal choice of method, " "should be MBAR or TINT")

    def __call__(self):
        self.computer.run_analysis()


def worker(workspace_dir, kwargs):
    try:
        cv = ComputeVolumes(workspace_dir, **kwargs)
        cv()
    except Exception:
        logging.info("find_k worker: %s" % (traceback.format_exc()))


def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute volumes from PT data, use either MBAR or TINT methods"
    )
    parser.add_argument(
        "-d",
        "--workspace_dir",
        type=str,
        help="top-level dir containing the packings, e.g. n32_phi88_2D."
        "If --all is activated a RegEx string matching "
        "all directories. Default: n*phi*phi*D*",
        default="n*phi*phi*D*",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run for all packing subdirectories",
        default=False,
    )
    parser.add_argument(
        "--nr_vpoints",
        type=int,
        default=-1,
        help="number of volume points, by default all " "otherwise select n at random",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="force to recompute volumes for already computed ones",
        default=False,
    )
    parser.add_argument(
        "-j",
        "--ncores",
        type=int,
        help="threads for parallel execution",
        default=4,
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        help="volume computation method",
        default="tint",
    )
    parser.add_argument(
        "-gl",
        "--gauss_lobato",
        action="store_true",
        help="type of thermodynamic integration scheme",
        default=False,
    )
    parser.add_argument(
        "--explore_dirs",
        type=str,
        help="String that packing directories to explore start with. "
        "Default: 'explore_bv_jammed_packing'",
        default="explore_bv_jammed_packing",
    )
    parser.add_argument(
        "--packings_dir",
        type=str,
        help="Directory containing the packings. " "Default: 'packings'",
        default="packings",
    )
    parser.add_argument(
        "--jammed_packings_dir",
        type=str,
        help="Directory containing the jammed packings. " "Default: 'jammed_packings'",
        default="jammed_packings",
    )

    parser.add_argument(
        "--set_explore_dir",
        type=str,
        help="Explore dir to calculate volumes for, overrides all other \
            directory settings. \
            For cases like hypercubes where no packings exist",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    ncores = args.ncores
    kwargs = dict(
        nr_volume_points=args.nr_vpoints,
        force_run=args.force,
        method=args.method,
        is_it_gausslobato=args.gauss_lobato,
        explore_bv_dir=args.explore_dirs,
        packings_dir=args.packings_dir,
        jammed_packings_dir=args.jammed_packings_dir,
        set_explore_dir=args.set_explore_dir,
    )
    print("args.all: {}".format(args.all))
    if not args.all:
        workspace_dir = os.path.abspath(args.workspace_dir)
        worker(workspace_dir, kwargs)
    else:
        subdirs = glob.glob(os.path.join(os.getcwd(), args.workspace_dir))
        logging.info(subdirs)
        if args.ncores > 1 and args.method != "mbar":
            mypool = mp.Pool(ncores)
            try:
                for folder in subdirs:
                    mypool.apply_async(
                        worker,
                        args=(
                            os.path.abspath(folder),
                            kwargs,
                        ),
                    )
            except Exception:
                mypool.terminate()
                mypool.join()
                raise
            mypool.close()
            mypool.join()
        else:
            for folder in subdirs:
                cv = ComputeVolumes(os.path.abspath(folder), **kwargs)
                cv()
                del cv
