from __future__ import division
import matplotlib
matplotlib.use('Agg')
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
import ConfigParser
import pickle
from basinvolume.utils import BasicPlot, sort_pair, log_volume_nball
from basinvolume.monte_carlo import CheckExponentiallyDecayingProfile, CheckPowerDecayingProfile
from basinvolume.utils import import_pt_cloud_drops_time_series_raw_single, compute_mean_var_cloud_ts
from basinvolume.utils import Bunch

matplotlib.rcParams['text.usetex']=False

allowed_systems = ["sphere", "sphere_exp_decay", "sphere_pow_decay", "cube", "cube_exp_decay"]

def plot_numerical(system_str, workspace=None, listdir="cloud_radius*_points*"):
    assert system_str in allowed_systems, "{}: not implemented".format(system_str)
    if workspace is None:
        workspace = os.getcwd()
    listdir = [d for d in glob.glob(listdir) if os.path.isdir(d)]
    p = BasicPlot(out_name=system_str+"_plot.pdf")
    p2 = BasicPlot(out_name=system_str+"_meanr2_plot.pdf")
    ndim_min, ndim_max = 2, 2
    for dir in listdir:
        # tmp = re.findall("[-+]?\d*\.\d+|\d+", dir)
        # cloud_radius, cloud_points = float(tmp[0]), float(tmp[1])
        if not os.path.isfile(os.path.join(workspace, dir + ".pickle")):
            listdir2 = glob.glob(os.path.join(dir, "explore_bv_*"))
            tmp = re.findall("[-+]?\d*\.\d+|\d+", os.path.basename(os.path.normpath(dir)))
            print tmp
            b = Bunch()
            b.ndim_list = []
            b.f_list = []
            b.ferr_list = []
            b.mean_r2_list = []
            b.std_err_r2_list = []
            b.var_r2_list = []
            b.geom_params_list = []
            b.system_str = system_str
            b.pt = True if "pt" in dir else False
            b.cloud_points = tmp[1]
            for dir2 in listdir2:
                tmp = re.findall("[-+]?\d*\.\d+|\d+", os.path.basename(os.path.normpath(dir2)))
                ndim = int(tmp[0])
                geom_params = [float(x) for x in tmp[1:]]
                b.geom_params_list.append(geom_params)
                configf = ConfigParser.ConfigParser()
                try:
                    configf.read(os.path.join(workspace, dir2, 'analysis/mbar_volume_data'))
                    f = configf.getfloat('VOLUME_MBAR', 'F0')
                    ferr = configf.getfloat('VOLUME_MBAR', 'sigF0')
                    print dir, ndim, f, ferr
                    b.ndim_list.append(ndim)
                    b.f_list.append(f)
                    b.ferr_list.append(ferr)
                except Exception, e:
                    print "volume not found: {}".format(dir2)
                try:
                    with open(os.path.join(workspace, dir2, '14/hist_mean'), "r") as fileHandle:
                        lineList = fileHandle.readlines()
                        std_err = float(lineList[-1].split()[3])
                    path = os.path.join(workspace, dir2, '14')
                    timeseries = import_pt_cloud_drops_time_series_raw_single(path, ncores=2)
                    mean, var = compute_mean_var_cloud_ts([timeseries[len(timeseries) / 2:] ** 2], [0])
                    b.mean_r2_list.append(mean)
                    b.var_r2_list.append(var)
                    b.std_err_r2_list.append(std_err)
                except Exception, e:
                    print e
                    print "<r2> not found: {}".format(dir2)
            pickle.dump(b, open(os.path.join(workspace, dir + ".pickle"), 'wb'), protocol=-1)
        else:
            b = pickle.load(open(os.path.join(workspace, dir + ".pickle"), 'r'))
        if len(b.ndim_list) > 0:
            ndim_copy, b.f_list = sort_pair(b.ndim_list, b.f_list, reverse=False)
            ndim_copy, b.ferr_list = sort_pair(b.ndim_list, b.ferr_list, reverse=False)
            ndim_copy, b.mean_r2_list = sort_pair(b.ndim_list, b.mean_r2_list, reverse=False)
            ndim_copy, b.std_err_r2_list = sort_pair(b.ndim_list, b.std_err_r2_list, reverse=False)
            ndim_copy, b.var_r2_list = sort_pair(b.ndim_list, b.var_r2_list, reverse=False)
            b.ndim_list = np.array(ndim_copy)
            p.ax.errorbar(b.ndim_list, b.f_list, yerr=2 * np.array(b.ferr_list), label=dir)
            p2.ax.errorbar(b.ndim_list, b.mean_r2_list, yerr=2 * np.array(b.std_err_r2_list), label=dir)
            ndim_max = np.amax(b.ndim_list) if np.amax(b.ndim_list) > ndim_max else ndim_max
            ndim_min = np.amin(b.ndim_list) if np.amin(b.ndim_list) < ndim_min else ndim_min
    f_exact = []
    mean_r2_exact = []
    ndim_exact = np.arange(ndim_min, ndim_max+1)
    u2 = None
    geom_params = b.geom_params_list[0]
    for n in ndim_exact:
        if "sphere_exp" in p.out_name:
            ts = CheckExponentiallyDecayingProfile(np.zeros(n), geom_params[0], geom_params[1], cubic=False)
            f = -np.log(ts.get_exact_volume())
        elif "cube_exp" in p.out_name:
            tc = CheckExponentiallyDecayingProfile(np.zeros(n), geom_params[0], geom_params[1], cubic=True)
            f = -np.log(tc.get_exact_volume())
        elif "sphere_pow" in p.out_name:
            tp = CheckPowerDecayingProfile(np.zeros(n), geom_params[0], n + 4)
            f = -np.log(tp.get_exact_volume())
        elif "sphere_" in p.out_name:
            f = -log_volume_nball(geom_params[0], n)
            u2 = n * geom_params[0] ** 2 / (n + 2)
        elif "cube_" in p.out_name:
            f = - n * np.log(geom_params[0])
        else:
            raise NotImplementedError
        f_exact.append(f)
        if u2 is not None:
            mean_r2_exact.append(u2)
    p.ax.plot(ndim_exact, f_exact, label='exact')
    if len(mean_r2_exact) > 0:
        p2.ax.plot(ndim_exact, mean_r2_exact, label='exact')
    p.save_and_close('best')
    p2.save_and_close('best')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="perform parallel tempering for basin volume method")
    parser.add_argument("system_name", type=str, help="system name: sphere, sphere_exp_decay, \
                                                        sphere_pow_decay, cube, cube_exp_decay")
    parser.add_argument("--listdir", type=str, help="directory with explore_bv...",
                        default="cloud_radius*_points*")

    args = parser.parse_args()

    plot_numerical(args.system_name, listdir=args.listdir)




