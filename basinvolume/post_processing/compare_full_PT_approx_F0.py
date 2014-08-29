from __future__ import division
try:
    import numpy as np
    import argparse
    import ConfigParser
    import os
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.optimize import curve_fit
except ImportError as err:
    print err

def save_pdf(plt, file_name):
    pdf = PdfPages(file_name)
    plt.savefig(pdf, format="pdf")
    pdf.close()
    plt.close()
    
class GenerateComparisonPlotPTApprox(object):
    def __init__(self, packings_dir, plot_ts_integrand_data = False):
        self.packings_dir = packings_dir
        self.plot_ts_integrand_data = plot_ts_integrand_data
        self._compute_F0()
        self._gather_data()
        self._generate_plots()
    def _compute_F0(self):
        self.explore_dirs = [self.packings_dir + "/" + f for f in os.listdir(self.packings_dir) if f.startswith("explore_bv_jammed_packing")]
        self.packing_strings = ["jammed_" + (s.split("/")[-1]).split("_")[3] for s in self.explore_dirs]
        from basinvolume.spheres import _collect_u2_vs_k
        sim = _collect_u2_vs_k()
        for (path, fname) in zip(self.explore_dirs, self.packing_strings):
            try:
                sim(fname = fname, explore_dir = path, packings_dir = os.path.abspath(self.packings_dir + "/jammed_packings"), plot_ts_integrand_data = self.plot_ts_integrand_data)
            except:
                print "failed packing!"
                print "name: ", fname
                print "path:", path
    def _gather_data(self):
        self.volume_files = [f + "/analysis/volume_data" for f in self.explore_dirs]
        print self.volume_files[:2]
        self.F0 = []
        self.unit_box_F0 = []
        self.sigF0 = []
        self.F0_approx = []
        self.unit_box_F0_approx = []
        for vf in self.volume_files:
            self._read_from_volume_file(vf)
    def _read_from_volume_file(self, vf):
        volf = ConfigParser.ConfigParser()
        volf.read(str(vf))
        try:
            self.F0.append(volf.getfloat('VOLUME_FULL_PT', 'F0'))
            self.unit_box_F0.append(volf.getfloat('VOLUME_FULL_PT', 'unit_box_F0'))
            self.sigF0.append(volf.getfloat('VOLUME_FULL_PT', 'sigF0'))
        except:
            print "no PT data availible"
            print "location:", vf
        try:
            self.F0_approx.append(volf.getfloat('VOLUME_APPROXIMATED', 'F0_approx'))
            self.unit_box_F0_approx.append(volf.getfloat('VOLUME_APPROXIMATED', 'unit_box_F0_approx'))
        except:
            print "no approx integral data available"
            print "location:", vf
    def _generate_plots(self):
        plt.hist(self.F0)
        save_pdf(plt, self.packings_dir + "/volume_histogram_F0.pdf")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare F0 form full PT data with F0 from integral approximation")
    parser.add_argument("-d", "--packings_dir", help = "top-level dir containing the packings, e.g. n32_phi88_2D")
    args = parser.parse_args()
    print args
    packings_dir = os.path.abspath(args.packings_dir)
    GenerateComparisonPlotPTApprox(packings_dir, plot_ts_integrand_data = False)