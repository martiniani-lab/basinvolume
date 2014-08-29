from __future__ import division
try:
    import numpy as np
    import argparse
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
        #self._generate_plots()
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
        self.volume_paths = [f in self.explore_dirs]
        print self.volume_paths[:2]
    #def _generate_plots(self):
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare F0 form full PT data with F0 from integral approximation")
    parser.add_argument("-d", "--packings_dir", help = "top-level dir containing the packings, e.g. n32_phi88_2D")
    args = parser.parse_args()
    print args
    packings_dir = os.path.abspath(args.packings_dir)
    plot_ts_integrand_data = False
    GenerateComparisonPlotPTApprox(packings_dir, plot_ts_integrand_data = plot_ts_integrand_data)