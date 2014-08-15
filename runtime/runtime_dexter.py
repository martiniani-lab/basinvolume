from __future__ import division
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

def save_pdf(plt, file_name):
    pdf = PdfPages(file_name)
    plt.savefig(pdf, format="pdf")
    pdf.close()
    plt.close()

def time_law(x, b, c):
    """
    Will presumably fit OK; could be made more complicated.
    """
    return b * x ** c

class RuntimeData(object):
    def __init__(self, N_new):
        if N_new is None:
            raise Exception("provide particle number with --N")
        """
        BEGIN: This data is an example only -- replace with true runtimes later!
        """
        self.N = [25, 50, 100, 200]
        self.time = ["00:05:56", "00:25:27", "01:43:43", "07:13:27"]
        """
        END: This data is an example only -- replace with true runtimes later!
        """
        self.time = [self.get_seconds(t) for t in self.time]
        plt.loglog(self.N, self.time, "o")
        plt.xlabel(r"Number of particles $N$")
        plt.ylabel(r"Runtime on dexter / seconds")
        popt, pcov = curve_fit(time_law, self.N, self.time)
        print "large-N exponent: N^", popt[1]
        xf = np.linspace(self.N[0], self.N[-1])
        plt.plot(xf, time_law(xf, popt[0], popt[1]), "k")
        save_pdf(plt, "runtime_scaling.pdf")
        print "prediced runtime for", N_new, "particles: "
        new_seconds = time_law(N_new, popt[0], popt[1])
        print new_seconds, "seconds"
        print "which is", new_seconds/60/60, "hours"
        print "and", new_seconds/60/60/24, "days"
    def get_seconds(self, time_str):
        hours = float(time_str.split(":")[0].strip())
        minutes = float(time_str.split(":")[1].strip())
        seconds = float(time_str.split(":")[2].strip())
        return seconds + 60 * minutes + 60 * 60 * hours

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='basinvolume: runtime analysis')
    parser.add_argument("--N", type=int, nargs="?")
    args = parser.parse_args()
    RuntimeData(args.N)