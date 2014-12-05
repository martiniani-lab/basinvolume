from __future__ import division
import numpy as np
import argparse
import os
import subprocess
from scipy.optimize import curve_fit
import basinvolume
from basinvolume.utils import MomentsAcc
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError as err:
    print err

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

def time_law_linear(x, b, c):
    """
    There is no reason for a "c" here, but it could correct for
    small-size effects.
    """
    return b * np.asarray(x) + c
    
class MomentsInTime(object):
    def __init__(self):
        self.mom = MomentsAcc()
    def add(self, time_string):
        self.mom.update(self.get_seconds(time_string))
    def get_mean(self):
        return self.mom.get_mean()
    def get_std(self):
        return self.mom.get_std()
    def get_seconds(self, time_str):
        hours = float(time_str.split(":")[0].strip())
        minutes = float(time_str.split(":")[1].strip())
        seconds = float(time_str.split(":")[2].strip())
        return seconds + 60 * minutes + 60 * 60 * hours

class TimeStatistics(object):
    def __init__(self, time_strings):
        tmp = MomentsInTime()
        for s in time_strings:
            tmp.add(s)
        self.mean = tmp.get_mean()
        self.std = tmp.get_std()

class RuntimeData(object):
    def __init__(self, N_new):
        if N_new is None:
            raise Exception("provide particle number with --N")
        self.N = []
        self.time = []
        self.time_std = []
        self.repo_path = os.path.dirname(basinvolume.__file__)
        self.data_script_path = os.path.join(os.path.split(self.repo_path)[0], "runtime/collect_runtime_data.sh")
        self.get_time_data()
        if len(self.N) != len(self.time):
            raise Exception("mismatch in time and particle number labels")
        plt.errorbar(self.N, self.time, fmt="o", yerr=self.time_std)
        plt.xlabel(r"Number of particles $N$")
        plt.ylabel(r"Runtime on dexter / seconds")
        popt, pcov = curve_fit(time_law, self.N, self.time, [10000, 1], sigma=self.time_std)
        popt_linear, pcov_linear = curve_fit(time_law_linear, self.N, self.time, [10000, 0], sigma=self.time_std)
        print "large-N exponent: N^", popt[1]
        print "slope linear law:", popt_linear[0]
        print "offset linear law:", popt_linear[1]
        xf = np.linspace(self.N[0], self.N[-1])
        plt.plot(xf, time_law(xf, popt[0], popt[1]), "k", label="Power law, no offset")
        plt.plot(xf, time_law_linear(xf, popt_linear[0], popt_linear[1]), label="Linear law, with offset")
        plt.legend(loc=2)
        save_pdf(plt, "runtime_scaling.pdf")
        print "prediced runtime for", N_new, "particles: "
        self.print_prediction("Power law", time_law(N_new, popt[0], popt[1]))
        self.print_prediction("Linear law", time_law_linear(N_new, popt_linear[0], popt_linear[1]))
    def print_prediction(self, name, new_seconds):
        print "---begin prediction", name, "---"
        print new_seconds, "seconds"
        print "which is", new_seconds/60/60, "hours"
        print "or", new_seconds/60/60/24, "days"
        print "---"
    def get_time_data(self):
        print ("collect time data")
        print self.data_script_path
        collect_input = 42 # This is not needed for now.
        subprocess.call([self.data_script_path, str(collect_input)])
    def get_time_data_manual(self):
        self.N.append(16)
        tmp = TimeStatistics(["08:20:09", "03:28:26", "02:01:36", "01:32:11", "01:16:41", "05:47:54", "03:56:56", "04:28:51", "01:20:07"])
        self.time.append(tmp.mean)
        self.time_std.append(tmp.std)
        self.N.append(20)
        tmp = TimeStatistics(["10:34:40", "01:42:41", "12:25:42", "02:42:58", "02:32:31", "03:07:56", "03:38:09", "08:58:46", "02:52:39"])
        self.time.append(tmp.mean)
        self.time_std.append(tmp.std)
        self.N.append(24)
        tmp = TimeStatistics(["08:11:51", "31:35:17", "05:01:53", "12:35:26", "04:02:25", "07:13:05", "04:59:09", "19:45:56", "21:49:32"])
        self.time.append(tmp.mean)
        self.time_std.append(tmp.std)
        self.N.append(32)
        tmp = TimeStatistics(["05:25:18", "28:07:44", "18:03:35", "13:58:00", "16:56:07", "43:00:49", "44:06:29", "19:08:42", "08:21:22", "07:17:09", "22:29:50", "06:54:43", "24:29:44", "40:43:31"])
        self.time.append(tmp.mean)
        self.time_std.append(tmp.std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='basinvolume: runtime analysis')
    parser.add_argument("--N", type=int, nargs="?")
    args = parser.parse_args()
    RuntimeData(args.N)
