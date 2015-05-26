"""
First part of basin volume analysis.
Compute the volumes of all basins present in the folder.
Do not compute entropy or any further information; this is done in compute_entropy.py.
"""

from __future__ import division
import os
import numpy as np
import argparse
from itertools import cycle
from basinvolume.post_processing import PackingDataSet
try:
    import pylab as plt
except ImportError as err:
    print err
    
class _compute_volumes(object):
    """
    Compute basin volumes from PT data with thermodynamic integration.
    *ts_skip number of points skipped when printing time series (every ts_skip)
    """
    def __init__(self, set_path=None, frozen=False, method=None):
        #
        self.set_path = set_path
        self.frozen = frozen
        self.method = method
        #
        if self.set_path is None or self.method is None:
            raise Exception("_compute_volumes: illegal input")
        #
        self.set_path = os.path.abspath(self.set_path)
        print("self.set_path", self.set_path)
        self.input_data = 
        self.output_data = PackingDataSet(self.set_path)
    def run(self):
        print("running volume computation")
               
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute basin volumes from PT data, use either TINT or MBAR")
    parser.add_argument("-p","--set_path", type=str, help="packing data set path, e.g. n24_phi50_phi70_3D", default=None)
    parser.add_argument("-m", "--method", type=str, help="volume computation method", default="MBAR")
    parser.add_argument("--frozen", action='store_true', help="has frozen atoms, default: False", default=False)
    args = parser.parse_args()
    print("args", args)
    comp = _compute_volumes(set_path = args.set_path, frozen=args.frozen, method=args.method)
    comp.run()
