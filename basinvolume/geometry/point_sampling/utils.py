from __future__ import print_function
from builtins import str
from builtins import map
from builtins import object
import numpy as np

class LineStitcher(object):
    def __init__(self, raw_lines):
        self.raw_lines = raw_lines
        #
        self.lines = []
        self.stitch_lines()
        print(("self.lines", self.lines))
        assert(False)
    def stitch_lines(self):
        status = []
        in_line = False
        tmp = []
        for l in self.raw_lines:
            if in_line:
                tmp.append(l)
                if l.endswith("]\n"):
                    in_line = False
                    tmp.append(l)
                    tmp = [s.replace("\n", "") for s in tmp]
                    tmp = [s.replace("\t", "") for s in tmp]
                    tmp = [s.replace("]", "],") for s in tmp]
                    tmp = np.asarray(tmp)
                    tmp = tmp.flatten()
                    
                    self.lines.append(tmp.squeeze())
                    tmp = []
            else:
                if l.startswith("["):
                    in_line = True
                    tmp.append(l)

def get_means_cov(gauss_path):
    print("reading means, cov from the following gauss path")
    print(("gauss_path", gauss_path))
    mean = []
    cov = []
    f = open(gauss_path, "r")
    #stitched_lines = LineStitcher(f.readlines())
    #for line in stitched_lines.lines:
    for line in f.readlines():
        if line.startswith("["):
            m = None
            c = None
            print(("line", line))
            if "," in line:
                m = list(map(float, (line.split(",")[0].replace("[", "")).replace("]", "").split()))
                c = list(map(float, (line.split(",")[1].replace("[", "")).replace("]", "").split()))
            else:
                m = list(map(float, (line.split("\t")[0].replace("[", "")).replace("]", "").split()))
                c = list(map(float, (line.split("\t")[1].replace("[", "")).replace("]", "").split()))
            mean.append(m)
            cov.append(c)
    f.close()
    return np.asarray(mean), np.asarray(cov)

def _append_geom_params(dname, geometry, geom_params):
    if geometry == "cube":
        sidelength = geom_params[0]
        dname2 = '_l'+str(sidelength)
    elif geometry == "sphere":
        radius = geom_params[0]
        dname2 = '_r' + str(radius)
    elif geometry == "cube_exp_decay":
        sidelength = geom_params[0]
        decay_length = geom_params[1]
        dname2 = '_l' + str(sidelength) + '_edl' + str(decay_length)
    elif geometry == "sphere_exp_decay":
        radius = geom_params[0]
        decay_length = geom_params[1]
        dname2 = '_r' + str(radius) + '_edl' + str(decay_length)
    elif geometry == "sphere_pow_decay":
        radius = geom_params[0]
        exponent = geom_params[1]
        dname2 = '_r' + str(radius) + '_p' + str(exponent)
    else:
        raise NotImplementedError
    return dname+dname2
