import numpy as np

class LineStitcher(object):
    def __init__(self, raw_lines):
        self.raw_lines = raw_lines
        #
        self.lines = []
        self.stitch_lines()
        print("self.lines", self.lines)
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
    print("gauss_path", gauss_path)
    mean = []
    cov = []
    f = open(gauss_path, "r")
    stitched_lines = LineStitcher(f.readlines())
    f.close()
    for line in stitched_lines.lines:
        if line.startswith("["):
            m = map(float, (line.split("\t")[0].replace("[", "")).replace("]", "").split())
            c = map(float, (line.split("\t")[1].replace("[", "")).replace("]", "").split())
            mean.append(m)
            cov.append(c)
    return np.asarray(mean), np.asarray(cov)
