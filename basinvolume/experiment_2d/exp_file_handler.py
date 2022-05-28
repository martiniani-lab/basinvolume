from __future__ import division
import numpy as np

class ExpFileHandler(object):
    """
    Reads in experimental data file of positions, radii.
    """
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        #
        input_file = open(self.data_file_path, "r")
        self.x = []
        self.y = []
        self.radii = []
        self.large = []
        while True:
            this_line = input_file.readline()
            if not this_line:
                break
            line_contents = this_line.split(",")
            self.x.append(line_contents[0])
            self.y.append(line_contents[1])
            self.radii.append(line_contents[2])
            self.large.append(line_contents[3].strip())
        input_file.close()
        self.x = map(float, self.x)
        self.y = map(float, self.y)
        self.radii = map(float, self.radii)
        self.large = map(bool, self.large)
        self.nr_particles = len(self.x)
        if len(self.y) != self.nr_particles or len(self.radii) != self.nr_particles or len(self.large) != self.nr_particles:
            raise Exception("ExpFileHandler: data read in error: extracted arrays with mismatching lengths")
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)
        self.radii = np.asarray(self.radii)
        self.large = np.asarray(self.large)
