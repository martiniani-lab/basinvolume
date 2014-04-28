from __future__ import division

class Experimental_Packing(object):
    """
    This reads in and holds the experimental packing data as required for the
    cutting of (smaller) packings below.
    """
    def __init__(self, input_file_name):
        self.input_file_name = input_file_name
        self.input_file = open(self.input_file_name,"r")
        self.x = [] 
        self.y = [] 
        self.r = [] 
        self.large = [] 
        while True:
            this_line = self.input_file.readline()
            if not this_line:
                break
            #split this_line (which is not empty) at comma and store data for processing
            line_contents = this_line.split(',')
            self.x.append(line_contents[0])
            self.y.append(line_contents[1])
            self.r.append(line_contents[2])
            self.large.append(line_contents[3].strip())
        self.x = map(float, self.x)
        self.y = map(float, self.y)
        self.r = map(float, self.r)
        self.large = map(bool, self.large)
        #print(self.x[0]+self.x[1])
        #print(self.large[-1])

class Cut_Out_Packings(object):
    """
    Out of the data set with the given index, this cuts out a given number of
    packings with the specified number of (non-frozen) particles.
    """
    def __init__(self, data_set_index=0, nr_of_particles=8, nr_of_packings=10, path_to_data="data"):
        if nr_of_particles <= 0:
            raise Exception('expect finite number of particles')
        if nr_of_packings <= 0:
            raise Exception('expect finite number of packings')
        if data_set_index < 0 or data_set_index >= 4:
            raise Exception('expect data set index to be between 0 and 3')
        self.data_file_idx = [1,3,4,5]
        self.data_set_index = self.data_file_idx[data_set_index]
        self.nr_of_particles = nr_of_particles
        self.nr_of_packings = nr_of_packings
        self.data_file_name = "PackingsData_%d.dat" % self.data_set_index
        self.path_to_data = path_to_data
        self.path_to_file = "/".join([self.path_to_data,self.data_file_name])
        self.all_particles = Experimental_Packing(self.path_to_file);
    
    def run(self):
        print(self.path_to_file)
        
if __name__ == "__main__":
    sys = Cut_Out_Packings()
    sys.run()
