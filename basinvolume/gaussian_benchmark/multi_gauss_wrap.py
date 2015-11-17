from __future__ import division

class MultiGaussWrap(object):
    """
    Wrapper for Shang's multi-gaussian potential.
    
    Writes and reads multi-gaussian potentials (ensembles) as needed for the gaussian benchmark computaions.
    Allows to use same interface to potentials for each method in the benchmark.
    """
    def __init__(self, potential_path, small_basin_index=0):
        self.potential_path = potential_path
        self.small_basin_index = small_basin_index
        
    def this_path(g, d, i):
        return os.path.join(self.potential_path, g, d, i)
        
    def exists(self, g, d, i):
        return os.path.exists(self.this_path(g, d, i))
    
    def generate(self, g, d, i):
        
        
    def select_benchmark_basins(self, g, d, i):
        
