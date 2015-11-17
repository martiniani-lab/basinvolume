from __future__ import division

class ComputerCommon(object):
    """
    Common functionality of volume computers for benchmark purposes.
    
    Basic structure for computing basin voulme as function of number of
    function calls and writing it to disk.
    """
    def __init__(self, results_path):
        self.results_path = results_path
