from __future__ import division
import numpy as np
import argparse
from basinvolume.spheres import configure_bv_mcrunner
from mcpele.parallel_tempering import MPI_PT_RLhandshake
from basinvolume.post_processing import spring_constants_variable_transform

class MPI_BV_PT_RLhandshake(MPI_PT_RLhandshake):
    """
    u2meank0 is mean of histogram from simulation done at k=0
    Tmax and Tmin here correspond to kmin and kmax, they should be computed by bv_find_params 
    """
    def __init__(self, mcrunner, Tmax, Tmin, u2meank0, max_ptiter=10, pfreq=1, base_directory=None, verbose=False):
        super(MPI_BV_PT_RLhandshake,self).__init__(mcrunner, Tmax, Tmin, max_ptiter= max_ptiter, pfreq=pfreq, 
                                                   base_directory=base_directory, verbose=verbose)
        self.u2meank0 = u2meank0 

    def _get_temps(self):
        """
        set up the spring constant. We give root the lowest temperature.
        This should increase performance when pair lists are used (they are updated less often at low temperature
        or when steps involve minimisation, as the low temperatures are closer to the minimum)
        """
        if (self.rank == 0):
            Tarray = spring_constants_variable_transform(self.nproc, self.Tmax, self.u2meank0, 
                                                         self.mcrunner.nparticles, self.mcrunner.bdim, self.Tmin)
            self.Tarray = np.array(Tarray,dtype='d')
        else:
            self.Tarray = None
    
    def _find_exchange_buddy(self, Earray):
        """
        This function determines the exchange pattern alternating swaps with right and left neighbours.
        An exchange pattern array is constructed, filled with self.no_exchange_int which
        signifies that no exchange should be attempted. This value is replaced with the
        rank of the processor with which to perform the swap if the swap attempt is successful.
        The exchange partner is then scattered to the other processors.
        """        
        if (self.rank == 0):
            assert(len(Earray)==len(self.Tarray))
            exchange_pattern = np.empty(len(Earray),dtype='int32')
            exchange_pattern.fill(self.no_exchange_int)
            self.anyswap = False
            for i in xrange(0,self.nproc,2):
                if self.verbose:
                    print 'exchange choice: ',self.exchange_dic[self.exchange_choice] #this is a print statement that has to be removed after initial implementation
                E1 = Earray[i]
                T1 = self.Tarray[i]
                E2 = Earray[i+self.exchange_choice]
                T2 = self.Tarray[i+self.exchange_choice]
                deltaE = E1 - E2
                deltabeta = 1
                w = min( 1. , np.exp( deltaE * deltabeta ) )
                rand = np.random.rand()
                #print "E1 {0} T1 {1} E2 {2} T2 {3} w {4}".format(E1,T1,E2,T2,w) 
                if w > rand:
                    #accept exchange
                    if self.verbose:
                        self.ex_outstream.write("accepting exchange %d %d %g %g %g %g %d\n" % (self.nodelist[i], self.nodelist[i+self.exchange_choice], E1, E2, T1, T2, self.ptiter))
                    assert(exchange_pattern[i] == self.no_exchange_int)                      #verify that is not using the same processor twice for swaps
                    assert(exchange_pattern[i+self.exchange_choice] == self.no_exchange_int) #verify that is not using the same processor twice for swaps
                    exchange_pattern[i] = self.nodelist[i+self.exchange_choice]
                    exchange_pattern[i+self.exchange_choice] = self.nodelist[i]
                    self.anyswap = True
            ############end of for loop###############
            #record self.permutation_pattern to print permutations in print function
            if self.anyswap:
                for i,buddy in enumerate(exchange_pattern):
                    if (buddy != self.no_exchange_int):
                        self.permutation_pattern[i] = buddy+1 #to conform to fortran notation
                    else:
                        self.permutation_pattern[i] = i+1 #to conform to fortran notation
        else:
            exchange_pattern = None
        
        self.exchange_choice *= -1 #swap direction of exchange choice
        #print "exchange_pattern",exchange_pattern
        return exchange_pattern
            
if __name__ == "__main__":
    """
    set Tmax to k_max
    set Tmin to k_min = 0
    set <u2>_min = mean of histogram from simulation done at k=0
    """
    parser = argparse.ArgumentParser(description="perform parallel tempering for basin volume method")
    parser.add_argument("jammed_packing_fname", type=str, help="name of xy[z]dr file")
    parser.add_argument("base_directory", type=str, help="directory in which to save results")
    args = parser.parse_args()
    
    path = args.base_directory
    fname = args.jammed_packing_fname
    
    #Parallel Tempering
    sim = configure_bv_mcrunner()
    mcrunner = sim(fname, niter=1e4, stepsize=1e-1, dtol=1e-4, hmin=0, 
                 hmax=100, hbinsize=1e-1, acceptance=0.2, adjustf=0.9, adjustf_niter = 5000, adjustf_navg = 100)
    kmin = 0
    displ_k_min = sim.displ_k_min
    var_displ_k_min = sim.displ_k_min
    kmax = sim.kmax
    #Tarray = spring_constants_variable_transform(4, kmax, displ_k_min, mcrunner.nparticles, 2, 0)
    ptrunner = MPI_BV_PT_RLhandshake(mcrunner, kmax, kmin, displ_k_min, max_ptiter=10, pfreq=1, base_directory=path, verbose=True)
    ptrunner.run()
    