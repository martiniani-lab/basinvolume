from __future__ import division
import numpy as np
from mcpele.parallel_tempering import MPI_PT_RLhandshake
from basinvolume.post_processing import spring_constants_variable_transform
from basinvolume.spheres import BV_MCrunner

class MPI_BV_PT_RLhandshake(MPI_PT_RLhandshake):
    """
    u2meank0 is mean of histogram from simulation done at k=0
    Tmax and Tmin here correspond to kmin and kmax, they should be computed by bv_find_params 
    """
    def __init__(self, mcrunner, Tmax, Tmin, u2meank0, max_ptiter=10, pfreq=1, skip=0, base_directory=None, verbose=False):
        super(MPI_BV_PT_RLhandshake,self).__init__(mcrunner, Tmax, Tmin, max_ptiter= max_ptiter, pfreq=pfreq, skip=skip, 
                                                   base_directory=base_directory, verbose=verbose)
        self.u2meank0 = u2meank0

    def _get_temps(self):
        """
        set up the spring constant. We give root the lowest temperature.
        This should increase performance when pair lists are used (they are updated less often at low temperature
        or when steps involve minimisation, as the low temperatures are closer to the minimum)
        """
        if (self.rank == 0):
            Tarray = spring_constants_variable_transform(self.nproc+2, self.Tmax, self.u2meank0, 
                                                         self.mcrunner.nparticles, self.mcrunner.bdim, self.Tmin)
            Tarray = np.array(Tarray[1:-1],dtype='d') #exclude k=0 and kmax entry, no need to be simulated, means already available
            self.Tarray = Tarray[::-1]
        else:
            self.Tarray = None
    
    def _attempt_exchange(self):
        """
        this function brings together all the functions necessary to attempt a configuration swap, it is structures as
        following:
        *root gathers the energies from the slaves
        
        """
        #gather energies, only root will do so
        Earray = self._gather_energies(self.energy)
        if self.verbose:
            if Earray is not None:
                print "Earray", Earray
        #find exchange pattern (list of exchange buddies)
        exchange_pattern = self._find_exchange_buddy(Earray)
        #now scatter the exchange pattern so that everybody knows who their buddy is
        exchange_buddy = self._scatter_single_value(np.array(exchange_pattern,dtype='d'))
        exchange_buddy = int(exchange_buddy)
        #attempt configurations swap
        self.config = self._exchange_pairs(exchange_buddy, self.config)
        #recompute energy (this assumes that mcrunner has member origin)
        assert isinstance(self.mcrunner,BV_MCrunner)
        dx = np.array(self.config-self.mcrunner.origin,dtype='d')
        self.E = 0.5*self.T*np.dot(dx,dx)
    
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
            for i in self.nodelist[0::2]:
                if self.verbose:
                    print 'exchange choice: ',self.exchange_dic[self.exchange_choice] #this is a print statement that has to be removed after initial implementation
                E1 = Earray[i]
                T1 = self.Tarray[i]
                E2 = Earray[i+self.exchange_choice]
                T2 = self.Tarray[i+self.exchange_choice]
                deltaE = E2/T2 - E1/T1
                deltabeta = T2 - T1
                w = min( 1. , np.exp( deltaE * deltabeta ) )
                rand = np.random.rand()
                #print 'w {} rand {}'.format(w,rand)
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
    