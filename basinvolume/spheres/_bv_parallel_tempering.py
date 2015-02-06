from __future__ import division
import numpy as np
import sys
from mcpele.parallel_tempering import MPI_PT_RLhandshake, trymakedir
from basinvolume.utils import get_dist_com, integratedAutocorrelationTime_fft
from basinvolume.post_processing import spring_constants_variable_transform
from basinvolume.spheres import BV_MCrunner
from pymbar.timeseries import detectEquilibration_binary_search
import copy, warnings, time

class MPI_BV_PT_RLhandshake(MPI_PT_RLhandshake):
    """
    u2meank0 is mean of histogram from simulation done at k=0
    Tmax and Tmin here correspond to kmin and kmax, they should be computed by bv_find_params
    eq_min_ptiter: determines the minimum number of pt steps to perform before checking convergence, 95% of initial assigned time 
    eq_max_ptiter: determines the maximum number of pt steps to perform if convergence is not reached before, 20 times initial assigned time
    fast_ct, if false perform full convergence test computing the segment of the recorded time series that maximises the number of uncorrelated samples
    else return the maximum equilibration time
    """
    def __init__(self, mcrunner, Tmax, Tmin, u2meank0, max_ptiter=10, pfreq=1, skip=0, test_convergence=True, fast_ct=False, 
                 rel_std_err=0.03, min_window=2.5e5, max_eq_time=2.5e5, print_status=False, base_directory=None, verbose=False):
        super(MPI_BV_PT_RLhandshake,self).__init__(mcrunner, Tmax, Tmin, max_ptiter=max_ptiter, pfreq=pfreq, skip=skip, 
                                                   print_status=print_status, base_directory=base_directory, verbose=verbose)
        self.u2meank0 = u2meank0
        self.mcrunner_eqsteps = mcrunner.equilibration_steps
        self.test_convergence = test_convergence
        self.autocorr = []
        self.timeseries2 = np.array([])
        self.eq_time = 0 #time at which equilibration was reached
        self.fast_ct = fast_ct
        self.rel_std_err = rel_std_err #relative standard error
        self.rel_std_err_arr = [] #array of measured relative standard errors
        self.eq_min_ptiter = int(self.max_ptiter*0.95) #initial maxptiter is passed from command line #int(1e5/self.mcrunner.niter)#
        self.eq_max_ptiter = int(2e6/self.mcrunner.niter)
        self.min_window = min_window
        self.max_eq_time = max_eq_time
        assert(self.eq_min_ptiter > self.skip)
        assert(self.max_ptiter > self.eq_min_ptiter)
        assert(self.eq_max_ptiter > self.eq_min_ptiter)
        assert((self.eq_max_ptiter-self.eq_min_ptiter)*self.mcrunner.niter > self.min_window) #condition on the minimal window size
        assert(self.min_window > self.mcrunner_eqsteps)
        assert(self.max_eq_time > self.mcrunner_eqsteps)
        
    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        directory = "{0}/{1}".format(base_directory,self.rank)
        trymakedir(directory)
        self._master_print_temperatures()
        self._all_print_parameters()
        self.status_stream = open('{0}/{1}'.format(directory,'status'),'w')
        self.histogram_mean_stream = open('{0}/{1}'.format(directory,'hist_mean'),'w')
        self.histogram_mean_stream.write('{:<15}\t{:<15}\t{:<15}\t{:<15}\n'.format('iteration','<(x-x0)**2>','variance','std_err'))
        if self.rank == 0:
            self.permutations_stream = open(r'{0}/rem_permutations'.format(base_directory),'w')
    
    def _print_data(self):
        self._all_dump_timeseries() #convergence is tested in this function
        #the histogram depends on self.timeseries that is not empty only once the ts test is passed
        if self.ptiter >= self.eq_min_ptiter and self.timeseries2.size > self.mcrunner_eqsteps:
            self._all_dump_histogram()
    
    def _test_convergence(self):
        tail_timeseries = self.mcrunner.get_timeseries()
        tail_timeseries2 = np.power(tail_timeseries,2)
        self.timeseries2 = np.append(self.timeseries2, tail_timeseries2)
        if self.test_convergence and self.ptiter > self.eq_min_ptiter:
            if self.timeseries2.size < self.mcrunner_eqsteps:
                print "core {} attempted to test convergence before the mcrunner equilibration steps had terminated".format(self.rank)
                return self.max_ptiter
            else:
                return self._test_ts_convergence()
        else:
            return self.max_ptiter
        
    def _test_ts_convergence(self):
        """
        in_timeseries is the last segment of the time series
        self.timeseries is the whole recorded timeseries
        """
        if self.eq_time == 0:
            start=time.time()
            if self.fast_ct:
                self.eq_time = np.amin([self.max_eq_time, self.timeseries2.size])
            else:
                print "detecting equilibration point"
                print "timeseries size", self.timeseries2.size
                eq_time = detectEquilibration_binary_search(self.timeseries2, bs_nodes=100)[0]
                eq_time = np.amin([self.max_eq_time, eq_time]) #this should avoid detecting artifacts near the end of the series
                new_eq_time = np.amax([eq_time, self.mcrunner_eqsteps]) #guarantees that eq_time is larger than the mcrunner adapted number of steps
                #gather values, find largest, then broadcast it
                new_eq_time_array = self._gather_data([new_eq_time])
                if self.rank == 0:
                    new_eq_time = np.amax(new_eq_time_array)
                else:
                    new_eq_time = None
                self.eq_time = self._broadcast_data([new_eq_time], 1)[0]
                self.eq_time = int(self.eq_time)
            end=time.time()
            print "core {} set_eq_time: {} comp_eq_time: {} \
            mcrunner_eqsteps: {} len(timeseseries2): {} \
            time detect equilibration: {}".format(self.rank, self.eq_time, eq_time, 
                                                  self.mcrunner_eqsteps, self.timeseries2.size, end-start)
        #only keep time series from after the equilibration point, this references original data
        timeseries2 = self.timeseries2[self.eq_time:]
        new_max_ptiter = self._find_new_max_ptiter(timeseries2)
        #if self.verbose:
        print "new max_ptiter {}, current ptiter {}".format(new_max_ptiter, self.ptiter)
        print "core {} autocorrelation time {}".format(self.rank, self.autocorr)
        return new_max_ptiter
    
    def _find_new_max_ptiter(self, timeseries2):
        """
        resets max_ptiter based on desired relative standard error that one wants to achieve. The longest estimate 
        is chosen for the full pt. In order to estimate the number of extra steps to perform uses the correlated
        estimate for the standard error (see Troyer Am. J. Phys. 78 (2)) from which one can easily find that
        M = sig^2*(1+2t)/(mu rel_std_err)^2
        it returns an estimate of the new maxptiter only once the timeseries is longer than min_window
        """
        #to reduce nskip (use more points) make the factor by which timeseries.size is divided by larger
        nskip = max(int(np.round(timeseries2.size/1e6)),1) 
        tau = integratedAutocorrelationTime_fft(timeseries2[::nskip]) * nskip
        self.autocorr.extend([tau])
        var = np.var(timeseries2)
        mean = np.mean(timeseries2)
        sample_size = timeseries2.size
        rel_err = np.sqrt(var*(1+2*tau)/sample_size) / mean
        self.rel_std_err_arr.extend([rel_err])
        print "core {} relative standard error {}".format(self.rank, rel_err)
        
        #compute by how much to extend the time series, if has at least 1e5
        if sample_size < self.min_window: #self.autocorr[-1]*100
            new_max_ptiter = self.eq_max_ptiter
        elif rel_err < self.rel_std_err:
            m = 0
            new_max_ptiter = self.ptiter
        else:
            m = var * (1+2*tau) / np.power(mean * self.rel_std_err, 2)
            new_max_ptiter = self.ptiter + int((m-sample_size)/self.mcrunner.niter)
            
        new_max_ptiter_array = self._gather_data([new_max_ptiter])
        if self.rank == 0:
            max_ptiter = np.amax(new_max_ptiter_array)
        else:
            max_ptiter = None
        
        max_ptiter = self._broadcast_data([max_ptiter], 1)[0]
        return min(int(max_ptiter),self.eq_max_ptiter)
    
    def _all_dump_timeseries(self):
        """for this to work the directory must have been initialised in _print_initialise"""
        base_directory = self.base_directory
        directory = "{0}/{1}".format(base_directory,self.rank)
        iteration = self.mcrunner.get_iterations_count()
        fname = "{0}/TimeSeries.{1}".format(directory,int(iteration))
        self.mcrunner.dump_timeseries(fname, clear=True)
    
    def _all_dump_histogram(self):
        """for this to work the directory must have been initialised in _print_initialise"""
        base_directory = self.base_directory
        directory = "{0}/{1}".format(base_directory,self.rank)
        iteration = self.mcrunner.get_iterations_count()
        fname = "{0}/Visits.his.{1}".format(directory,float(iteration))
        if not self.suppress_histogram:
            mean, variance = self.mcrunner.dump_histogram(fname)
            self.histogram_mean_stream.write('{:<15}\t{:>15.15e}\t{:>15.15e}\n'.format(iteration,mean,variance))
        else:
            mean = np.mean(self.timeseries2[self.eq_time:])
            variance = np.var(self.timeseries2[self.eq_time:])
            std_err = self.rel_std_err_arr[-1]*mean
            self.histogram_mean_stream.write('{:<15}\t{:>15.15e}\t{:>15.15e}\t{:>15.15e}\n'.format(iteration,mean,variance,std_err))
        self.histogram_mean_stream.flush() #print every time not to lose data
    
    def _get_temps(self):
        """
        NOTE: BECAUSE K0 IS INCLUDED IN THE CALCULATION TARRAY CANNOT BE REVERSED AS [::-1]
        set up the spring constant. We give root the lowest temperature.
        This should increase performance when pair lists are used (they are updated less often at low temperature
        or when steps involve minimisation, as the low temperatures are closer to the minimum)
        """
        if (self.rank == 0):
            Tarray = spring_constants_variable_transform(self.nproc+1, self.Tmax, self.u2meank0, 
                                                         self.mcrunner.nparticles, self.mcrunner.bdim, self.Tmin)
            Tarray = Tarray[::-1]
            Tarray = np.array(Tarray[1:],dtype='d') #exclude kmax entry, no need to be simulated, mean is already available
            self.Tarray = Tarray
        else:
            self.Tarray = None
    
    def _attempt_exchange(self):
        """
        this function brings together all the functions necessary to attempt a configuration swap, it is structures as
        following:
        *root gathers the energies from the slaves
        *red_origin is just the origin for systems with pbc and is the reduced set of coordinates for systems with frozen coordinates 
        """
        #compute dx with com correction for each replica
        assert isinstance(self.mcrunner,BV_MCrunner)
        dx = get_dist_com(np.array(self.config,dtype='d'),np.array(self.mcrunner.red_origin,dtype='d'),self.mcrunner.bdim)
        
        #gather dx, only root will do so
        dx_array = self._gather_energies(dx)
        if self.verbose:
            if dx_array is not None:
                print "dx_array", dx_array
        #find exchange pattern (list of exchange buddies)
        exchange_pattern = self._find_exchange_buddy(dx_array)
        #now scatter the exchange pattern so that everybody knows who their buddy is
        exchange_buddy = self._scatter_single_value(np.array(exchange_pattern,dtype='d'))
        exchange_buddy = int(exchange_buddy)
        #attempt configurations swap
        assert(self.mcrunner.potential.get_k() == self.T) #debug
        self.config = self._exchange_pairs(exchange_buddy, np.array(self.config,dtype='d'))
        if (exchange_buddy != self.no_exchange_int):
            #recompute energy (this assumes that mcrunner has member origin)
            self.energy = self.mcrunner.potential.getEnergy(np.array(self.config,dtype='d'))
        
    def _find_exchange_buddy(self, dx_array):
        """
        This function determines the exchange pattern alternating swaps with right and left neighbours.
        An exchange pattern array is constructed, filled with self.no_exchange_int which
        signifies that no exchange should be attempted. This value is replaced with the
        rank of the processor with which to perform the swap if the swap attempt is successful.
        The exchange partner is then scattered to the other processors.
        """        
        if (self.rank == 0):
            assert(len(dx_array)==len(self.Tarray))
            exchange_pattern = np.empty(len(dx_array),dtype='int32')
            exchange_pattern.fill(self.no_exchange_int) #reset exchange pattern to no exchange
            self.anyswap = False
            
            for i in self.nodelist[1::2]:
                if self.verbose:
                    print 'exchange choice: ',self.exchange_dic[self.exchange_choice] #this is a print statement that has to be removed after initial implementation
                
                dx1 = dx_array[i]
                T1 = self.Tarray[i]
                dx2 = dx_array[i+self.exchange_choice]
                T2 = self.Tarray[i+self.exchange_choice]

                #Hamiltonia replica exchange
                deltaE = 0.5*dx2*dx2 - 0.5*dx1*dx1
                deltabeta = T2 - T1
                w = np.exp(deltaE * deltabeta)
                rand = np.random.rand()
                
                #print 'w {} rand {}'.format(w,rand)
                #print 'deltaE {} deltaT {}'.format(deltaE, deltabeta)
                #print "E1 {0} T1 {1} E2 {2} T2 {3} w {4}".format(E1,T1,E2,T2,w) 
                if w > rand:
                    #accept exchange
                    if self.verbose:
                        self.ex_outstream.write("accepting exchange %d %d %g %g %g %g %d\n" % (self.nodelist[i], self.nodelist[i+self.exchange_choice], dx1, dx2, T1, T2, self.ptiter))
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
                self._master_print_permutations()
        else:
            exchange_pattern = None
        
        self.exchange_choice *= -1 #swap direction of exchange choice
        #print "exchange_pattern",exchange_pattern
        return exchange_pattern
    