#ifndef _BV_FIND_NR_DECORRELATION_STEPS_H
#define _BV_FIND_NR_DECORRELATION_STEPS_H

#include "mcpele/mc.h"
#include "mcpele/histogram.h"

#include "rsmd_tracker.h"

namespace bv{


class FindNrDecorrelationSteps: public mcpele::Action{
private:
    const size_t nr_iterations_start;
    const size_t nr_samples_avergage;
    mcpele::Moments nr_decorrelation_steps;
    RSMDTracker rsmd_tracker;
    void compute_and_update_rmsd(pele::Array<double>);
public:
    FindNrDecorrelationSteps(const double desired_mean_rsm_displ_,
            const size_t nr_iterations_start_, const size_t nr_samples_avergage_,
            pele::Array<double> initial_coords_, const size_t boxdim_);
    virtual ~FindNrDecorrelationSteps(){}
    virtual void action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc);
    size_t get_nr_decorrelation_steps() const;
    bool done() const;
};


} //namespace bv

#endif //#ifndef _BV_FIND_NR_DECORRELATION_STEPS_H
