#ifndef _BV_FIND_NR_DECORRELATION_STEPS_H
#define _BV_FIND_NR_DECORRELATION_STEPS_H

#include "mcpele/mc.h"
#include "mcpele/histogram.h"
#include "mcpele/actions.h"

namespace bv{


class FindNrDecorrelationSteps: public mcpele::Action{
private:
    const double desired_mean_rsm_displ;
    const size_t nr_iterations_start;
    const size_t nr_samples_avergage;
    mcpele::Moments nr_decorrelation_steps;
public:
    FindNrDecorrelationSteps(const double desired_mean_rsm_displ_,
            const size_t nr_iterations_start_, const size_t nr_samples_avergage_ = 1);
    virtual ~FindNrDecorrelationSteps(){}
    virtual void action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc);
    size_t get_nr_decorrelation_steps() const;
};


} //namespace bv

#endif //#ifndef _BV_FIND_NR_DECORRELATION_STEPS_H
