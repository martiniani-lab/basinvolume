#ifndef _BV_FIND_NR_DECORRELATION_STEPS_H
#define _BV_FIND_NR_DECORRELATION_STEPS_H

#include "mcpele/mc.h"
#include "mcpele/histogram.h"
#include "mcpele/actions.h"
#include "mcpele/rsm_displacement.h"

namespace bv{

class RSMDTracker{
private:
    mcpele::GetMeanRMSDisplacement* rsmd;
    const size_t boxdim;
    size_t nr_decorrelation_steps;
    bool sufficient_diffusion;
    const double desired_mean_rsm_displ;
public:
    RSMDTracker(pele::Array<double> initial_, const size_t boxdim_,
            const double desired_mean_rsm_displ_)
        : rsmd(new mcpele::GetMeanRMSDisplacement(initial_, boxdim_)),
          boxdim(boxdim_),
          nr_decorrelation_steps(0),
          sufficient_diffusion(false),
          desired_mean_rsm_displ(desired_mean_rsm_displ_)
    {}
    virtual ~RSMDTracker()
    {
        delete rsmd;
    }
    void reset(pele::Array<double> initial)
    {
        delete rsmd;
        rsmd = new mcpele::GetMeanRMSDisplacement(initial, boxdim);
        nr_decorrelation_steps = 0;
        sufficient_diffusion = false;
    }
    void check_next(pele::Array<double> coords)
    {
        const double this_rsmd = rsmd->compute_mean_rsm_displacement(coords);
        if (this_rsmd > desired_mean_rsm_displ) {
            sufficient_diffusion = true;
        }
        else {
            ++nr_decorrelation_steps;
        }
    }
    bool sufficient_diffusion_happened() const
    {
        return sufficient_diffusion;
    }
    double get_nr_decorrelation_steps() const
    {
        return nr_decorrelation_steps;
    }
};

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
