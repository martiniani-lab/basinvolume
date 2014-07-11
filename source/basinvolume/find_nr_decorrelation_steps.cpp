#include "find_nr_decorrelation_steps.h"

namespace bv{

FindNrDecorrelationSteps::FindNrDecorrelationSteps(const double desired_mean_rsm_displ_,
        const size_t nr_iterations_start_, const size_t nr_samples_avergage_,
        pele::Array<double> initial_coords_, const size_t boxdim_)
    : nr_iterations_start(nr_iterations_start_),
      nr_samples_avergage(nr_samples_avergage_),
      rsmd_tracker(initial_coords_, boxdim_, desired_mean_rsm_displ_)
{}

size_t FindNrDecorrelationSteps::get_nr_decorrelation_steps() const
{
    if (nr_decorrelation_steps.count() == 0) {
        throw std::runtime_error("FindNrDecorrelationSteps::get_nr_decorrelation_steps: illegal read attempt");
    }
    return nr_decorrelation_steps.mean();
}

bool FindNrDecorrelationSteps::done() const
{
    return nr_decorrelation_steps.count() == nr_samples_avergage;
}

void FindNrDecorrelationSteps::action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc)
{
    if (mc->get_iterations_count() < nr_iterations_start) {
        return;
    }
    compute_and_update_rmsd(coords);
}

void FindNrDecorrelationSteps::compute_and_update_rmsd(pele::Array<double> coords)
{
    rsmd_tracker.check_next(coords);
    if (rsmd_tracker.sufficient_diffusion_happened()) {
        nr_decorrelation_steps.update(rsmd_tracker.get_nr_decorrelation_steps());
        rsmd_tracker.reset(coords);
    }
}


} //namespace bv
