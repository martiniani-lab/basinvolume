#include "find_nr_decorrelation_steps.h"

namespace bv{

FindNrDecorrelationSteps::FindNrDecorrelationSteps(const double desired_mean_rsm_displ_,
        const size_t nr_iterations_start_, const size_t nr_samples_avergage_)
    : desired_mean_rsm_displ(desired_mean_rsm_displ_),
      nr_iterations_start(nr_iterations_start_),
      nr_samples_avergage(nr_samples_avergage_)
{}


size_t FindNrDecorrelationSteps::get_nr_decorrelation_steps() const
{
    // This part only works with histogram_extension part of mcpele.
    /*
    if (nr_decorrelation_steps.count() == 0) {
        throw std::runtime_error("FindNrDecorrelationSteps::get_nr_decorrelation_steps: illegal read attempt");
    }
    */
    return nr_decorrelation_steps.mean();
}

void FindNrDecorrelationSteps::action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc)
{

}


} //namespace bv
