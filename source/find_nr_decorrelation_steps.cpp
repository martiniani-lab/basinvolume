#include "basinvolume/find_nr_decorrelation_steps.h"
#include <iostream>
#include <sstream>
#include <string>

namespace bv {

FindNrDecorrelationSteps::FindNrDecorrelationSteps(
    const double desired_mean_rsm_displ_, const size_t nr_iterations_start_,
    const size_t nr_samples_average_, pele::Array<double> initial_coords_,
    const size_t boxdim_)
    : nr_iterations_start(nr_iterations_start_),
      nr_samples_average(nr_samples_average_),
      rsmd_tracker(initial_coords_, boxdim_, desired_mean_rsm_displ_) {}

size_t FindNrDecorrelationSteps::get_nr_decorrelation_steps() const {
  if (!done()) {
    std::stringstream message;
    message << "Only done " << nr_decorrelation_steps.count() << " out of "
            << nr_samples_average << " measurements.";
    std::cout << message.str() << std::endl << std::flush;
    throw std::runtime_error("FindNrDecorrelationSteps::get_nr_decorrelation_"
                             "steps: Illegal read attempt. " +
                             message.str());
  }
  return nr_decorrelation_steps.mean();
}

bool FindNrDecorrelationSteps::done() const {
  return nr_decorrelation_steps.count() == nr_samples_average;
}

void FindNrDecorrelationSteps::action(pele::Array<double> &coords,
                                      double energy, bool accepted,
                                      mcpele::MC *mc) {
  if (done()) {
    // this will trigger premature exit from the MC run loop
    mc->m_niter =
        std::numeric_limits<size_t>::max(); // can use terminate() when
                                            // avaialbe, leave for now
    return;
  }

  if (mc->get_iterations_count() < nr_iterations_start - 1) {
    return;
  } else if (mc->get_iterations_count() == nr_iterations_start - 1) {
    rsmd_tracker.reset(coords);
  } else {
    compute_and_update_rmsd(coords);
  }
}

void FindNrDecorrelationSteps::compute_and_update_rmsd(
    pele::Array<double> coords) {
  rsmd_tracker.check_next(coords);
  if (rsmd_tracker.sufficient_diffusion_happened()) {
    nr_decorrelation_steps.update(rsmd_tracker.get_nr_decorrelation_steps());
    rsmd_tracker.reset(coords);
  }
}

} // namespace bv
