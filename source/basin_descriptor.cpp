#include "basinvolume/inside_basin_test.h"

namespace bv {

InsideBasinTest::InsideBasinTest(AbstractGradientBasin* basin)
    : _basin(basin), _collect_attractors(false) {
    auto collectable_basin =
        dynamic_cast<AbstractBasinCollector *>(_basin);
    if (collectable_basin) {
        _collect_attractors = true;
    }
}

bool InsideBasinTest::conf_test(
    pele::Array<double> &trial_coords, mcpele::MC *mc) {
    
    auto optimizer = _basin->get_optimizer();
    optimizer->reset(trial_coords);

    const size_t opt_maxiter = optimizer->get_maxiter();

    bool is_basin_attractor = true;
    bool optimizer_converged = true;

    while (static_cast<size_t>(optimizer->get_niter()) < opt_maxiter && !is_basin_attractor) {
        optimizer_converged = optimizer->stop_criterion_satisfied();
        
        if (optimizer_converged) {
            is_basin_attractor = false;
            if (_collect_attractors) {
                auto collectable_basin = dynamic_cast<AbstractBasinCollector*>(_basin);
                if (collectable_basin) {
                    collectable_basin->collect_attractor(optimizer->get_x(), optimizer);
                }
            }
            break;
        }
        optimizer->one_iteration();
        is_basin_attractor = _basin->is_same_attractor(optimizer->get_x());
    }
    // add number of energy evaluations to mc eval count
    const size_t nfev = optimizer->get_nfev();
    mc->m_neval += nfev;
    _failed_optimizations.update(!optimizer_converged);
    return is_basin_attractor;
}
} // namespace bv


