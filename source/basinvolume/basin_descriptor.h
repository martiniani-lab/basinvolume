/*
* Abstract class that defines a BasinDescriptor for basin volume calculation
* A BasinDescriptor should be defined by 
* 1. A potential
* 2. An identification function that identifies whether two minimized configurations
*    are the same minimum (i.e account for symmetries etc)
* 3. a descent algorithm, could be an optimization algorithm or an ODE solver
*    though the interface is defined in pele and can be phrased as a descent problem
*/

#include <memory>

#include "pele/array.hpp"
#include "pele/optimizer.hpp"
#include "mcpele/mc.h"

namespace bv {

// Forward declarations for types that may be defined elsewhere
class InsideBasinStatistics;
class MinimaList;



class AbstractBasinDescriptor {
    protected:
        const std::shared_ptr<pele::BasePotential> _potential;
        const std::shared_ptr<pele::GradientOptimizer> _optimizer;
        const double _dtol;
    public:
        AbstractBasinDescriptor(std::unique_ptr<pele::BasePotential> potential, std::unique_ptr<pele::GradientOptimizer> optimizer);
        virtual ~AbstractBasinDescriptor() = default;
        virtual bool is_same_minimum(const pele::Array<double> &minimum, const pele::Array<double> &minimized_trial) const = 0;

        const std::shared_ptr<pele::GradientOptimizer> get_optimizer() const { return _optimizer; }
        const std::shared_ptr<pele::BasePotential> get_potential() const { return _potential; }
};


class InsideBasin : public mcpele::ConfTest {
    protected:
        std::unique_ptr<AbstractBasinDescriptor> _basin_descriptor;
        pele::Array<double> _minimum;
        InsideBasinStatistics _statistics;
        MinimaList _minima_list;// 
        double _dtol;
        
    public:
        InsideBasin(std::unique_ptr<AbstractBasinDescriptor> basin_descriptor,
         pele::Array<double> const &minimum);
        virtual ~InsideBasin() = default;
    virtual bool conf_test(pele::Array<double> &trial_coords, mcpele::MC *mc) override;
    virtual bool _quench(pele::Array<double> &trial_coords);
};


bool InsideBasin::_quench(
    pele::Array<double> &trial_coords) {
    
    auto optimizer = _basin_descriptor->get_optimizer();
    optimizer->reset(trial_coords);
    bool success = true;
    const size_t opt_maxiter = optimizer->get_maxiter();
    while (static_cast<size_t>(optimizer->get_niter()) < opt_maxiter) {
        if (optimizer->stop_criterion_satisfied()) {
            success = false;
            break;
        }
        optimizer->one_iteration();
        _basin_descriptor->is_same_minimum(_minimum, optimizer->get_x());
    }
    return success;
}



























} // namespace bv















