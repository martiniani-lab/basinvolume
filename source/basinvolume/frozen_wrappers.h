#ifndef _BV_FROZEN_WRAPPERS_H
#define _BV_FROZEN_WRAPPERS_H

#include <cmath>
#include <memory>
#include <stdexcept>

#include "pele/array.h"
#include "pele/distance.h"

#include "mcpele/mc.h"
#include "pele/neighbor_iterator.h"
#include "pele/frozen_atoms.h"

namespace bv {


/**
 * ConfTest frozen wrapper
 */
template <typename ConfTestType>
class ConfTestFrozenWrapper : public mcpele::ConfTest {
public:
    pele::FrozenCoordsConverter coords_converter;
protected:
    virtual ~ConfTestFrozenWrapper() {}
    std::shared_ptr<ConfTestType> _underlying_conftest;
    ConfTestFrozenWrapper(std::shared_ptr<ConfTestType> conftest,
            pele::Array<double> const &reference_coords,
            pele::Array<size_t> const & frozen_dof) :
        coords_converter(reference_coords, frozen_dof),
        _underlying_conftest(conftest)
    {}
    inline bool conf_test(pele::Array<double> &reduced_coords, mcpele::MC * mc)
    {
        if (reduced_coords.size() != coords_converter.ndof_mobile()){
            throw std::runtime_error("reduced coords does not have the right size");
        }
        pele::Array<double> full_coords(coords_converter.get_full_coords(reduced_coords));
        return _underlying_conftest->conf_test(full_coords, mc);
    }
};

} // namespace bv

#endif // #ifndef _BV_CHECK_OVERLAP_H
