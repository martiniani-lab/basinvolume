#ifndef _BV_CELL_LISTS_WITH_BREAK_LOOP_H
#define _BV_CELL_LISTS_WITH_BREAK_LOOP_H

#include "pele/cell_lists.hpp"

#include "cell_loop_break.h"

namespace pele {

template <typename distance_policy = periodic_distance<3>>
class CellListsWithBreak : public CellLists<distance_policy> {
public:
  static const size_t m_ndim = CellLists<distance_policy>::m_ndim;
  virtual ~CellListsWithBreak() {}
  CellListsWithBreak(std::shared_ptr<distance_policy> dist,
                     pele::Array<double> const boxv, const double rcut,
                     const double ncellx_scale = 1.0)
      : CellLists<distance_policy>(dist, boxv, rcut, ncellx_scale) {}
  template <class callback_class>
  CellListsLoopBreak<callback_class, distance_policy>
  get_atom_pair_looper_break(callback_class &callback) const {
    return CellListsLoopBreak<callback_class, distance_policy>(
        callback, CellLists<distance_policy>::m_container,
        CellLists<distance_policy>::m_lattice_tool);
  }
};

} // namespace pele

#endif // #ifndef _BV_CELL_LISTS_WITH_BREAK_LOOP_H
