#ifndef _BV_CELL_LISTS_WITH_BREAK_LOOP_H
#define _BV_CELL_LISTS_WITH_BREAK_LOOP_H

#include "pele/cell_lists.h"

#include "cell_loop_break.h"

namespace pele {
    
template<typename distance_policy=periodic_distance<3> >
class CellListsWithBreak : public CellLists<distance_policy> {
public:
    typedef typename CellLists<distance_policy>::container_type container_type;
    virtual ~CellListsWithBreak() {}
    CellListsWithBreak(std::shared_ptr<distance_policy> dist, pele::Array<double> const boxv, const double rcut, const double ncellx_scale=1.0)
        : CellLists<distance_policy>(dist, boxv, rcut, ncellx_scale)
    {}
    template<class T>
    CellListsLoopBreak<T> get_atom_pair_looper_break(T& callback) const
    {
        return CellListsLoopBreak<T>(callback, CellLists<distance_policy>::m_container);
    }
};

} // namespace pele

#endif // #ifndef _BV_CELL_LISTS_WITH_BREAK_LOOP_H
