#ifndef _BV_CELL_LOOP_BREAK_H
#define _BV_CELL_LOOP_BREAK_H

#include "pele/cell_lists.h"

namespace pele {

/**
 * Looping over atom pair similar to CellListsLoop.
 * The loop in loop_through_atom_pairs can terminate based on the result
 * of m_visitor.insert_atom_pair(atomi, atomj).
 * This is used to have a cell-list-based overlap check.
 */
template <class visitor_t, size_t ndim>
class CellListsLoopBreak : public CellListsLoop<visitor_t, ndim> {
public:
    virtual ~CellListsLoopBreak() {}
    CellListsLoopBreak(visitor_t& visitor, CellListsContainer<ndim> const& container)
        : CellListsLoop<visitor_t, ndim>(visitor, container)
    {}
    void loop_through_atom_pairs()
    {
        typename CellListsContainer<ndim>::const_iterator iiter, jiter, iend, jend;
        iend = CellListsLoop<visitor_t, ndim>::m_container.end();
        for (auto const& ijpair : CellListsLoop<visitor_t, ndim>::m_container.m_cell_neighbor_pairs) {
            const size_t icell = ijpair.first;
            const size_t jcell = ijpair.second;
            for (iiter = CellListsLoop<visitor_t, ndim>::m_container.begin(icell); iiter != iend; ++iiter) {
                jend = (icell == jcell) ? iiter : CellListsLoop<visitor_t, ndim>::m_container.end();
                for (jiter = CellListsLoop<visitor_t, ndim>::m_container.begin(jcell); jiter != jend; ++jiter) {
                    if (CellListsLoop<visitor_t, ndim>::m_visitor.insert_atom_pair(*iiter, *jiter)) {
                        return;
                    }
                }
            }
        }
    }
};

} // namespace pele

#endif // #ifndef _BV_CELL_LOOP_BREAK_H
