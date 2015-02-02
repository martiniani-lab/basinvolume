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
        for (auto const & ijpair : CellListsLoop<visitor_t, ndim>::m_cell_neighbor_pairs) {
            const size_t icell = ijpair.first;
            const size_t jcell = ijpair.second;
            // do double loop through atoms, avoiding duplicate pairs
            for (auto iiter = AtomInCellIterator<ndim>(CellListsLoop<visitor_t, ndim>::m_ll.data(), CellListsLoop<visitor_t, ndim>::m_hoc[icell]); !iiter.done(); ++iiter) {
                size_t const atomi = *iiter;
                // if icell==jcell we need to avoid duplicate atom pairs
                long const loop_end = (icell == jcell) ? atomi : CELL_END;
                for (auto jiter = AtomInCellIterator<ndim>(CellListsLoop<visitor_t, ndim>::m_ll.data(), CellListsLoop<visitor_t, ndim>::m_hoc[jcell], loop_end); !jiter.done(); ++jiter) {
                    size_t const atomj = *jiter;
                    const bool break_loop = CellListsLoop<visitor_t, ndim>::m_visitor.insert_atom_pair(atomi, atomj);
                    if (break_loop) {
                        return;
                    }
                }
            }
        }
    }

};

} // namespace pele

#endif // #ifndef _BV_CELL_LOOP_BREAK_H
