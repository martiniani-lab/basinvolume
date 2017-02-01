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

    void loop_through_atom_pairs(pele::Array<double> const & coords)
    {
        for (auto const & ijpair : CellListsLoop<visitor_t, ndim>::m_container.m_cell_neighbor_pairs) {
            const std::vector<size_t>* icell = ijpair.first;
            const std::vector<size_t>* jcell = ijpair.second;
            // do double loop through atoms, avoiding duplicate pairs
            for (auto iatom = icell->begin(); iatom != icell->end(); ++iatom) {
                // if icell==jcell we need to avoid duplicate atom pairs
                auto jend = (icell == jcell) ? iatom : jcell->end();
                for (auto jatom = jcell->begin(); jatom != jend; ++jatom) {
                    if (CellListsLoop<visitor_t, ndim>::m_visitor.insert_atom_pair(coords, *iatom, *jatom)) {
                        return;
                    }
                }
            }
        }
    }
};

} // namespace pele

#endif // #ifndef _BV_CELL_LOOP_BREAK_H
