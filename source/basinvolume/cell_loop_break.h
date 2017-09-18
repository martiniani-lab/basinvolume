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
template <class visitor_t, typename distance_policy = periodic_distance<3> >
class CellListsLoopBreak : public CellListsLoop<visitor_t, distance_policy> {
    using CellListsLoop<visitor_t, distance_policy>::m_container;
    using CellListsLoop<visitor_t, distance_policy>::m_lattice_tool;
protected:
    typedef VecN<distance_policy::_ndim, size_t> cell_vec_t;
    bool m_break = false;
public:
    virtual ~CellListsLoopBreak() {}
    CellListsLoopBreak(
        visitor_t& visitor,
        CellListsContainer<distance_policy::_ndim> const& container,
        pele::LatticeNeighbors<distance_policy> lattice_tool)
        : CellListsLoop<visitor_t, distance_policy>(visitor, container, lattice_tool)
    {}

    void loop_cell_pairs(std::vector< std::array<long*, 2> > const & neighbor_pairs,
                         const size_t isubdom)
    {
        for (auto const & ijpair : neighbor_pairs) {
            // do double loop through atoms, avoiding duplicate pairs
            for (auto icell_iter = m_container.getIterator(ijpair[0]);
                 *icell_iter != CELL_END;
                 ++icell_iter) {
                // if icell==jcell we need to avoid duplicate atom pairs
                auto jend = (ijpair[0] == ijpair[1]) ? *icell_iter : CELL_END;
                for (auto jcell_iter = m_container.getIterator(ijpair[1]);
                     *jcell_iter != jend;
                     ++jcell_iter) {
                    if (CellListsLoop<visitor_t, distance_policy>::m_visitor.insert_atom_pair(*icell_iter, *jcell_iter, isubdom)) {
                        m_break = true;
                    }
                    if (m_break) {
                        return;
                    }
                }
            }
        }
    }

    void loop_cell_pairs_specific(std::vector<size_t> const & icells, std::vector<long> const & iatoms)
    {
        for (size_t i = 0; i < icells.size(); ++i) {
            size_t isubdom = m_lattice_tool.get_subdomain(icells[i]);
            for (long* jcell : m_container.m_cell_neighbors[icells[i]]) {
                for (auto jcell_iter = m_container.getIterator(jcell);
                     *jcell_iter != CELL_END;
                     ++jcell_iter) {
                    if (iatoms[i] != *jcell_iter) {
                        if (CellListsLoop<visitor_t, distance_policy>::m_visitor.insert_atom_pair(iatoms[i], *jcell_iter, isubdom)) {
                            m_break = true;
                        }
                        if (m_break) {
                            return;
                        }
                    }
                }
            }
        }
    }
};

} // namespace pele

#endif // #ifndef _BV_CELL_LOOP_BREAK_H
