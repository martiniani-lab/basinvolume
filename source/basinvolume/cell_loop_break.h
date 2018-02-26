#ifndef _BV_CELL_LOOP_BREAK_H
#define _BV_CELL_LOOP_BREAK_H

#include "pele/cell_lists.h"

namespace pele {

// Type of each cell inside the cell list
using cell_t = std::vector<long>;

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

    void loop_cell_pairs(std::vector< std::array<cell_t*, 2> > const & neighbor_pairs,
                         const size_t isubdom)
    {
        for (auto const & ijpair : neighbor_pairs) {
            cell_t* icell = ijpair[0];
            cell_t* jcell = ijpair[1];
            // do double loop through atoms, avoiding duplicate pairs
            for (auto iatom = icell->begin(); iatom != icell->end(); ++iatom) {
                // if icell==jcell we need to avoid duplicate atom pairs
                auto jend = (icell == jcell) ? iatom : jcell->end();
                for (auto jatom = jcell->begin(); jatom != jend; ++jatom) {
                    if (CellListsLoop<visitor_t, distance_policy>::m_visitor.insert_atom_pair(*iatom, *jatom, isubdom)) {
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
            for (cell_t* jcell : m_container.m_cell_neighbors[icells[i]]) {
                for (auto jatom = jcell->begin(); jatom != jcell->end(); ++jatom) {
                    if (iatoms[i] != *jatom) {
                        if (CellListsLoop<visitor_t, distance_policy>::m_visitor.insert_atom_pair(iatoms[i], *jatom, isubdom)) {
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
