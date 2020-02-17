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
      
        size_t isubdom = m_lattice_tool.get_subdomain(icells.at(i));
        // std::cout << m_container.m_cell_neighbors[icells.at(i)] << "\n";
        for (cell_t* jcell : m_container.m_cell_neighbors[icells.at(i)]) {
          // std::cout << m_container.m_cell_neighbors;
          //  whyd does m_container have m_cell_neighbors,
          // How are the neighbors of this cell initialized
          // std::cout << m_container.m_cell_neighbors[0] << "\n";
          // std::cout << jcell->size() << "size of jcell" << "\n";
          for (auto jatom = jcell->begin(); jatom != jcell->end(); ++jatom) {
            // std::cout << icells[i] << " icells[i] \n";
            // std::cout << i << " i \n";
            // std::cout << *(jcell->begin()) << " begin \n";
            // std::cout << *(jcell->end()) << " end \n";
            // std::cout << jcell->size() << "size of jcell" << "\n";
            // std::cout << icells.size() << " icells size loop started \n";
            // std::cout << jcell->empty() << "empty \n";
            // std::cout << jcell->begin() << "\n";
            // std::cout << jcell->end() << "\n";
            // std::cout << "third part does" << "\n";
            // std::cout << iatoms[i] << "\n";
            // std::cout << "seems to do the thing" << "\n";
            // std::cout << "m_visitor \n";
            if (iatoms.at(i) != *jatom) {
              // std::cout <<  *jatom <<" \n";
              // std::cout << iatoms[i] << "\n";
              // std::cout << isubdom  << "\n";
              // std::cout << visitor_t << "\n";
              if (CellListsLoop<visitor_t, distance_policy>::m_visitor.insert_atom_pair(iatoms.at(i), *jatom, isubdom)) {
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
