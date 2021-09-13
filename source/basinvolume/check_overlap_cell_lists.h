#ifndef _BV_CHECK_OVERLAP_CELL_LISTS_H
#define _BV_CHECK_OVERLAP_CELL_LISTS_H

#include "pele/cell_lists.hpp"

#include "cell_lists_with_break_loop.h"

namespace bv {

template <class distance_policy>
class OverlapAccumulator {
private:
    const static size_t m_ndim = distance_policy::_ndim;
    std::shared_ptr<distance_policy> m_dist;
    const pele::Array<double> * m_coords;
    const pele::Array<double> m_radii;
    bool m_legal;
public:
    OverlapAccumulator(std::shared_ptr<distance_policy> & dist,
                       pele::Array<double> const & radii)
        : m_dist(dist),
          m_radii(radii),
          m_legal(true)
    {}

    void reset_data(const pele::Array<double> * coords) {
        m_coords = coords;
        m_legal = true;
    }

    bool configuration_is_legal() const {
        return m_legal;
    }

    double get_squared_atom_distance(double const * const r1, double const * const r2) const
    {
        double dr[m_ndim];
        m_dist->get_rij(dr, r1, r2);
        return std::inner_product(dr, dr + m_ndim, dr, double(0));
    }

    bool insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        const size_t xi_off = m_ndim * atom_i;
        const size_t xj_off = m_ndim * atom_j;
        const double dij2 = get_squared_atom_distance(m_coords->data() + xi_off, m_coords->data() + xj_off);
        const double radius_sum = m_radii[atom_i] + m_radii[atom_j];
        if(dij2 < radius_sum * radius_sum) {
            m_legal = false;
        }
        return !m_legal;
    }
};

template <typename distance_policy>
class CellListCheckOverlap : public mcpele::ConfTest {
protected:
  const static size_t m_ndim = distance_policy::_ndim;
  std::shared_ptr<distance_policy> m_dist;
  std::shared_ptr<pele::CellListsWithBreak<distance_policy> > m_cell_lists;
  const pele::Array<double> m_radii;
  OverlapAccumulator<distance_policy> m_overlap_acc;
  const bool m_specific;
  std::vector<long> m_last_changed_atoms;
  std::vector<double> m_last_changed_coords;

  
  void merge_last_and_current_changes(std::vector<long> & changed_atoms, std::vector<double> & changed_coords_old)
  { 
    for (size_t i = 0; i < m_last_changed_atoms.size(); ++i) {
      auto new_iatom = std::find(changed_atoms.begin(), changed_atoms.end(), m_last_changed_atoms[i]);
      if (new_iatom == changed_atoms.end()) {
        changed_atoms.push_back(m_last_changed_atoms[i]);
        for (size_t idim = 0; idim < m_ndim; ++idim) {
          changed_coords_old.push_back(m_last_changed_coords[i * m_ndim + idim]);
        }
      } else {
        size_t new_icoords = m_ndim * (new_iatom - changed_atoms.begin());
        for (size_t idim = 0; idim < m_ndim; ++idim) {
          changed_coords_old[new_icoords + idim] = m_last_changed_coords[i * m_ndim + idim];
        }
      }
    }
  }

    void save_changes(pele::Array<double> & trial_coords, mcpele::MC * mc)
    {
        m_last_changed_atoms = std::vector<long>(mc->get_changed_atoms());
        m_last_changed_coords = std::vector<double>(m_ndim * m_last_changed_atoms.size());
        for (size_t i = 0; i < m_last_changed_atoms.size(); ++i) {
            long iatom_ndim = m_last_changed_atoms[i] * m_ndim;
            size_t i_ndim = i * m_ndim;
            for (size_t j = 0; j < m_ndim; ++j) {
                m_last_changed_coords[i_ndim + j] = trial_coords[iatom_ndim + j];
            }
        }
    }


  
public:
    virtual ~CellListCheckOverlap() {};
    CellListCheckOverlap(pele::Array<double> & hs_radii, std::shared_ptr<distance_policy> dist, std::shared_ptr<pele::CellListsWithBreak<distance_policy> > cell_lists, bool specific)
        :   m_dist(dist),
            m_radii(hs_radii.copy()),
            m_cell_lists(cell_lists),
            m_overlap_acc(m_dist, m_radii),
            m_specific(specific)
    {
        if (m_dist == NULL || m_cell_lists == NULL) {
            throw std::runtime_error("CellListCheckOverlap: distance or celliter uninitialised");
        }
        if (m_dist == NULL) {
            throw std::runtime_error("CellListCheckOverlap: distance uninitialised");
        }
        if (hs_radii.size() == 0) {
            throw std::runtime_error("CellListCheckOverlap: illegal input: hs_radii");
        }
        static_assert(distance_policy::_ndim > 0, "CellListCheckOverlap: illegal input: distance policy");
    }

    bool conf_test(pele::Array<double> & trial_coords, mcpele::MC * mc)
    {
        if (trial_coords.size() % m_ndim) {
            throw std::runtime_error("CellListCheckOverlap::conf_test: illegal input");
        }
        if (trial_coords.size() / m_ndim != m_radii.size()) {
            throw std::runtime_error("CellListCheckOverlap::conf_test: illegal input");
        }

        if (!std::isfinite(trial_coords[0]) || !std::isfinite(*(trial_coords.end() - 1))) {
            return false;
        }

        std::vector<long> changed_atoms = mc->get_changed_atoms();
        // for (auto i = changed_atoms.begin(); i != changed_atoms.end(); ++i)
        //   std::cout << *i << ' ';

        // for (auto i : changed_atoms) // access by value, the type of i is int
        //   std::cout << i << ",  ";
        // std::cout <<  " ] changed aaatoms \n";

        
        if (m_specific && changed_atoms.size() > 0) {
          
          std::vector<double> changed_coords_old = mc->get_changed_coords_old();
          // for (auto i: changed_coords_old) {
          //   std::cout << *i << "changed coords old in \n";
          // };

          if (!mc->get_last_success()) {
            merge_last_and_current_changes(changed_atoms, changed_coords_old);
          }

          
          
          m_cell_lists->update_specific(trial_coords, changed_atoms, changed_coords_old);
          
          // std::cout << changed_atoms << "\n";
          // std::cout << changed_coords_old << "\n";
          // std::cout << "atom looper pair" << "\n";
          m_overlap_acc.reset_data(&trial_coords);
          // std::cout << "data reset" << "\n";
          auto joe_the_looper = m_cell_lists->get_atom_pair_looper_break(m_overlap_acc);

          // std::cout << "loop through atoms";
          // std::cout << trial_coords <<"\n";
          // ss
          joe_the_looper.loop_through_atom_pairs_specific(trial_coords, changed_atoms);
          // std::cout << "loop through atoms works" << "\n";
          save_changes(trial_coords, mc);
          // std::cout << "out if" << "\n";

        } else {
          // std::cout << "in else \n";
          // std::cout << trial_coords << " not changed \n";
          m_cell_lists->update(trial_coords);
          m_overlap_acc.reset_data(&trial_coords);
          auto joe_the_looper = m_cell_lists->get_atom_pair_looper_break(m_overlap_acc);
          joe_the_looper.loop_through_atom_pairs();
          // std::cout << "out else" << "\n";
        }

        return m_overlap_acc.configuration_is_legal();
    }
};

template <size_t ndim>
class CheckOverlapPeriodicCellLists : public CellListCheckOverlap<pele::periodic_distance<ndim> > {
public:
    CheckOverlapPeriodicCellLists(pele::Array<double> hs_radii, pele::Array<double> boxvec, bool specific, double ncellx_scale=1.0)
        : CellListCheckOverlap<pele::periodic_distance<ndim> >(hs_radii,
            std::make_shared<pele::periodic_distance<ndim> >(boxvec),
            std::make_shared<pele::CellListsWithBreak<pele::periodic_distance<ndim> > >(std::make_shared<pele::periodic_distance<ndim> >(boxvec), boxvec, 2 * hs_radii.get_max(), ncellx_scale),
            specific)
    {}
};

template<size_t ndim>
class CheckOverlapPeriodicCellListsFrozen : public ConfTestFrozenWrapper<CheckOverlapPeriodicCellLists<ndim> > {
public:
    CheckOverlapPeriodicCellListsFrozen(pele::Array<double> reference_coords,
            pele::Array<size_t>& frozen_dof, pele::Array<double> hs_radii,
            pele::Array<double> boxvec, bool specific, double ncellx_scale=1.0)
        : ConfTestFrozenWrapper< CheckOverlapPeriodicCellLists<ndim> > (
                std::make_shared<CheckOverlapPeriodicCellLists<ndim> >(
                        hs_radii, boxvec, specific, ncellx_scale),
                        reference_coords.copy(), frozen_dof.copy())
    {}
};

template<size_t ndim>
class CheckOverlapCartesianCellLists : public CellListCheckOverlap<pele::cartesian_distance<ndim> > {
public:
    CheckOverlapCartesianCellLists(pele::Array<double> hs_radii, pele::Array<double> boxvec, bool specific, double ncellx_scale=1.0)
        : CellListCheckOverlap<pele::cartesian_distance<ndim> >(hs_radii,
                std::make_shared<pele::cartesian_distance<ndim> >(),
                std::make_shared<pele::CellListsWithBreak<pele::cartesian_distance<ndim> > >(std::make_shared<pele::cartesian_distance<ndim> >(), boxvec, ncellx_scale),
                specific)
    {}
};

template<size_t ndim>
class CheckOverlapCartesianCellListsFrozen : public ConfTestFrozenWrapper<CheckOverlapCartesianCellLists<ndim> > {
public:
    CheckOverlapCartesianCellListsFrozen(pele::Array<double> reference_coords,
            pele::Array<size_t>& frozen_dof, pele::Array<double> hs_radii,
            pele::Array<double> boxvec, bool specific, double ncellx_scale=1.0)
        : ConfTestFrozenWrapper< CheckOverlapCartesianCellLists<ndim> > (
                std::make_shared<CheckOverlapCartesianCellLists<ndim> >(
                        hs_radii, boxvec, specific, ncellx_scale),
                        reference_coords.copy(), frozen_dof.copy())
    {}
};

template <size_t ndim>
class CheckOverlapLeesEdwardsCellLists : public CellListCheckOverlap<pele::leesedwards_distance<ndim> > {
public:
    CheckOverlapLeesEdwardsCellLists(pele::Array<double> hs_radii, pele::Array<double> boxvec, const double shear, bool specific, double ncellx_scale=1.0)
        : CellListCheckOverlap<pele::leesedwards_distance<ndim> >(hs_radii,
            std::make_shared<pele::leesedwards_distance<ndim> >(boxvec, shear),
            std::make_shared<pele::CellListsWithBreak<pele::leesedwards_distance<ndim> > >(std::make_shared<pele::leesedwards_distance<ndim> >(boxvec, shear), boxvec, 2 * hs_radii.get_max(), ncellx_scale), specific)
    {}
};

template<size_t ndim>
class CheckOverlapLeesEdwardsCellListsFrozen : public ConfTestFrozenWrapper<CheckOverlapLeesEdwardsCellLists<ndim> > {
public:
    CheckOverlapLeesEdwardsCellListsFrozen(pele::Array<double> reference_coords,
            pele::Array<size_t>& frozen_dof, pele::Array<double> hs_radii,
            pele::Array<double> boxvec, const double shear, bool specific, double ncellx_scale=1.0)
        : ConfTestFrozenWrapper< CheckOverlapLeesEdwardsCellLists<ndim> > (
                std::make_shared<CheckOverlapLeesEdwardsCellLists<ndim> >(
                        hs_radii, boxvec, shear, specific, ncellx_scale),
                        reference_coords.copy(), frozen_dof.copy())
    {}
};

} // namespace bv

#endif // #ifndef _BV_CHECK_OVERLAP_CELL_LISTS_H
