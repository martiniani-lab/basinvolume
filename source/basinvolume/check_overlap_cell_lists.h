#ifndef _BV_CHECK_OVERLAP_CELL_LISTS_H
#define _BV_CHECK_OVERLAP_CELL_LISTS_H

#include "pele/cell_lists.h"

#include "cell_lists_with_break_loop.h"

namespace bv {

template <class distance_policy>
class OverlapAccumulator {
public:
    const static size_t m_ndim = distance_policy::_ndim;
private:
    std::shared_ptr<distance_policy> m_dist;
    pele::Array<double> m_h;
    bool m_legal;
public:
    OverlapAccumulator(std::shared_ptr<distance_policy> dist, pele::Array<double> hs_radii)
        : m_dist(dist),
          m_h(hs_radii),
          m_legal(true)
    {}

    bool configuration_is_legal() const {
        return m_legal;
    }

    double get_squared_atom_distance(
        pele::Array<double> const & coords, const size_t atom_i, const size_t atom_j) const
    {
        double dr[m_ndim];
        m_dist->get_rij(dr, coords.data() + m_ndim * atom_i, coords.data() + m_ndim * atom_j);
        return std::inner_product(dr, dr + m_ndim, dr, double(0));
    }

    bool insert_atom_pair(pele::Array<double> const & coords, const size_t atom_i, const size_t atom_j)
    {
        const double dij2 = get_squared_atom_distance(coords, atom_i, atom_j);
        const double tmp = (m_h[atom_i] + m_h[atom_j]);
        if(dij2 < tmp * tmp) {
            m_legal = false;
        }
        return !m_legal;
    }
};

template <typename DIST_POL>
class CellListCheckOverlap : public mcpele::ConfTest {
protected:
    const static size_t m_ndim = DIST_POL::_ndim;
    pele::Array<double> m_hs_radii;
    const size_t m_nparticles;
    std::shared_ptr<DIST_POL> m_dist;
    std::shared_ptr<pele::CellListsWithBreak<DIST_POL> > m_cell_lists;
public:
    virtual ~CellListCheckOverlap() {};
    CellListCheckOverlap(pele::Array<double> & hs_radii, std::shared_ptr<DIST_POL> dist, std::shared_ptr<pele::CellListsWithBreak<DIST_POL> > cell_lists)
        :   m_hs_radii(hs_radii.copy()),
            m_nparticles(m_hs_radii.size()),
            m_dist(dist),
            m_cell_lists(cell_lists)
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
        static_assert(DIST_POL::_ndim > 0, "CellListCheckOverlap: illegal input: distance policy");
    }
    bool conf_test(pele::Array<double> &trial_coords, mcpele::MC * mc)
    {
        if (trial_coords.size() % m_ndim) {
            throw std::runtime_error("CellListCheckOverlap::conf_test: illegal input");
        }
        if (trial_coords.size() / m_ndim != m_nparticles) {
            throw std::runtime_error("CellListCheckOverlap::conf_test: illegal input");
        }
        m_cell_lists->update(trial_coords);
        OverlapAccumulator<DIST_POL> acc(m_dist, m_hs_radii);
        pele::CellListsLoopBreak<OverlapAccumulator<DIST_POL>, m_ndim> joe_the_looper = m_cell_lists->get_atom_pair_looper_break(acc);
        joe_the_looper.loop_through_atom_pairs(trial_coords);
        return acc.configuration_is_legal();
    }
};

template <size_t ndim>
class CheckOverlapPeriodicCellLists : public CellListCheckOverlap<pele::periodic_distance<ndim> > {
public:
    CheckOverlapPeriodicCellLists(pele::Array<double> hs_radii, pele::Array<double> boxvec, double ncellx_scale=1.0)
        : CellListCheckOverlap<pele::periodic_distance<ndim> >(hs_radii,
            std::make_shared<pele::periodic_distance<ndim> >(boxvec),
            std::make_shared<pele::CellListsWithBreak<pele::periodic_distance<ndim> > >(std::make_shared<pele::periodic_distance<ndim> >(boxvec), boxvec, 2 * hs_radii.get_max(), ncellx_scale))
    {}
};

template<size_t ndim>
class CheckOverlapPeriodicCellListsFrozen : public ConfTestFrozenWrapper<CheckOverlapPeriodicCellLists<ndim> > {
public:
    CheckOverlapPeriodicCellListsFrozen(pele::Array<double> reference_coords,
            pele::Array<size_t>& frozen_dof, pele::Array<double> hs_radii,
            pele::Array<double> boxvec, double ncellx_scale=1.0)
        : ConfTestFrozenWrapper< CheckOverlapPeriodicCellLists<ndim> > (
                std::make_shared<CheckOverlapPeriodicCellLists<ndim> >(
                        hs_radii, boxvec, ncellx_scale),
                        reference_coords.copy(), frozen_dof.copy())
    {}
};

template<size_t ndim>
class CheckOverlapCartesianCellLists : public CellListCheckOverlap<pele::cartesian_distance<ndim> > {
public:
    CheckOverlapCartesianCellLists(pele::Array<double> hs_radii, pele::Array<double> boxvec, double ncellx_scale=1.0)
        : CellListCheckOverlap<pele::cartesian_distance<ndim> >(hs_radii,
                std::make_shared<pele::cartesian_distance<ndim> >(),
                std::make_shared<pele::CellListsWithBreak<pele::cartesian_distance<ndim> > >(std::make_shared<pele::cartesian_distance<ndim> >(), boxvec, ncellx_scale))
    {}
};

template<size_t ndim>
class CheckOverlapCartesianCellListsFrozen : public ConfTestFrozenWrapper<CheckOverlapCartesianCellLists<ndim> > {
public:
    CheckOverlapCartesianCellListsFrozen(pele::Array<double> reference_coords,
            pele::Array<size_t>& frozen_dof, pele::Array<double> hs_radii,
            pele::Array<double> boxvec, double ncellx_scale=1.0)
        : ConfTestFrozenWrapper< CheckOverlapCartesianCellLists<ndim> > (
                std::make_shared<CheckOverlapCartesianCellLists<ndim> >(
                        hs_radii, boxvec, ncellx_scale),
                        reference_coords.copy(), frozen_dof.copy())
    {}
};

template <size_t ndim>
class CheckOverlapLeesEdwardsCellLists : public CellListCheckOverlap<pele::leesedwards_distance<ndim> > {
public:
    CheckOverlapLeesEdwardsCellLists(pele::Array<double> hs_radii, pele::Array<double> boxvec, const double shear, double ncellx_scale=1.0)
        : CellListCheckOverlap<pele::leesedwards_distance<ndim> >(hs_radii,
            std::make_shared<pele::leesedwards_distance<ndim> >(boxvec, shear),
            std::make_shared<pele::CellListsWithBreak<pele::leesedwards_distance<ndim> > >(std::make_shared<pele::leesedwards_distance<ndim> >(boxvec, shear), boxvec, 2 * hs_radii.get_max(), ncellx_scale))
    {}
};

template<size_t ndim>
class CheckOverlapLeesEdwardsCellListsFrozen : public ConfTestFrozenWrapper<CheckOverlapLeesEdwardsCellLists<ndim> > {
public:
    CheckOverlapLeesEdwardsCellListsFrozen(pele::Array<double> reference_coords,
            pele::Array<size_t>& frozen_dof, pele::Array<double> hs_radii,
            pele::Array<double> boxvec, const double shear, double ncellx_scale=1.0)
        : ConfTestFrozenWrapper< CheckOverlapLeesEdwardsCellLists<ndim> > (
                std::make_shared<CheckOverlapLeesEdwardsCellLists<ndim> >(
                        hs_radii, boxvec, shear, ncellx_scale),
                        reference_coords.copy(), frozen_dof.copy())
    {}
};

} // namespace bv

#endif // #ifndef _BV_CHECK_OVERLAP_CELL_LISTS_H
