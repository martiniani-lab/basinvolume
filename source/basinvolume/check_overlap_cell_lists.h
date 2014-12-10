#ifndef _BV_CHECK_OVERLAP_CELL_LISTS_H
#define _BV_CHECK_OVERLAP_CELL_LISTS_H

#include "cell_loop_break.h"

namespace bv {

/**
 * Test for overlap of the hard sphere cores
 */

template <typename DIST_POL>
class CellListCheckOverlap : public mcpele::ConfTest {
protected:
    const static size_t m_ndim = DIST_POL::_ndim;
    pele::Array<double> m_hs_radii;
    const size_t m_nparticles;
    std::shared_ptr<DIST_POL> m_dist;
    std::shared_ptr<pele::CellLists<DIST_POL> > m_celliter;
public:
    virtual ~CellListCheckOverlap() {};
    CellListCheckOverlap(pele::Array<double> hs_radii,
            std::shared_ptr<DIST_POL> dist, std::shared_ptr<pele::CellLists<DIST_POL> > celliter)
        :   m_hs_radii(hs_radii.copy()),
            m_nparticles(m_hs_radii.size()),
            m_dist(dist),
            m_celliter(celliter)
    {
        if (m_dist == NULL || m_celliter == NULL) {
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
        //refresh cell lists
        m_celliter->reset(trial_coords);
        const double* x = trial_coords.data();
        for (auto ijpair = m_celliter->begin(); ijpair != m_celliter->end(); ++ijpair) {
            const size_t i = ijpair->first;
            const size_t j = ijpair->second;
            const size_t xi_off = m_ndim * i;
            const size_t xj_off = m_ndim * j;
            double dr[m_ndim];
            m_dist->get_rij(dr, x + xi_off, x + xj_off);
            const double dij2 = std::inner_product(dr, dr + m_ndim, dr, double(0));
            const double tmp = (m_hs_radii[i] + m_hs_radii[j]);
            if (dij2 < tmp * tmp) {
                return false;
            }
        }
        return true;
    }
};

template <size_t ndim>
class CheckOverlapPeriodicCellLists : public CellListCheckOverlap< pele::periodic_distance<ndim> > {
public:
    CheckOverlapPeriodicCellLists(pele::Array<double> hs_radii, pele::Array<double> boxvec,
            double rcut, double ncellx_scale=1.0)
    : CellListCheckOverlap< pele::periodic_distance<ndim> >(hs_radii,
            std::make_shared<pele::periodic_distance<ndim> >(boxvec),
            std::make_shared<pele::CellLists<pele::periodic_distance<ndim> > >(std::make_shared<pele::periodic_distance<ndim> >(boxvec), boxvec, rcut, ncellx_scale))
    {}
};

template<size_t ndim>
class CheckOverlapPeriodicCellListsFrozen : public ConfTestFrozenWrapper<CheckOverlapPeriodicCellLists<ndim> > {
public:
    CheckOverlapPeriodicCellListsFrozen(pele::Array<double> reference_coords,
            pele::Array<size_t>& frozen_dof, pele::Array<double> hs_radii,
            pele::Array<double> boxvec, double rcut, double ncellx_scale=1.0)
        : ConfTestFrozenWrapper< CheckOverlapPeriodicCellLists<ndim> > (
                std::make_shared<CheckOverlapPeriodicCellLists<ndim> >(
                        hs_radii, boxvec, rcut, ncellx_scale),
                        reference_coords.copy(), frozen_dof.copy())
    {}
};

template<size_t ndim>
class CheckOverlapCartesianCellLists : public CellListCheckOverlap<pele::cartesian_distance<ndim> > {
public:
    CheckOverlapCartesianCellLists(pele::Array<double> hs_radii, pele::Array<double> boxvec,
            double rcut, double ncellx_scale=1.0)
        : CellListCheckOverlap<pele::cartesian_distance<ndim> >(hs_radii,
                std::make_shared<pele::cartesian_distance<ndim> >(),
                std::make_shared<pele::CellLists<pele::cartesian_distance<ndim> > >(std::make_shared<pele::cartesian_distance<ndim> >(), boxvec, rcut, ncellx_scale))
    {}
};

template<size_t ndim>
class CheckOverlapCartesianCellListsFrozen : public ConfTestFrozenWrapper<CheckOverlapCartesianCellLists<ndim> > {
public:
    CheckOverlapCartesianCellListsFrozen(pele::Array<double> reference_coords,
            pele::Array<size_t>& frozen_dof, pele::Array<double> hs_radii,
            pele::Array<double> boxvec, double rcut, double ncellx_scale=1.0)
        : ConfTestFrozenWrapper< CheckOverlapCartesianCellLists<ndim> > (
                std::make_shared<CheckOverlapCartesianCellLists<ndim> >(
                        hs_radii, boxvec, rcut, ncellx_scale),
                        reference_coords.copy(), frozen_dof.copy())
    {}
};

} // namespace bv

#endif // #ifndef _BV_CHECK_OVERLAP_CELL_LISTS_H
