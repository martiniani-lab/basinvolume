#ifndef _BV_CHECK_OVERLAP_H
#define _BV_CHECK_OVERLAP_H

#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "pele/array.h"
#include "pele/distance.h"
#include "pele/neighbor_iterator.h"

#include "mcpele/mc.h"

#include "frozen_wrappers.h"

namespace bv {

/**
 * Test for overlap of the hard sphere cores
 */
template <typename DIST_POL>
class CheckOverlap : public mcpele::ConfTest {
protected:
    const static size_t m_ndim = DIST_POL::_ndim;
    pele::Array<double> m_hs_radii;
    const size_t m_nparticles;
    std::shared_ptr<DIST_POL> m_dist;
public:
    virtual ~CheckOverlap() {};
    CheckOverlap(pele::Array<double> hs_radii, std::shared_ptr<DIST_POL> dist)
        : m_hs_radii(hs_radii.copy()),
          m_nparticles(m_hs_radii.size()),
          m_dist(dist)
    {
        if (m_dist == NULL) {
            throw std::runtime_error("CheckOverlap: distance uninitialised");
        }
    }
    bool conf_test(pele::Array<double> &trial_coords, mcpele::MC * mc)
    {
        double dr[m_ndim];
        for (size_t i = 0; i < m_nparticles; ++i) {
            const size_t i1 = m_ndim * i;
            for (size_t j = i + 1; j < m_nparticles; ++j) {
                const size_t j1 = m_ndim * j;
                m_dist->get_rij(dr, &trial_coords[i1], &trial_coords[j1]);
                const double dij2 = std::inner_product(dr, dr + m_ndim, dr, double(0));
                const double tmp = (m_hs_radii[i] + m_hs_radii[j]);
                if (dij2 < tmp * tmp) {
                    return false;
                }
            }
        }
        return true;
    }

};

template <size_t ndim>
class CheckOverlapPeriodic : public CheckOverlap<pele::periodic_distance<ndim> > {
public:
    CheckOverlapPeriodic(pele::Array<double> hs_radii, pele::Array<double> boxvec)
        : CheckOverlap< pele::periodic_distance<ndim> >(hs_radii,
                std::make_shared<pele::periodic_distance<ndim> >(boxvec))
    {}
};

template<size_t ndim>
class CheckOverlapPeriodicFrozen : public ConfTestFrozenWrapper<CheckOverlapPeriodic<ndim> > {
public:
    CheckOverlapPeriodicFrozen(pele::Array<double> hs_radii, pele::Array<double> boxvec,
            pele::Array<double>& reference_coords, pele::Array<size_t>& frozen_dof)
        : ConfTestFrozenWrapper< CheckOverlapPeriodic<ndim> > (
                std::make_shared<CheckOverlapPeriodic<ndim> >(hs_radii, boxvec),
                reference_coords.copy(), frozen_dof.copy())
    {}
};

template <size_t ndim>
class CheckOverlapCartesian : public CheckOverlap<pele::cartesian_distance<ndim> > {
public:
    CheckOverlapCartesian(pele::Array<double> hs_radii)
        : CheckOverlap<pele::cartesian_distance<ndim> >(hs_radii,
                std::make_shared<pele::cartesian_distance<ndim> >())
    {}
};

template<size_t ndim>
class CheckOverlapCartesianFrozen : public ConfTestFrozenWrapper<CheckOverlapCartesian<ndim> > {
public:
    CheckOverlapCartesianFrozen(pele::Array<double> hs_radii,
            pele::Array<double>& reference_coords, pele::Array<size_t>& frozen_dof)
        : ConfTestFrozenWrapper< CheckOverlapCartesian<ndim> > (
                std::make_shared<CheckOverlapCartesian<ndim> >(hs_radii),
                reference_coords.copy(), frozen_dof.copy())
    {}
};

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
    std::shared_ptr<pele::CellIter<DIST_POL> > m_celliter;
public:
    virtual ~CellListCheckOverlap() {};
    CellListCheckOverlap(pele::Array<double> hs_radii,
            std::shared_ptr<DIST_POL> dist, std::shared_ptr<pele::CellIter<DIST_POL> > celliter)
        :   m_hs_radii(hs_radii.copy()),
            m_nparticles(m_hs_radii.size()),
            m_dist(dist),
            m_celliter(celliter)
    {
        if (m_dist == NULL || m_celliter == NULL) {
            throw std::runtime_error("CheckOverlap: distance uninitialised");
        }
    }
    bool conf_test(pele::Array<double> &trial_coords, mcpele::MC * mc)
    {
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
            std::make_shared<pele::CellIter<pele::periodic_distance<ndim> > >(std::make_shared<pele::periodic_distance<ndim> >(boxvec), boxvec, rcut, ncellx_scale))
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
                std::make_shared<pele::CellIter<pele::cartesian_distance<ndim> > >(std::make_shared<pele::cartesian_distance<ndim> >(), boxvec, rcut, ncellx_scale))
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

#endif // #ifndef _BV_CHECK_OVERLAP_H
