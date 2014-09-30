#ifndef _BV_CHECK_OVERLAP_H
#define _BV_CHECK_OVERLAP_H

#include <cmath>
#include <memory>
#include <stdexcept>

#include "pele/array.h"
#include "pele/distance.h"

#include "mcpele/mc.h"
#include "pele/neighbor_iterator.h"

namespace bv{


/**
 * Test for overlap of the hard sphere cores
 */
template<typename DIST_POL>
class CheckOverlap:public mcpele::ConfTest{
protected:
    const static size_t m_ndim = DIST_POL::_ndim;
    pele::Array<double> m_hs_radii;
    size_t m_nparticles;
    std::shared_ptr<DIST_POL> m_periodic_dist;

public:
    CheckOverlap(pele::Array<double> hs_radii, std::shared_ptr<DIST_POL> dist=NULL)
        : m_hs_radii(hs_radii.copy()),
          m_nparticles(m_hs_radii.size()),
          m_periodic_dist(dist)
    {
        if (m_periodic_dist == NULL)
            throw std::runtime_error("CheckOverlap::periodic distance uninitialised");
    }

    virtual ~CheckOverlap() {};

    bool conf_test(pele::Array<double> &trial_coords, mcpele::MC * mc)
    {
        double dr[m_ndim];

        for (size_t i=0;i<m_nparticles;++i){
            size_t i1 = m_ndim*i;
            for (size_t j= i + 1;j<m_nparticles;++j){
                size_t j1 = m_ndim*j;
                m_periodic_dist->get_rij(dr, &trial_coords[i1], &trial_coords[j1]);
                double dij2 = 0;
                for (size_t k =0;k<m_ndim;++k){
                    dij2 += dr[k]*dr[k];
                }
                double tmp = (m_hs_radii[i] + m_hs_radii[j]);
                if (dij2 < tmp * tmp) {
                    return false;
                }
            }
        }
        return true;
    }

};

template<size_t ndim>
class CheckOverlapPeriodic:public CheckOverlap<pele::periodic_distance<ndim> >{
public:
    CheckOverlapPeriodic(pele::Array<double> hs_radii, pele::Array<double> boxvec)
        : CheckOverlap< pele::periodic_distance<ndim> >(hs_radii,
                std::make_shared<pele::periodic_distance<ndim> >(boxvec))
    {}
};


/*
 * Test for overlap of the hard sphere cores
*/

template<typename DIST_POL>
class CellListCheckOverlap:public mcpele::ConfTest{
protected:
    const static size_t m_ndim = DIST_POL::_ndim;
    pele::Array<double> m_hs_radii;
    size_t m_nparticles;
    std::shared_ptr<DIST_POL> m_periodic_dist;
    std::shared_ptr<pele::CellIter<DIST_POL> > m_celliter;

public:
    CellListCheckOverlap(pele::Array<double> hs_radii,
            std::shared_ptr<DIST_POL> dist=NULL, std::shared_ptr<pele::CellIter<DIST_POL> > celliter=NULL)
        :   m_hs_radii(hs_radii.copy()),
            m_nparticles(m_hs_radii.size()),
            m_periodic_dist(dist),
            m_celliter(celliter)
    {
        if (m_periodic_dist == NULL || m_celliter == NULL)
            throw std::runtime_error("CheckOverlap::periodic distance uninitialised");
    }

    virtual ~CellListCheckOverlap() {};

    bool conf_test(pele::Array<double> &trial_coords, mcpele::MC * mc)
    {
        //refresh cell lists
        m_celliter->reset(trial_coords);

        const double* x = trial_coords.data();

        for (auto ijpair = m_celliter->begin(); ijpair != m_celliter->end(); ++ijpair)
        {
            const size_t i = ijpair->first;
            const size_t j = ijpair->second;
            const size_t xi_off = m_ndim * i;
            const size_t xj_off = m_ndim * j;
            double dr[m_ndim];
            m_periodic_dist->get_rij(dr, x + xi_off, x + xj_off);
            double dij2 = 0;
            for (size_t k = 0; k < m_ndim; ++k) {
                dij2 += dr[k] * dr[k];
            }
            double tmp = (m_hs_radii[i] + m_hs_radii[j]);
            if (dij2 < tmp * tmp) {
                return false;
            }
        }
        return true;
    }
};

template<size_t ndim>
class CheckOverlapPeriodicCellLists : public CellListCheckOverlap< pele::periodic_distance<ndim> > {
public:
    CheckOverlapPeriodicCellLists(pele::Array<double> coords, pele::Array<double> hs_radii, pele::Array<double> boxvec, double rcut, double ncellx_scale = 1.0)
    : CellListCheckOverlap< pele::periodic_distance<ndim> >(hs_radii,
            std::make_shared<pele::periodic_distance<ndim> >(boxvec),
            std::make_shared<pele::CellIter<pele::periodic_distance<ndim> > >(coords, boxvec, rcut, ncellx_scale))
    {}
};

}//namespace bv

#endif//#ifndef _BV_CHECK_OVERLAP_H
