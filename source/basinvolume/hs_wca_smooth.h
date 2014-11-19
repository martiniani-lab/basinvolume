#ifndef _BV_HS_WCA_Smooth_H
#define _BV_HS_WCA_Smooth_H

#include "pele/simple_pairwise_potential.h"
#include "pele/simple_pairwise_ilist.h"
#include "pele/atomlist_potential.h"
#include "pele/distance.h"
#include "pele/frozen_atoms.h"
#include <memory>

namespace bv {

/**
 * Pairwise interaction for Hard Sphere + Weeks-Chandler-Andersen (HS_WCA_Smooth) potential, refer to D. Asenjo PhD thesis pp 66
 * _prfac is the cubic power of _sca/(2**(1/6))
 * well depth _eps and scaling factor (shell thickness = sca * R, where R is the hard core radius), sca determined the thickness of the shell
 */
struct HS_WCA_Smooth_interaction {
    double const _eps, _sca;
    double const _infty, _prfac, _c, _gamma;
    pele::Array<double> const _radii;

    HS_WCA_Smooth_interaction(double eps, double sca, pele::Array<double> radii)
        : _eps(eps), _sca(sca),
          _infty(std::pow(10.0,80)), _prfac(1./std::sqrt(2)), _c(0.5*(1+std::sqrt(1+(_infty-eps)/eps))),
          _gamma(_sca/(std::pow(2*_c,1./6)-1)),
          _radii(radii.copy())
    {}

    /* calculate energy from distance squared, r0 is the hard core distance, r is the distance between the centres */
    double inline energy(double r2, size_t atomi, size_t atomj) const 
    {
        double E;
        double r = sqrt(r2);
        double r0 = _radii[atomi] + _radii[atomj]; //sum of the hard core radii
        double C = (_sca+_gamma)*r0;
        double C3 = _prfac*C*C*C;
        double C6 = C3*C3;
        double C12 = C6*C6;
        double coff = r0*(1.0 +_sca); //distance at which the soft cores are at contact
        double dr = r + r0*(_gamma - 1);
        double dr2 = dr*dr;
        double dr6 = dr2*dr2*dr2;
        double ir6 = 1./(dr6);
        double ir12 = 1./(dr6*dr6);

        if (r <= r0) {
            double gamma = _gamma*r0;
            double gamma3 = gamma*gamma*gamma;
            double gamma6 = gamma3*gamma3;
            double gamma12 = gamma6*gamma6;
            double m = 4 * _eps * (6*C6/(gamma6*gamma) - 12*C12/(gamma12*gamma));
            E = m*(r-r0) + _infty;
            //std::cout<<"WARNING: distance between atoms "<<atomi<<" and "<<atomj<<" is "<<r0-r<<", less than their hard core separation"<<std::endl;
        }
        else if (r < coff )
            E = 4.*_eps*(-C6*ir6 + C12*ir12) + _eps;
        else
            E = 0.;

        return E;
    }

    /* calculate energy and gradient from distance squared, gradient is in g/|rij|, r0 is the hard core distance, r is the distance between the centres */
    double inline energy_gradient(double r2, double *gij, size_t atomi, size_t atomj) const 
    {
        double E;
        double r = sqrt(r2);
        double r0 = _radii[atomi] + _radii[atomj]; //sum of the hard core radii
        double C = (_sca+_gamma)*r0;
        double C3 = _prfac*C*C*C;
        double C6 = C3*C3;
        double C12 = C6*C6;
        double coff = r0*(1.0 +_sca); //distance at which the soft cores are at contact
        double dr = r + r0*(_gamma - 1);
        double dr2 = dr*dr;
        double dr6 = dr2*dr2*dr2;
        double ir6 = 1./(dr6);
        double ir12 = 1./(dr6*dr6);

        if (r <= r0) {
            double gamma = _gamma*r0;
            double gamma3 = gamma*gamma*gamma;
            double gamma6 = gamma3*gamma3;
            double gamma12 = gamma6*gamma6;
            double m = 4 * _eps * (6*C6/(gamma6*gamma) - 12*C12/(gamma12*gamma));
            E = m*(r-r0) + _infty;
            *gij = -m/r;
            //std::cout<<"WARNING: distance between atoms "<<atomi<<" and "<<atomj<<" is "<<r0-r<<"less than their hard core separation"<<std::endl;
        }
        else if (r < coff) {
            E = 4.*_eps*(- C6 * ir6 + C12 * ir12) + _eps;
            *gij = 4.*_eps*(- 6 * C6 * ir6 + 12 * C12 * ir12) / (dr*r); //1/dr because powers must be 7 and 13, this is -g|gij| (for consistency with the loop in pairwise potential)
            //*gij = 4.*_eps*(- 6 * C6 * dr6 * ir6 * ir6 + 12 * C12 * dr12 * ir12 * ir12) / (dr*r);
        }
        else {
            E = 0.;
            *gij = 0.;
        }

        return E;

    }

    double inline energy_gradient_hessian(double r2, double *gij, double *hij, size_t atomi, size_t atomj) const
    {
        double E;
        double r = sqrt(r2);
        double r0 = _radii[atomi] + _radii[atomj]; //sum of the hard core radii
        double C = (_sca+_gamma)*r0;
        double C3 = _prfac*C*C*C;
        double C6 = C3*C3;
        double C12 = C6*C6;
        double coff = r0*(1.0 +_sca); //distance at which the soft cores are at contact
        double dr = r + r0*(_gamma - 1);
        double dr2 = dr*dr;
        double dr6 = dr2*dr2*dr2;
        double ir6 = 1./(dr6);
        double ir12 = 1./(dr6*dr6);

        if (r <= r0) {
            double gamma = _gamma*r0;
            double gamma3 = gamma*gamma*gamma;
            double gamma6 = gamma3*gamma3;
            double gamma12 = gamma6*gamma6;
            double m = 4 * _eps * (6*C6/(gamma6*gamma) - 12*C12/(gamma12*gamma));
            E = m*(r-r0) + _infty;
            *gij = -m/r;
            *hij = 0.0;
            //std::cout<<"WARNING: distance between atoms "<<atomi<<" and "<<atomj<<" is "<<r0-r<<"less than their hard core separation"<<std::endl;
        } else if (r < coff) {
            E = 4.*_eps*(- C6 * ir6 + C12 * ir12) + _eps;
            *gij = 4.*_eps*(- 6 * C6 * ir6 + 12 * C12 * ir12) / (dr*r); //1/dr because powers must be 7 and 13, this is -g|gij| (for consistency with the loop in pairwise potential)
            *hij = 4.*_eps*(- 42 * C6 * ir6 + 156 * C12 * ir12) / dr2;
        } else {
            E = 0.;
            *gij = 0.;
            *hij = 0.;
        }

        return E;
    }

};

//
// combine the components (interaction, looping method, distance function) into
// defined classes
//

/**
 * Pairwise HS_WCA_Smooth potential
 */
class HS_WCA_Smooth : public pele::SimplePairwisePotential< HS_WCA_Smooth_interaction > {
public:
    HS_WCA_Smooth(double eps, double sca, pele::Array<double> radii)
        : pele::SimplePairwisePotential< HS_WCA_Smooth_interaction >(
                std::make_shared<HS_WCA_Smooth_interaction>(eps, sca, radii) )
    {}
};


class HS_WCA_Smooth2D : public pele::SimplePairwisePotential< HS_WCA_Smooth_interaction, pele::cartesian_distance<2> > {
public:
    HS_WCA_Smooth2D(double eps, double sca, pele::Array<double> radii)
        : pele::SimplePairwisePotential< HS_WCA_Smooth_interaction, pele::cartesian_distance<2> >(
                std::make_shared<HS_WCA_Smooth_interaction>(eps, sca, radii),
                std::make_shared<pele::cartesian_distance<2>>() )
    {}
};

/**
 * Pairwise HS_WCA_Smooth potential in a rectangular box
 */
class HS_WCA_SmoothPeriodic : public pele::SimplePairwisePotential< HS_WCA_Smooth_interaction, pele::periodic_distance<3> > {
public:
    HS_WCA_SmoothPeriodic(double eps, double sca, pele::Array<double> radii, pele::Array<double> const boxvec)
        : pele::SimplePairwisePotential< HS_WCA_Smooth_interaction, pele::periodic_distance<3>> (
                std::make_shared<HS_WCA_Smooth_interaction>(eps, sca, radii),
                std::make_shared<pele::periodic_distance<3>>(boxvec)
                )
    {}
};

class HS_WCA_SmoothPeriodic2D : public pele::SimplePairwisePotential< HS_WCA_Smooth_interaction, pele::periodic_distance<2> > {
public:
    HS_WCA_SmoothPeriodic2D(double eps, double sca, pele::Array<double> radii, pele::Array<double> const boxvec)
        : pele::SimplePairwisePotential< HS_WCA_Smooth_interaction, pele::periodic_distance<2>> (
                std::make_shared<HS_WCA_Smooth_interaction>(eps, sca, radii),
                std::make_shared<pele::periodic_distance<2>>(boxvec)
                )
    {}
};

/**
 * Frozen particle HS_WCA_Smooth potential
 */
class HS_WCA_SmoothFrozen : public pele::FrozenPotentialWrapper<HS_WCA_Smooth> {
public:
    HS_WCA_SmoothFrozen(double eps, double sca, pele::Array<double> radii, pele::Array<double>& reference_coords, pele::Array<size_t>& frozen_dof)
        : pele::FrozenPotentialWrapper< HS_WCA_Smooth > ( std::make_shared<HS_WCA_Smooth>(eps, sca,
                    radii), reference_coords, frozen_dof)
    {}
};

class HS_WCA_Smooth2DFrozen : public pele::FrozenPotentialWrapper<HS_WCA_Smooth2D> {
public:
    HS_WCA_Smooth2DFrozen(double eps, double sca, pele::Array<double> radii, pele::Array<double>& reference_coords, pele::Array<size_t>& frozen_dof)
        : pele::FrozenPotentialWrapper< HS_WCA_Smooth2D > ( std::make_shared<HS_WCA_Smooth2D>(eps,
                    sca, radii), reference_coords, frozen_dof)
    {}
};

/**
 * Frozen particle HS_WCA_SmoothPeriodic potential
 */
class HS_WCA_SmoothPeriodicFrozen : public pele::FrozenPotentialWrapper<HS_WCA_SmoothPeriodic> {
public:
    HS_WCA_SmoothPeriodicFrozen(double eps, double sca, pele::Array<double> radii,
            pele::Array<double> const boxvec, pele::Array<double>& reference_coords,
            pele::Array<size_t>& frozen_dof)
        : pele::FrozenPotentialWrapper< HS_WCA_SmoothPeriodic > (
                std::make_shared<HS_WCA_SmoothPeriodic>(eps, sca, radii, boxvec),
                reference_coords, frozen_dof)
    {}
};

class HS_WCA_SmoothPeriodic2DFrozen : public pele::FrozenPotentialWrapper<HS_WCA_SmoothPeriodic2D> {
public:
    HS_WCA_SmoothPeriodic2DFrozen(double eps, double sca, pele::Array<double> radii,
            pele::Array<double> const boxvec, pele::Array<double>& reference_coords, pele::Array<size_t>&
            frozen_dof)
        : pele::FrozenPotentialWrapper< HS_WCA_SmoothPeriodic2D > (
                std::make_shared<HS_WCA_SmoothPeriodic2D>(eps, sca, radii, boxvec),
                reference_coords, frozen_dof)
    {}
};
}
#endif
