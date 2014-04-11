#ifndef _BV_ACTIONS_H
#define _BV_ACTIONS_H

#include <math.h>
#include <algorithm>
#include <list>
#include "pele/array.h"
#include "pele/mc.h"
#include "pele/histogram.h"
#include "pele/distance.h"
#include "pele/actions.h"

using std::runtime_error;
using pele::Array;
using pele::MC;
using std::sqrt;

namespace bv{

/*
 * Record displacement square histogram
*/

template<typename distance_policy = pele::cartesian_distance >
class BaseRecordDisp2Histogram : public pele::RecordEnergyHistogram {
protected:
	distance_policy *_dist;
	pele::Array<double> _origin, _rattlers, _distance;
	size_t _N;
	BaseRecordDisp2Histogram(pele::Array<double> origin, pele::Array<double> rattlers, double min,
			double max, double bin, size_t eqsteps, distance_policy *dist=NULL):
	RecordEnergyHistogram(min, max, bin, eqsteps),
	_dist(dist),_origin(origin.copy()),_rattlers(rattlers.copy()),
	_distance(origin.size()),_N(origin.size())
	{
		if(_dist == NULL) _dist = new distance_policy;
	}
public:

	virtual ~BaseRecordDisp2Histogram() {
		delete _hist;
		if (_dist != NULL) delete _dist;
	}
	virtual void action(Array<double> &coords, double energy, bool accepted, MC* mc);
};
template<typename distance_policy>
void BaseRecordDisp2Histogram<distance_policy>::action(Array<double> &coords, double energy, bool accepted, MC* mc) {
		double dr[3];
		_count = mc->get_iterations_count();
		if (_count > _eqsteps)
		{
			//compute distances subtracting the origin's coordinates
			for(size_t i=0;i<_N/3;++i){
				size_t i1 = 3*i;
				_dist->get_rij(dr, &coords[i1], &_origin[i1]);
				_distance[i1] = dr[0];
				_distance[i1+1] = dr[1];
				_distance[i1+2] = dr[2];
				}

			//set to 0 distances of rattlers
			for (size_t j = 0; j < _N; ++j){
				_distance[j] *= _rattlers[j];
			}
			//compute square displacement from origin
			double _d = norm(_distance);
			_hist->add_entry(_d*_d);
		}
}

class RecordDisp2Histogram : public BaseRecordDisp2Histogram<>
	{
		public:
		RecordDisp2Histogram(pele::Array<double> origin, pele::Array<double> rattlers, double min,
				double max, double bin, size_t eqsteps)
				: BaseRecordDisp2Histogram(origin, rattlers, min,
						max, bin, eqsteps){}
	};

class RecordDisp2HistogramPeriodic : public BaseRecordDisp2Histogram<pele::periodic_distance>
	{
		public:
		RecordDisp2HistogramPeriodic(pele::Array<double> origin, pele::Array<double> rattlers, double min,
				double max, double bin, size_t eqsteps, double const *boxvec)
				: BaseRecordDisp2Histogram<pele::periodic_distance>(
						origin, rattlers, min, max, bin, eqsteps,
						new pele::periodic_distance(boxvec[0], boxvec[1], boxvec[2])){}
	};






}
#endif
