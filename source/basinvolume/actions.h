#ifndef _BV_ACTIONS_H
#define _BV_ACTIONS_H

#include <math.h>
#include <algorithm>
#include <list>
#include <vector>
#include "pele/array.h"
#include "pele/distance.h"
#include "mcpele/mc.h"
#include "mcpele/histogram.h"
#include "mcpele/actions.h"

using std::runtime_error;
using pele::Array;
using mcpele::MC;
using std::sqrt;
using mcpele::Action;

namespace bv{

/*
 * Record displacement square histogram
*/

template<typename distance_policy = pele::cartesian_distance >
class BaseRecordDisp2Histogram : public mcpele::RecordEnergyHistogram {
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

/*
 * Record energy time series, measuring every __record_every-th step.
 */
class RecordEnergyTimeseries : public Action{
	private:
		void _record_energy_value(const double energy);
		const size_t _record_every;
		size_t _counter;
		std::vector<double> _time_series;
	public:
		RecordEnergyTimeseries(const size_t record_every);
		virtual ~RecordEnergyTimeseries(){}
		virtual void action(Array<double> &coords, double energy, bool accepted, MC* mc);
		pele::Array<double> get_time_series();
};

RecordEnergyTimeseries::RecordEnergyTimeseries(const size_t record_every)
	:_record_every(record_every),_counter(0)
	{
		if (record_every==0) throw std::runtime_error("RecordEnergyTimeseries: __record_every expected to be at least 1");
	}

void RecordEnergyTimeseries::action(Array<double> &coords, double energy, bool accepted, MC* mc){
	++_counter;
	if (_counter % _record_every == 0)
		_record_energy_value(energy);
}

void RecordEnergyTimeseries::_record_energy_value(const double energy){
	_time_series.push_back(energy);
}

pele::Array<double> RecordEnergyTimeseries::get_time_series(){
	_time_series.shrink_to_fit();
	return pele::Array<double>(_time_series);
}



}
#endif
