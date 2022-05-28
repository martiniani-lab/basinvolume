#ifndef _BV_RECORD_DISPLACEMENT_TIMESERIES_H
#define _BV_RECORD_DISPLACEMENT_TIMESERIES_H

#include <vector>

#include "pele/array.h"
#include "pele/distance.h"

#include "mcpele/mc.h"
#include "mcpele/record_scalar_timeseries.h"
#include "mcpele/cloud_test.h"

namespace bv {

/*
 * Record displacement time series, measuring every __record_every-th step.
 */

class RecordDisplacementTimeseries : public mcpele::RecordScalarTimeseries {
    private:
        void m_get_vec_distance(const pele::Array<double>& x);
        pele::Array<double> m_origin;
        pele::Array<double> m_distance;
        const size_t m_ndim;
        const size_t m_nparticles;
        const bool m_fix_com;
    public:
        RecordDisplacementTimeseries(pele::Array<double> origin, const size_t ndim, const size_t niter, const size_t record_every, const bool fix_com=true);
        virtual ~RecordDisplacementTimeseries(){}
        virtual double get_recorded_scalar(pele::Array<double> &coords, const double energy, const bool accepted, mcpele::MC* mc);
};

class RecordCloudDisplacementTimeseries : public mcpele::RecordCloudScalarTimeseries {
        pele::Array<double> m_origin;
        std::mt19937_64 m_generator;
        std::uniform_real_distribution<double> m_uniform_real_distribution;
    public:
        RecordCloudDisplacementTimeseries(pele::Array<double> origin, const size_t niter, const size_t record_every,
        const size_t rseed)
        : RecordCloudScalarTimeseries(niter, record_every),
          m_origin(origin.copy()),
          m_generator(rseed),
          m_uniform_real_distribution(0.0, 1.0){}
        virtual ~RecordCloudDisplacementTimeseries(){}
        double do_action(const std::shared_ptr<mcpele::Drop> drop, mcpele::MC* mc){
            return get_drop_dr(drop->x);
        }
        double get_drop_dr(const pele::Array<double>& drop_x) const
        {
            double r2 = 0;
            for (size_t i = 0; i < drop_x.size(); ++i) {
                const double tmp = drop_x[i] - m_origin[i];
                r2 += tmp * tmp;
            }
            return std::sqrt(r2);
        }

//      this is different from the usual cloud_action record_cloud_action
//      because we record every single distance for the drop
        void record_cloud_action(const mcpele::Cloud& c, mcpele::MC* mc)
        {
            const size_t counter = mc->get_iterations_count();
            if (counter % m_record_every == 0) {
                double sum_of_weights = 0;
                for (const std::shared_ptr<mcpele::Drop> drop : c) {
                    const double tmp = drop->bias * drop->oracle;
                    sum_of_weights += tmp;
                }
                for (const std::shared_ptr<mcpele::Drop> drop : c) {
                    if (drop->oracle){
                        const double rand = m_uniform_real_distribution(m_generator);
                        const double drop_weight = drop->bias/sum_of_weights;
                        if (rand < drop_weight){
                            const double r = do_action(drop, mc);
                            m_record_scalar_value(r);
                        }
                    }
                }
            }
        }
};

} // namespace bv

#endif // #ifndef _BV_RECORD_DISPLACEMENT_TIMESERIES_H
