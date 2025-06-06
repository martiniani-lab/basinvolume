#include "basin_descriptors.h"
#include "mcpele/mc.h"
#include "mcpele/histogram.h"
#include <memory>



namespace bv {



class InsideBasinTest : public mcpele::ConfTest {
    protected:
        std::unique_ptr<AbstractGradientBasin> _basin; // description of what the basin is
        bool _collect_attractors; 
        // statistics
        mcpele::Moments _failed_optimizations;

    public:
        InsideBasinTest(std::unique_ptr<AbstractGradientBasin> basin);
        virtual ~InsideBasinTest() = default;
    bool conf_test(pele::Array<double> &trial_coords, mcpele::MC *mc) override;
};


}