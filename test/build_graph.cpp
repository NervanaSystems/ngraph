#include "values/descriptor.hpp"
#include "values/function.hpp"

using namespace std;
using namespace ngraph;

void build_simple_graph()
{
    auto cluster_0 = Function::make(
        TensorViewType::make(element_type_float, Shape({32, 3})),
        {TensorViewType::make(element_type_float, Shape({7, 3})),
         TensorViewType::make(element_type_float, Shape({3})),
         TensorViewType::make(element_type_float, Shape({32, 7})),
         TensorViewType::make(element_type_float, Shape({32, 7}))
        });
    auto arg3 = cluster_0->parameter(3);
    auto broadcast_1 = Broadcast::make(arg3, {1});
    auto arg2 = cluster_0->parameter(2);
    auto arg0 = cluster_0->parameter(0);
}
