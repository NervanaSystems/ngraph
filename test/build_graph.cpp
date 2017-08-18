#include "values/descriptors.hpp"
#include "values/function.hpp"

using namespace std;
using namespace ngraph;

void build_simple_graph()
{
    auto tv_a = TensorViewType::make(element_type_float, {2, 3, 5});
    auto tp_b = TupleType::make({tv_a, tv_a});
    auto tp_c = TupleType::make({tp_b, tv_a});
    auto tensor_d = TensorViewDescriptor::make(element_type_float, {2, 3, 5});
    auto tuple_d = TupleDescriptor::make({tensor_d, tensor_d});

    auto cluster_0 = Function::make(
        TensorViewType::make(element_type_float, {32, 3}),
        {TensorViewType::make(element_type_float, {7, 3}),
        TensorViewType::make(element_type_float, {3}),
        TensorViewType::make(element_type_float, {32, 7}),
        TensorViewType::make(element_type_float, {32, 7})
        });
}
