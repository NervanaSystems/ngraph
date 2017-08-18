#include "values/descriptors.hpp"

using namespace std;
using namespace ngraph;

void build_simple_graph()
{
    auto tv_a = TensorViewType::make_shared(element_type_float, {2,3, 5});
    auto tp_b = TupleType::make_shared({tv_a, tv_a});
    auto tp_c = TupleType::make_shared({tp_b, tv_a});
    auto tensor_d = TensorViewDescriptor::make_shared(element_type_float, {2, 3, 5});
    auto tuple_d = TupleDescriptor::make_shared({tensor_d, tensor_d});
}
