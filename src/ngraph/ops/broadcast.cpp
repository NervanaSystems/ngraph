// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/sum.hpp"

using namespace std;
using namespace ngraph;

op::Broadcast::Broadcast(const std::shared_ptr<Node>& arg,
                         const Shape& shape,
                         const AxisSet& broadcast_axes)
    : RequiresTensorViewArgs("Broadcast", {arg})
    , m_shape(shape)
    , m_broadcast_axes(broadcast_axes)
{
    auto arg_tensor_view_type = m_inputs.at(0).get_tensor_view_type();
    vector<size_t> target_shape = m_shape;
    for (auto i = m_broadcast_axes.rbegin(); i != m_broadcast_axes.rend(); ++i)
    {
        target_shape.erase(target_shape.begin() + *i);
    }
    if (Shape{target_shape} != arg_tensor_view_type->get_shape())
    {
        throw ngraph_error("Broadcast arg, shape, and axes are incompatible");
    }
    set_value_type_checked(
        make_shared<TensorViewType>(arg_tensor_view_type->get_element_type(), m_shape));
}

void op::Broadcast::generate_adjoints(autodiff::Adjoints& adjoints,
                                      const std::shared_ptr<Node>& delta)
{
    auto x = get_output_node(0);

    adjoints.add_delta(x, make_shared<op::Sum>(delta, m_broadcast_axes));
}
