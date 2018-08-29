//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/sum.hpp"

using namespace std;
using namespace ngraph;

op::Broadcast::Broadcast(const shared_ptr<Node>& arg,
                         const Shape& shape,
                         const AxisSet& broadcast_axes)
    : RequiresTensorViewArgs("Broadcast", {arg})
    , m_shape(shape)
    , m_broadcast_axes(broadcast_axes)
{
    auto& input = m_inputs.at(0);
    Shape target_shape = m_shape;
    for (auto i = m_broadcast_axes.rbegin(); i != m_broadcast_axes.rend(); ++i)
    {
        NODE_VALIDATION_ASSERT(this, *i < target_shape.size())
            << "Broadcast axis index (" << *i << ") exceeds target shape rank "
            << "(broadcast axes: " << m_broadcast_axes << ", target shape: " << target_shape
            << ").";

        target_shape.erase(target_shape.begin() + *i);
    }

    // TODO(amprocte): We can probably have a more helpful error message here.
    // There are two things that can go wrong, which are being picked up in
    // one fell swoop by this check: either the number of broadcast axes is not
    // enough (arg->get_shape().size() + broadcast_axes.size() != shape.size())
    // or there is a mismatch with one of the pre-broadcast axis lengths
    // (i.e. target_shape.size() == arg->get_shape.size() but there is some i
    // where target_shape[i] != arg->get_shape[i]).
    NODE_VALIDATION_ASSERT(this, target_shape == input.get_shape())
        << "Broadcast argument shape, target shape, and axes are incompatible "
        << "(argument shape: " << arg->get_shape() << ", target shape: " << m_shape
        << ", broadcast axes: " << m_broadcast_axes << ").";

    set_value_type_checked(make_shared<TensorViewType>(input.get_element_type(), m_shape));
}

shared_ptr<Node> op::Broadcast::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args, 1);
    return make_shared<Broadcast>(new_args.at(0), m_shape, m_broadcast_axes);
}

void op::Broadcast::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_argument(0);

    adjoints.add_delta(x, make_shared<op::Sum>(delta, m_broadcast_axes));
}
