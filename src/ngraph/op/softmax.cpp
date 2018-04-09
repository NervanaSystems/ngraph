/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/op/softmax.hpp"

#include <algorithm>
#include <numeric>

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"

using namespace std;
using namespace ngraph;

op::Softmax::Softmax(const shared_ptr<Node>& arg, const AxisSet& axes)
    : UnaryElementwiseArithmetic("Softmax", arg)
    , m_axes(axes)
{
    for (auto axis : m_axes)
    {
        if (axis >= get_shape().size())
        {
            throw ngraph_error("Axis for softmax reduction operator is out of bounds");
        }
    }

    // empty axes == all axes
    if (m_axes.size() == 0)
    {
        for (size_t i = 0; i < get_shape().size(); ++i)
        {
            m_axes.insert(i);
        }
    }
}

shared_ptr<Node> op::Softmax::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Softmax>(new_args.at(0), m_axes);
}

void op::Softmax::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto z = delta * shared_from_this();
    auto zsum = make_shared<op::Sum>(z, m_axes);

    Shape shape;
    for (size_t i = 0; i < get_shape().size(); ++i)
    {
        if (m_axes.find(i) == m_axes.end())
        {
            shape.push_back(get_shape()[i]);
        }
        else
        {
            shape.push_back(1);
        }
    }
    AxisVector order(zsum->get_shape().size());
    iota(order.begin(), order.end(), 0);
    auto zreshape = make_shared<op::Reshape>(zsum, order, shape);

    auto adjoint =
        z - builder::make_with_numpy_broadcast<op::Multiply>(shared_from_this(), zreshape);

    auto x = get_input_op(0);
    adjoints.add_delta(x, adjoint);
}
