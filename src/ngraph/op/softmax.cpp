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

void ngraph::op::Softmax::generate_adjoints(autodiff::Adjoints& adjoints,
                                            const std::shared_ptr<Node>& delta)
{
    auto z = delta * shared_from_this();
    auto zsum = std::make_shared<op::Sum>(z, m_axes);

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
    std::iota(order.begin(), order.end(), 0);
    auto zreshape = std::make_shared<op::Reshape>(zsum, order, shape);

    auto adjoint =
        z - builder::make_with_numpy_broadcast<op::Multiply>(shared_from_this(), zreshape);

    auto x = get_input_op(0);
    adjoints.add_delta(x, adjoint);
}
