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

#include <algorithm>
#include <memory>
#include <typeindex>
#include <typeinfo>

#include "ngraph/node.hpp"
#include "ngraph/op/trace.hpp"

using namespace std;
using namespace ngraph;

op::Trace::Trace(const std::shared_ptr<Node> arg,
                 Coordinate lower,
                 Coordinate upper,
                 Strides strides)
    : RequiresTensorViewArgs("Trace", {arg})
    , m_lower(lower)
    , m_upper(upper)
    , m_strides(strides)
{
    const auto& arg_shape = arg->get_shape();
    if (m_lower.size() != arg_shape.size() || m_upper.size() != arg_shape.size())
    {
        throw ngraph_error("window rank must be equal to arg shape");
    }

    for (size_t i = 0; i < arg_shape.size(); i++)
    {
        if (m_lower.at(i) > arg_shape.at(i) || m_upper.at(i) > arg_shape.at(i))
        {
            throw ngraph_error("window is out of bounds");
        }

        if (m_lower.at(i) >= m_upper.at(i))
        {
            throw ngraph_error("lower bounds must be less than higher bounds");
        }
    }
    set_value_type_checked(arg->get_element_type(), arg_shape);
}

shared_ptr<Node> op::Trace::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    auto res = make_shared<Trace>(new_args.at(0), m_lower, m_upper, m_strides);
    return res;
}

void op::Trace::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);
    adjoints.add_delta(get_argument(0), delta);
}
