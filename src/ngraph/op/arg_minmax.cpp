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

#include "ngraph/op/arg_minmax.hpp"
#include "ngraph/op/broadcast.hpp"

using namespace std;
using namespace ngraph;

op::IndexReduction::IndexReduction(const std::string& node_type,
                                   const std::shared_ptr<Node>& arg,
                                   size_t axis,
                                   bool keep_dimensions,
                                   bool is_int64)
    : RequiresTensorViewArgs(node_type, {arg})
    , m_axis(axis)
    , m_is_int64(is_int64)
{
    auto rank = arg->get_shape().size();
    if (rank < 1)
    {
        throw ngraph_error("Tensor's rank should be at least 1");
    }

    if (rank <= axis)
    {
        throw ngraph_error("Axis is greater than rank");
    }

    Shape output_shape = arg->get_shape();
    if (keep_dimensions)
    {
        output_shape.at(axis) = 1;
    }
    else
    {
        output_shape.erase(output_shape.begin() + axis);
    }

    set_value_type_checked(is_int64 ? element::i64 : element::i32, output_shape);
}

void op::IndexReduction::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("Not yet implemented!");
}

shared_ptr<Node> op::ArgMin::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    auto keep_dimensions = new_args.at(0)->get_shape().size() == get_shape().size();
    return make_shared<ArgMin>(new_args.at(0), m_axis, keep_dimensions, m_is_int64);
}
