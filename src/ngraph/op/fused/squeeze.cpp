//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include "ngraph/op/fused/squeeze.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

op::Squeeze::Squeeze(const shared_ptr<Node>& data, const AxisVector& axes)
    : FusedOp("Squeeze", {data})
    , m_axes(axes)
{
    constructor_validate_and_infer_types();
}

NodeVector op::Squeeze::decompose_op() const
{
    auto data = get_argument(0);
    auto data_shape = data->get_shape();

    AxisVector input_order{ngraph::get_default_order(data_shape.size())};

    // Prepare set of unique axes marked to be removed from input data.
    if (m_axes.empty())
    {
        // Default behaviour is to remove all single dimension axes.
        for (std::size_t idx = 0; idx < data_shape.size(); ++idx)
        {
            if (data_shape.at(idx) == 1)
            {
                // Mark with zero elements to remove;
                data_shape.at(idx) = 0;
            }
        }
    }
    else
    {
        std::set<std::size_t, std::greater<std::size_t>> unique_axes(std::begin(m_axes),
                                                                     std::end(m_axes));
        for (uint64_t axis : unique_axes)
        {
            NODE_VALIDATION_CHECK(
                this,
                (data_shape.at(axis) == 1),
                "provided axis value is invalid. Only single dimension axes may be removed.");

            // Mark with zero elements to remove;
            data_shape.at(axis) = 0;
        }
    }

    Shape output_data_shape;
    for (std::size_t idx = 0; idx < data_shape.size(); ++idx)
    {
        if (data_shape.at(idx) != 0)
        {
            output_data_shape.push_back(data_shape.at(idx));
        }
    }
    return {std::make_shared<ngraph::op::Reshape>(data, input_order, output_data_shape)};
}

shared_ptr<Node> op::Squeeze::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Squeeze>(new_args.at(0), m_axes);
}
