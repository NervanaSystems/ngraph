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
#include <algorithm>
#include <iterator>

#include "ngraph/builder/norm.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/op/util/reshape.hpp"
#include "normalize.hpp"

using namespace std;
using namespace ngraph;

op::Normalize::Normalize(const shared_ptr<ngraph::Node>& data,
                         const shared_ptr<ngraph::Node>& scale,
                         bool across_spatial,
                         bool channel_shared,
                         float eps)
    : FusedOp("Normalize", {data, scale})
    , m_across_spatial{across_spatial}
    , m_channel_shared{channel_shared}
    , m_eps{eps}
{
    constructor_validate_and_infer_types();
}

void op::Normalize::pre_validate_and_infer_types()
{
    const auto& data_pshape = get_input_partial_shape(0);
    const auto& scale_pshape = get_input_partial_shape(1);

    if (data_pshape.is_static() && scale_pshape.is_static())
    {
        const auto& data_shape{data_pshape.to_shape()};
        const auto& scale_shape{scale_pshape.to_shape()};

        // Input data must be 2, 3 or 4D tensor.
        NODE_VALIDATION_CHECK(this,
                              (data_shape.size() >= 2 && data_shape.size() <= 4),
                              "Input tensor rank must be 2, 3 or 4 dimensional (actual input "
                              "shape: ",
                              data_shape,
                              ").");
        if (m_channel_shared)
        {
            NODE_VALIDATION_CHECK(this,
                                  scale_shape.size() == 0,
                                  "Scale must be a scalar if 'channels_shared' parameter is true");
        }
        else
        {
            // only HW
            if (data_shape.size() == 2)
            {
                NODE_VALIDATION_CHECK(this,
                                      scale_shape.size() == 0,
                                      "Scale must be a scalar if input tensor is of rank 2.");
            }
            else
            {
                size_t n_channels = data_shape.size() == 3 ? data_shape.at(0) : data_shape.at(1);
                NODE_VALIDATION_CHECK(
                    this,
                    (scale_shape.size() == 1 && scale_shape.at(0) == n_channels),
                    "Scale must be a vector of size of input tensor channels if input tensor is "
                    "of rank greater equal 3.");
            }
        }
    }
}

NodeVector op::Normalize::decompose_op() const
{
    const auto input_node{get_argument(0)};
    const auto& input_shape{input_node->get_shape()};

    auto data{input_node};
    // Reshape to 4D tensor.
    if (input_shape.size() != 4)
    {
        Shape data_shape(4 - input_shape.size(), 1);
        copy(begin(input_shape), end(input_shape), back_inserter(data_shape));
        data = util::reshape(data, data_shape);
    }

    // Calculate norm over CHW axes.
    AxisSet reduction_axes{1, 2, 3};
    if (m_across_spatial)
    {
        // Calculate norm only onver HW axes.
        reduction_axes = AxisSet{2, 3};
    }

    // Calculate l2 norm across channels.
    shared_ptr<Node> norm = builder::l2_norm(data, reduction_axes, m_eps);
    norm = make_broadcast_node(norm, data->get_shape(), 0);

    auto scale_node{get_argument(1)};

    // Broadcast scale to data tensor shape.
    if (m_channel_shared)
    {
        // Scale is a scalar.
        scale_node = make_broadcast_node(scale_node, data->get_shape());
    }
    else
    {
        // Scale is a vector of size equal to C axis.
        scale_node = make_broadcast_node(scale_node, data->get_shape(), 1);
    }

    data = data / norm * scale_node;

    // get back original input tensor rank
    if (input_shape.size() != 4)
    {
        data = util::reshape(data, input_shape);
    }

    return {data};
}

shared_ptr<Node> op::Normalize::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Normalize>(
        new_args.at(0), new_args.at(1), m_across_spatial, m_channel_shared, m_eps);
}
