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
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "normalize_l2.hpp"

using namespace std;
using namespace ngraph;

const string op::NormalizeL2::type_name{"NormalizeL2"};

op::NormalizeL2::NormalizeL2(const shared_ptr<ngraph::Node>& data,
                         const shared_ptr<ngraph::Node>& axes,
                         float eps)
    : FusedOp(check_single_output_args({data, axes}))
    , m_eps{eps}
{
    constructor_validate_and_infer_types();
}

void op::NormalizeL2::pre_validate_and_infer_types()
{
    const auto& data_pshape = get_input_partial_shape(0);
    const auto& axes_pshape = get_input_partial_shape(1);

    if (data_pshape.is_static() && axes_pshape.is_static())
    {
        const Shape data_shape{data_pshape.to_shape()};
        const Shape scale_shape{axes_pshape.to_shape()};

        // Input data must be 2, 3 or 4D tensor.
        NODE_VALIDATION_CHECK(this,
                              (data_shape.size() >= 2 && data_shape.size() <= 4),
                              "Input tensor rank must be 2, 3 or 4 dimensional (actual input "
                              "shape: ",
                              data_shape,
                              ").");
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
        // TODO ADD CHECKING AS IN LRN CASE
    }
}

NodeVector op::NormalizeL2::decompose_op() const
{
    shared_ptr<Node> data{get_argument(0)};
    const Shape input_shape{data->get_shape()};

    // Reshape to 4D tensor.
    if (input_shape.size() != 4)
    {
        Shape data_shape(4 - input_shape.size(), 1);
        copy(begin(input_shape), end(input_shape), back_inserter(data_shape));
        data = builder::reshape(data, data_shape);
    }

    // Calculate norm over axes indicated by axes input param
    AxisSet reduction_axes = get_argument(1)->outputs;

    // Calculate l2 norm across channels.
    shared_ptr<Node> norm = builder::l2_norm(data, reduction_axes, m_eps);
    norm = make_broadcast_node(norm, data->get_shape(), 0);

    data = data / norm;

    // get back original input tensor rank
    if (input_shape.size() != 4)
    {
        data = builder::reshape(data, input_shape);
    }

    return {data};
}

shared_ptr<Node> op::NormalizeL2::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<NormalizeL2>(
        new_args.at(0), new_args.at(1), m_eps);
}
