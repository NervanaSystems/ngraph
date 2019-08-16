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
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/fused/normalize_l2.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

const string op::NormalizeL2::type_name{"NormalizeL2"};

op::NormalizeL2::NormalizeL2(const Output<Node>& data,
                             const Output<Node>& axes,
                             float eps,
                             EpsMode eps_mode)
    : FusedOp({data, axes})
    , m_eps{eps}
    , m_eps_mode{eps_mode}
{
    constructor_validate_and_infer_types();
}

void op::NormalizeL2::pre_validate_and_infer_types()
{
    const auto& data_pshape = get_input_partial_shape(0);
    const auto& axes_pshape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this, data_pshape.is_static(), "Input data must be static.");
    NODE_VALIDATION_CHECK(this, axes_pshape.is_static(), "Input axes must be static.");

    const Shape data_shape{data_pshape.to_shape()};

    // Input data must be 2, 3 or 4D tensor.
    NODE_VALIDATION_CHECK(this,
                          (data_shape.size() >= 2 && data_shape.size() <= 4),
                          "Input tensor rank must be 2, 3 or 4 dimensional (actual input "
                          "shape: ",
                          data_shape,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          static_cast<size_t>(axes_pshape.rank()) == 1,
                          "Input axes must have rank equals 1 (axes shape: ",
                          axes_pshape,
                          ").");
}

NodeVector op::NormalizeL2::decompose_op() const
{
    Output<Node> data{input_value(0)};
    const Shape input_shape{data.get_shape()};

    // Reshape to 4D tensor.
    if (input_shape.size() != 4)
    {
        Shape data_shape(4 - input_shape.size(), 1);
        copy(begin(input_shape), end(input_shape), back_inserter(data_shape));
        data = builder::reshape(data, data_shape);
    }

    auto axes_node = input(1).get_source_output().get_node_shared_ptr();
    NODE_VALIDATION_CHECK(this,
                          axes_node->is_constant(),
                          "doesn't support 'axes' input of other type than a Constant.");

    // Calculate norm over axes indicated by axes input param
    auto axes_constant = dynamic_pointer_cast<op::Constant>(axes_node);
    auto axes_vector = axes_constant->get_vector<size_t>();
    AxisSet reduction_axes{axes_vector};

    // Calculate l2 norm across axes determined by axes input
    auto builder_bias_mode =
        (m_eps_mode == EpsMode::MAX) ? builder::BiasMode::MAX : builder::BiasMode::ADD;
    Output<Node> norm = builder::l2_norm(data, reduction_axes, m_eps, builder_bias_mode);
    norm = make_broadcast_node(norm, data.get_shape(), 0);

    data = data / norm;

    // get back original input tensor rank
    if (input_shape.size() != 4)
    {
        data = builder::reshape(data, input_shape);
    }

    return as_node_vector({data});
}

shared_ptr<Node> op::NormalizeL2::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<NormalizeL2>(new_args.at(0), new_args.at(1), m_eps, m_eps_mode);
}
