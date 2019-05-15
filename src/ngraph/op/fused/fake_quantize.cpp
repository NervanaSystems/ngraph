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

#include <memory>

#include "fake_quantize.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

op::FakeQuantize::FakeQuantize(const shared_ptr<Node>& data,
                               const shared_ptr<Node>& input_low,
                               const shared_ptr<Node>& input_high,
                               const shared_ptr<Node>& output_low,
                               const shared_ptr<Node>& output_high,
                               size_t levels)
    : FusedOp("FakeQuantize", {data})
    , m_levels(levels)
{
    constructor_validate_and_infer_types();
}

void op::FakeQuantize::pre_validate_and_infer_types()
{
    const auto& data_pshape = get_input_partial_shape(0);
    const auto& input_low_pshape = get_input_partial_shape(1);
    const auto& input_high_pshape = get_input_partial_shape(2);
    const auto& output_low_pshape = get_input_partial_shape(3);
    const auto& output_high_pshape = get_input_partial_shape(4);

    if (data_pshape.is_static() && input_low_pshape.is_static() && input_high_pshape.is_static() &&
        output_low_pshape.is_static() && output_high_pshape.is_static())
    {
        const Shape& data_shape{data_pshape.to_shape()};
        const Shape& input_low_shape{input_low_pshape.to_shape()};
        const Shape& input_high_shape{input_high_pshape.to_shape()};
        const Shape& output_low_shape{output_low_pshape.to_shape()};
        const Shape& output_high_shape{output_high_pshape.to_shape()};

        NODE_VALIDATION_CHECK(
            this,
            (input_low_shape.size() == 0 ||
             (input_low_shape.size() == 1 && input_low_shape.at(0) == data_shape.at(1))),
            "Input low tensor shape: ",
            input_low_shape,
            ", must either be a scalar or a vector of size equal to number of channels.");
        NODE_VALIDATION_CHECK(
            this,
            (input_high_shape.size() == 0 ||
             (input_high_shape.size() == 1 && input_high_shape.at(0) == data_shape.at(1))),
            "Input high tensor shape: ",
            input_high_shape,
            ", must either be a scalar or a vector of size equal to number of channels.");
        NODE_VALIDATION_CHECK(
            this,
            (output_low_shape.size() == 0 ||
             (output_low_shape.size() == 1 && output_low_shape.at(0) == data_shape.at(1))),
            "Output low tensor shape: ",
            output_low_shape,
            ", must either be a scalar or a vector of size equal to number of channels.");
        NODE_VALIDATION_CHECK(
            this,
            (output_high_shape.size() == 0 ||
             (output_high_shape.size() == 1 && output_high_shape.at(0) == data_shape.at(1))),
            "Output high tensor shape: ",
            output_high_shape,
            ", must either be a scalar or a vector of size equal to number of channels.");
    }
}

NodeVector op::FakeQuantize::decompose_op() const
{
    shared_ptr<Node> data{get_argument(0)};
    shared_ptr<Node> input_low{get_argument(1)};
    shared_ptr<Node> input_high{get_argument(2)};
    shared_ptr<Node> output_low{get_argument(3)};
    shared_ptr<Node> output_high{get_argument(4)};

    NodeVector broadcasted_nodes =
        numpy_style_broadcast(NodeVector{data, input_low, input_high, output_low, output_high});

    data = broadcasted_nodes.at(0);
    input_low = broadcasted_nodes.at(1);
    input_high = broadcasted_nodes.at(2);
    output_low = broadcasted_nodes.at(3);
    output_high = broadcasted_nodes.at(4);
    shared_ptr<Node> levels_minus_one =
        Constant::create(element::i32,
                         data->get_shape(),
                         vector<size_t>(shape_size(data->get_shape()), m_levels - 1));

    // TODO: arogowiec
    //   Probably should use Quantize -> Dequantize pattern.
    //   May use ngraph::builder::ScaledQuantize and ngraph::builder::ScaledDequantize.
    //   For quantize we probably should change implementation of quantization_util::get_scale
    //   to take levels value in to consideration.

    // if x <= input_low:
    //     output = output_low
    // # round halfway cases away from zero
    // round((x - input_low) / (input_high - input_low) * (levels-1)) /
    //     (levels-1) * (output_high - output_low) + output_low
    // elif x > input_high:
    //     output = output_high
    // else:
    //     # input_low < x <= input_high
    //
    throw ngraph_error("Not yet implemented");
}

shared_ptr<Node> op::FakeQuantize::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 5)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<FakeQuantize>(new_args.at(0), // X
                                     new_args.at(1), // input_low
                                     new_args.at(2), // input_high
                                     new_args.at(3), // output_low
                                     new_args.at(4), // output_high
                                     m_levels);
}
