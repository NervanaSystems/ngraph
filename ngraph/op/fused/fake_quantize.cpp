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
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/subtract.hpp"
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
    : FusedOp("FakeQuantize", {data, input_low, input_high, output_low, output_high})
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
        const Shape data_shape{data_pshape.to_shape()};
        const Shape input_low_shape{input_low_pshape.to_shape()};
        const Shape input_high_shape{input_high_pshape.to_shape()};
        const Shape output_low_shape{output_low_pshape.to_shape()};
        const Shape output_high_shape{output_high_pshape.to_shape()};

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

    if (input_low->get_shape().size() == 0)
    {
        NodeVector broadcasted_nodes =
            numpy_style_broadcast(NodeVector{data, input_low, input_high, output_low, output_high});

        data = broadcasted_nodes.at(0);
        input_low = broadcasted_nodes.at(1);
        input_high = broadcasted_nodes.at(2);
        output_low = broadcasted_nodes.at(3);
        output_high = broadcasted_nodes.at(4);
    }
    else
    {
        input_low = legacy_style_broadcast_for_binary_operation(data, input_low, 1).at(1);
        input_high = legacy_style_broadcast_for_binary_operation(data, input_high, 1).at(1);
        output_low = legacy_style_broadcast_for_binary_operation(data, output_low, 1).at(1);
        output_high = legacy_style_broadcast_for_binary_operation(data, output_high, 1).at(1);
    }

    const auto input_data_shape = data->get_shape();
    const auto input_data_type = data->get_element_type();

    const auto levels_minus_one =
        Constant::create(input_data_type,
                         input_data_shape,
                         vector<size_t>(shape_size(input_data_shape), m_levels - 1));

    // map the number of quantization levels to the nGraph's quantization and dequantization scales
    const auto quant_scale = (input_high - input_low) / levels_minus_one;
    const auto dequant_scale = (output_high - output_low) / levels_minus_one;

    // zero_point type needs to match the quantization output type
    const auto zero_point = Constant::create(element::i32, data->get_shape(), {0.0});
    const auto axes = get_default_order(input_data_shape);

    // clip the input data to the range <input_low;input_high>
    data =
        std::make_shared<op::Minimum>(input_high, std::make_shared<op::Maximum>(input_low, data));

    // shift the input data so that it contains only positive values (and zeros)
    data = data - input_low;

    shared_ptr<Node> quantized_data =
        make_shared<op::Quantize>(data,
                                  quant_scale,
                                  zero_point,
                                  element::i32,
                                  axes,
                                  op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY);

    quantized_data = make_shared<op::Convert>(quantized_data, input_data_type);

    // dequantization without using the Dequantize op (just a multiplication by the dequant_scale)
    const auto dequantized_data = quantized_data * dequant_scale;

    // shift the results so that they fall into the <output_low;output_high> range
    return {dequantized_data + output_low};
}

shared_ptr<Node> op::FakeQuantize::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<FakeQuantize>(new_args.at(0), // X
                                     new_args.at(1), // input_low
                                     new_args.at(2), // input_high
                                     new_args.at(3), // output_low
                                     new_args.at(4), // output_high
                                     m_levels);
}
