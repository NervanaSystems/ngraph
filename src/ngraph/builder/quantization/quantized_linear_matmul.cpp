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

#include "ngraph/builder/quantization/quantized_linear_matmul.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/builder/quantization.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/experimental/quantized_dot.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace builder
    {
        namespace quantization
        {
            // TODO: this code is falling back to fp32 dot
            //       1) add support in reference kernel for zero point
            shared_ptr<Node> QuantizedLinearMatmul(const shared_ptr<Node>& input0,
                                                   const shared_ptr<Node>& input1,
                                                   const shared_ptr<Node>& input0_scale,
                                                   const shared_ptr<Node>& input0_zero_point,
                                                   const shared_ptr<Node>& input1_scale,
                                                   const shared_ptr<Node>& input1_zero_point,
                                                   const shared_ptr<Node>& output_scale,
                                                   const shared_ptr<Node>& output_zero_point)
            {
                // Check if zero point is constant and zero
                if (ngraph::is_zero(input0_zero_point) && ngraph::is_zero(input1_zero_point) &&
                    ngraph::is_zero(output_zero_point))
                {
                    auto requantization_scale = (input0_scale * input1_scale) / output_scale;
                    return make_shared<op::QuantizedDot>(input0, input1, requantization_scale);
                }
                else
                {
                    AxisSet axes;

                    auto dq_input0 = make_shared<op::Dequantize>(input0,
                                                                 input0_scale,
                                                                 input0_zero_point,
                                                                 input0_scale->get_element_type(),
                                                                 axes);

                    auto dq_input1 = make_shared<op::Dequantize>(input1,
                                                                 input1_scale,
                                                                 input1_zero_point,
                                                                 input1_scale->get_element_type(),
                                                                 axes);

                    auto dot = make_shared<op::Dot>(dq_input0, dq_input1, 1);
                    return make_shared<op::Quantize>(
                        dot,
                        output_scale,
                        output_zero_point,
                        output_zero_point->get_element_type(),
                        axes,
                        op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN);
                }
            }

            shared_ptr<Node> QuantizedLinearMatmulInteger(const shared_ptr<Node>& input0,
                                                          const shared_ptr<Node>& input1)
            {
                auto output_scale = make_constant(element::f32, Shape{}, 1);
                return make_shared<op::QuantizedDot>(input0, input1, output_scale, false, false);
            }

            shared_ptr<Node>
                QuantizedLinearMatmulInteger(const std::shared_ptr<Node>& input0,
                                             const std::shared_ptr<Node>& input1,
                                             const std::shared_ptr<Node>& input0_zero_point,
                                             const std::shared_ptr<Node>& input1_zero_point)
            {
                // Check if zero points are constant and zero
                if (ngraph::is_zero(input0_zero_point) && ngraph::is_zero(input1_zero_point))
                {
                    return QuantizedLinearMatmulInteger(input0, input1);
                }
                else
                {
                    // Fall back to performing matmul on dequantized floating-point values
                    const auto input0_scale = make_constant(element::f32, Shape{}, 1);
                    const auto input1_scale = make_constant(element::f32, Shape{}, 1);
                    const auto output_scale = make_constant(element::f32, Shape{}, 1);
                    const auto output_zero_point = make_constant(element::i32, Shape{}, 0);
                    const AxisSet axes;

                    const auto dq_input0 =
                        make_shared<op::Dequantize>(input0,
                                                    input0_scale,
                                                    input0_zero_point,
                                                    input0_scale->get_element_type(),
                                                    axes);

                    const auto dq_input1 =
                        make_shared<op::Dequantize>(input1,
                                                    input1_scale,
                                                    input1_zero_point,
                                                    input1_scale->get_element_type(),
                                                    axes);

                    const auto dot = make_shared<op::Dot>(dq_input0, dq_input1, 1);
                    return make_shared<op::Quantize>(
                        dot,
                        output_scale,
                        output_zero_point,
                        output_zero_point->get_element_type(),
                        axes,
                        op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN);
                }
            }
        }
    }
}
