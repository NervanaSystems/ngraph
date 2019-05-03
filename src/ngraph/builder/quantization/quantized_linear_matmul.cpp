//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
            shared_ptr<Node> QuantizedLinearMatmul(shared_ptr<Node> input,
                                                   shared_ptr<Node> filter,
                                                   shared_ptr<Node> input_scale,
                                                   shared_ptr<Node> filter_scale,
                                                   shared_ptr<Node> output_scale)
            {
                auto requantization_scale = (input_scale * filter_scale) / output_scale;

                // Since u8u8 goes to ref reshape the filter so that the ref dot can do a matmul
                // which requires [m, n] * [n, k] order where as mkldnn requires [m, n] * [k, n]
                if (input->get_element_type() == element::u8 &&
                    filter->get_element_type() == element::u8)
                {
                    filter = make_shared<op::Reshape>(
                        filter,
                        AxisVector{1, 0},
                        Shape{filter->get_shape()[1], filter->get_shape()[0]});
                }

                return make_shared<op::QuantizedDot>(input, filter, requantization_scale);
            }

            // TODO: this code is falling back to fp32 dot
            //       1) add support in reference kernel for zero point
            shared_ptr<Node> QuantizedLinearMatmul(shared_ptr<Node> input,
                                                   shared_ptr<Node> filter,
                                                   shared_ptr<Node> input_scale,
                                                   shared_ptr<Node> input_zero_point,
                                                   shared_ptr<Node> filter_scale,
                                                   shared_ptr<Node> filter_zero_point,
                                                   shared_ptr<Node> output_scale,
                                                   shared_ptr<Node> output_zero_point)
            {
                AxisSet axes;

                auto dq_input = make_shared<op::Dequantize>(
                    input, input_scale, input_zero_point, input_scale->get_element_type(), axes);

                auto dq_filter = make_shared<op::Dequantize>(filter,
                                                             filter_scale,
                                                             filter_zero_point,
                                                             filter_scale->get_element_type(),
                                                             axes);

                auto dot = make_shared<op::Dot>(dq_input, dq_filter, 1);
                auto q_dot =
                    make_shared<op::Quantize>(dot,
                                              output_scale,
                                              output_zero_point,
                                              output_zero_point->get_element_type(),
                                              axes,
                                              op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN);
                return q_dot;
            }

            shared_ptr<Node> QuantizedLinearMatmulInteger(shared_ptr<Node> input,
                                                          shared_ptr<Node> filter)
            {
                auto output_scale = make_constant(element::f32, Shape{}, 1);
                return make_shared<op::QuantizedDot>(input, filter, output_scale, false, false);
            }
        }
    }
}
