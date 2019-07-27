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

#include "quantization_utils.hpp"

namespace ngraph
{
    namespace builder
    {
        namespace quantization_utils
        {
            std::shared_ptr<Node> max_abs(const Output<Node>& a, const Output<Node>& b)
            {
                auto abs_a = std::make_shared<op::Abs>(a);
                auto abs_b = std::make_shared<op::Abs>(b);
                return std::make_shared<op::Maximum>(abs_a, abs_b);
            }

            std::shared_ptr<Node> get_scale(const Output<Node>& input_min_range,
                                            const Output<Node>& input_max_range,
                                            const ngraph::element::Type& quant_type,
                                            bool bump_by_eps)
            {
                auto type = input_min_range.get_element_type();
                if (type != input_max_range.get_element_type())
                {
                    throw ngraph_error("get_scale: min and max must have same type");
                }

                auto shape = input_min_range.get_shape();
                if (shape != input_max_range.get_shape())
                {
                    throw ngraph_error("get_scale: min and max must have same shape");
                }

                auto min_range = input_min_range;
                auto max_range = input_max_range;

                if (bump_by_eps)
                {
                    auto zero = make_constant(type, shape, 0);
                    min_range = std::make_shared<op::Minimum>(zero, input_min_range);

                    auto max_abs_input_range = max_abs(input_min_range, input_max_range);

                    auto one = make_constant(type, shape, 1);
                    auto hundred = make_constant(type, shape, 100);
                    auto epsilon =
                        std::make_shared<op::Maximum>(one, max_abs_input_range) / hundred;

                    max_range = std::make_shared<op::Maximum>(input_max_range, min_range + epsilon);
                    max_range = std::make_shared<op::Maximum>(zero, max_range);
                }

                size_t bw = quant_type.bitwidth();
                float range = static_cast<float>(
                    (quant_type.is_signed() ? std::pow(2, (bw - 1)) : std::pow(2, bw)) - 1);

                auto max_abs_range = max_abs(min_range, max_range);
                auto target_range = make_constant(type, shape, range);

                return max_abs_range / target_range;
            }
        }
    }
}
