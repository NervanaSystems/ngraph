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

#include "core/node.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector reverse_sequence(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);

                    const auto sequence_lengths = node.get_ng_inputs().at(1);
                    // nGraph supports only int32 type of sequence_lengths
                    const auto sequence_lengths_i32 = std::make_shared<ngraph::op::Convert>(
                        node.get_ng_inputs().at(1), element::i32);

                    const auto batch_axis = node.get_attribute_value<int64_t>("batch_axis", 1);
                    const auto time_axis = node.get_attribute_value<int64_t>("time_axis", 0);

                    return {std::make_shared<ngraph::op::ReverseSequence>(
                        data, sequence_lengths_i32, batch_axis, time_axis)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
