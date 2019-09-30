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
#include "utils/common.hpp"

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
                    std::size_t valid_batch_axis =
                        common::validate_axis(node, batch_axis, data->get_shape().size());
                    const auto time_axis = node.get_attribute_value<int64_t>("time_axis", 0);
                    std::size_t valid_time_axis =
                        common::validate_axis(node, time_axis, data->get_shape().size());

                    NGRAPH_CHECK(valid_batch_axis == 0 || valid_batch_axis == 1,
                                 "Allowed values of the 'batch_axis' attribute for ReverseSequence "
                                 "operator are 0 and 1");

                    NGRAPH_CHECK(valid_time_axis == 0 || valid_time_axis == 1,
                                 "Allowed values of the 'time_axis' attribute for ReverseSequence "
                                 "operator are 0 and 1");

                    NGRAPH_CHECK(valid_batch_axis != valid_time_axis,
                                 "'batch_axis' and 'time_axis' attributes of the ReverseSequence "
                                 "operator can't point to the same dimension");

                    return {std::make_shared<ngraph::op::ReverseSequence>(
                        data, sequence_lengths_i32, valid_batch_axis, valid_time_axis)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
