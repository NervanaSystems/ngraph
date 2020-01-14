//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "utils/arg_min_max_factory.hpp"
#include "builder/reshape.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/opsets/opset0.hpp"
#include "ngraph/validation_util.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace arg
        {
            ArgMinMaxFactory::ArgMinMaxFactory(const Node& node)
                : m_keep_dims{node.get_attribute_value<std::int64_t>("keepdims", 1)}
            {
                m_input_node = node.get_ng_inputs().at(0);

                const auto axis = node.get_attribute_value<std::int64_t>("axis", 0);
                m_normalized_axis = ngraph::normalize_axis(
                    node.get_description(), axis, m_input_node->get_shape().size());
            }

            std::shared_ptr<ngraph::Node> ArgMinMaxFactory::make_arg_max() const
            {
                return make_topk_subgraph(default_opset::TopK::Mode::MAX);
            }

            std::shared_ptr<ngraph::Node> ArgMinMaxFactory::make_arg_min() const
            {
                return make_topk_subgraph(default_opset::TopK::Mode::MIN);
            }

            std::shared_ptr<ngraph::Node>
                ArgMinMaxFactory::make_topk_subgraph(default_opset::TopK::Mode mode) const
            {
                const auto k_node =
                    default_opset::Constant::create(ngraph::element::i64, Shape{}, {1});
                const auto topk =
                    std::make_shared<default_opset::TopK>(m_input_node,
                                                          k_node,
                                                          m_normalized_axis,
                                                          mode,
                                                          default_opset::TopK::SortType::NONE);

                const auto indices = std::make_shared<ngraph::opset0::GetOutputElement>(topk, 1);

                if (m_keep_dims == 0)
                {
                    const auto reshaped_indices = reshape::remove_dim(indices, m_normalized_axis);
                    return std::make_shared<ngraph::op::Convert>(reshaped_indices, element::i64);
                }
                return std::make_shared<ngraph::op::Convert>(indices, element::i64);
            }
        }
    }
}
