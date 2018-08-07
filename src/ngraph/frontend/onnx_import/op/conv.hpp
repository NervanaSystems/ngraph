/*******************************************************************************
 * Copyright 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#pragma once

#include <cmath>

#include "ngraph/frontend/onnx_import/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
	namespace error
	{	
		namespace op
		{
			namespace conv
			{

	            struct op_value_error : ngraph_error
	            {
	                explicit op_value_error(const std::string& op_name,
	                                        const std::string& name,
	                                        const std::string& message)
	                    : ngraph_error{op_name + " node (" + name + "): " + message}
	                {
	                }
	            };

			} // namespace  conv

		} // namespace  op

	} // namespace  error
	
	namespace onnx_import
	{
		namespace op
		{
			namespace detail
			{
				inline std::shared_ptr<ngraph::op::Op> 
					make_ng_convolution(const std::shared_ptr<ngraph::Node>& data,
										const std::shared_ptr<ngraph::Node>& weights,
										std::vector<std::size_t> strides,
										std::vector<std::size_t> dilations,
										std::vector<std::size_t> weights,
										std::vector<std::size_t> padding_below,
										std::vector<std::size_t> padding_above,
										int groups)
					{
						if (groups != 1)
						{
							// Split one convolution op to N ops where N is the number of groups 
							// and concat results after computation.
		    				// reference: https://github.com/NervanaSystems/ngraph-mxnet/blob/fdd692/src/ngraph/ngraph_emitter.cc#L822-L856
							const size_t n_data_channels = data->get_shape().at(1);
							const size_t n_weights_channels = weights->get_shape().at(0);
							const size_t data_group_size = n_data_channels / groups;
							const size_t weights_group_size = n_weights_channels / groups;
							NodeVector convolution_nodes;

							// initial bounds for splice
							std::vector<size_t> data_lower_bounds(data->get_shape().size());
							std::vector<size_t> data_upper_bounds{data->get_shape()};
							std::vector<size_t> weights_lower_bounds(weights->get_shape().size());
							std::vector<size_t> weights_upper_bounds{weights->get_shape()};

							for (std::size_t group{0}; group < groups; ++group)
							{
								// slice data 
					            data_lower_bounds[1] = group * data_group_size;
					            data_upper_bounds[1] = (group + 1) * data_group_size;
					            auto sliced_data = 
					            	std::make_shared<ngraph::op::Slice>(data, data_lower_bounds, 
					            					            		data_upper_bounds);
					            // slice weights
					            weights_lower_part[0] = group * weights_group_size;
					            weights_upper_part[0] = std::max(
					            							(group + 1) * weights_group_size, 1);
					            auto sliced_weights = 
				            		std::make_shared<ngraph::op::Slice>(weights, 
				            				weights_lower_part, weights_upper_part);

					            convolution_nodes.push_back(
					            	std::make_shared<ngraph::op::Convolution>(sliced_data, 
					            		sliced_weights, strides, dilations, padding_below, 
					            		padding_above));
							}
							size_t concatenation_axis = 1;
							return std::make_shared<ngraph::op::Concat>(convolution_nodes, 
								concatenation_axis);
						}
						else
						{
							return std::make_shared<ngraph::op::Convolution>(data, weights, 
								strides, dilations, padding_below, padding_above);
						}
					}

			} // namespace  detail

			/**
			 * @brief [brief description]
			 * @details [long description]
			 * 
			 * @param node [description]
			 * @param inputs [description]
			 * 
			 * @return [description]
			 */
			inline NodeVector conv(const Node& node, const NodeVector& inputs)
			{
				auto data = inputs.at(0);
				auto weights = inputs.at(1);

				int groups{node.get_attribute_value<int>("group", 1)};
				// TODO validate groups value
				if (groups <= 0 )
				{
					throw error::op::conv::op_value_error("Conv", node.get_name(), 
						"incorrect value of 'group' attribute: " + std::to_string(groups));
				}
	
				auto strides = get_strides(node);
				auto dilations = get_dilations(node);
				auto paddings = get_pads(node);
				auto padding_below = paddings.at(0);
				auto padding_above = paddings.at(1);

				auto conv_node = detail::make_ng_convolution(data, weights, strides, dilations,
									 padding_below, padding_above, groups);

				if (inputs.size() >= 3)
				{
					auto bias = inputs.at(2);
					const Shape& new_shape = conv_node->get_shape();

					// TODO impl get_broadcast_axes
					// conv_node = std::make_shared<ngraph::op::Add>(conv_node, 
					// 	std::make_shared<ngraph::op::Broadcast>(bias, new_shape, 
					// 		get_broadcast_axes(new_shape, bias->get_shape(), 1))));
				}

				return conv_node;
			}

		} // namespace op

	} // namespace onnx_import

} // namespace ngraph
