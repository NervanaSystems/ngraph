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
#include <onnx/onnx_pb.h> // onnx types

#include "common.hpp"
#include "default_opset.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/opsets/opset0.hpp"
#include "validation_util.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace common
        {
            const ngraph::element::Type& get_ngraph_element_type(int64_t onnx_type)
            {
                switch (onnx_type)
                {
                case onnx::TensorProto_DataType_BOOL: return element::boolean;
                case onnx::TensorProto_DataType_DOUBLE: return element::f64;
                case onnx::TensorProto_DataType_FLOAT16: return element::f16;
                case onnx::TensorProto_DataType_FLOAT: return element::f32;
                case onnx::TensorProto_DataType_INT8: return element::i8;
                case onnx::TensorProto_DataType_INT16: return element::i16;
                case onnx::TensorProto_DataType_INT32: return element::i32;
                case onnx::TensorProto_DataType_INT64: return element::i64;
                case onnx::TensorProto_DataType_UINT8: return element::u8;
                case onnx::TensorProto_DataType_UINT16: return element::u16;
                case onnx::TensorProto_DataType_UINT32: return element::u32;
                case onnx::TensorProto_DataType_UINT64: return element::u64;
                case onnx::TensorProto_DataType_UNDEFINED: return element::dynamic;
                }
                throw ngraph_error("unsupported element type: " +
                                   onnx::TensorProto_DataType_Name(
                                       static_cast<onnx::TensorProto_DataType>(onnx_type)));
            }

            std::size_t validate_axis(const ngraph::onnx_import::Node& node,
                                      std::int64_t axis,
                                      std::int64_t tensor_rank)
            {
                // Accepted range of value for axis is [-tensor_rank, tensor_rank-1].
                return validate_axis(node, axis, tensor_rank, -tensor_rank, tensor_rank - 1);
            }

            std::size_t validate_axis(const ngraph::onnx_import::Node& node,
                                      std::int64_t axis,
                                      std::int64_t tensor_rank,
                                      std::int64_t axis_range_min,
                                      std::int64_t axis_range_max)
            {
                return ngraph::normalize_axis(
                    node.get_description(), axis, tensor_rank, axis_range_min, axis_range_max);
            }

            std::vector<std::size_t> validate_axes(const ngraph::onnx_import::Node& node,
                                                   std::vector<std::int64_t> axes,
                                                   std::int64_t tensor_rank)
            {
                std::vector<std::size_t> new_axes;

                for (auto a : axes)
                {
                    new_axes.push_back(validate_axis(node, a, tensor_rank));
                }

                return new_axes;
            }

            ngraph::NodeVector get_outputs(const std::shared_ptr<ngraph::Node>& node)
            {
                const auto outputs_number = node->get_output_size();
                ngraph::NodeVector outputs(outputs_number);
                for (int i = 0; i < outputs_number; ++i)
                {
                    if (node->output(i).get_node_shared_ptr()->get_output_size() == 1)
                    {
                        outputs[i] = node->get_output_as_single_output_node(i);
                    }
                    else
                    {
                        outputs[i] = std::make_shared<ngraph::opset0::GetOutputElement>(node, i);
                    }
                }
                return outputs;
            }

        } // namespace  common
    }     // namespace onnx_import
} // namespace ngraph
