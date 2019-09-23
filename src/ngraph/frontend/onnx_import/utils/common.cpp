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
                // Accepted range of value for axis is [axis_range_min, axis_range_max].
                NGRAPH_CHECK(((axis >= axis_range_min) && (axis <= axis_range_max)),
                             node.get_description(),
                             "Parameter axis ",
                             axis,
                             " out of the tensor rank [-",
                             axis_range_min,
                             ", ",
                             axis_range_max,
                             "].");

                if (axis < 0)
                {
                    axis = axis + tensor_rank;
                }

                return static_cast<size_t>(axis);
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

        } // namespace  common
    }     // namespace onnx_import
} // namespace ngraph
