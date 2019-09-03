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

            std::size_t convert_negative_axis(std::int64_t axis, std::size_t tensor_rank)
            {
                if (axis >= 0)
                {
                    return std::min(static_cast<size_t>(axis), tensor_rank);
                }
                else
                {
                    std::int64_t new_axis = axis + static_cast<std::int64_t>(tensor_rank);
                    if (new_axis < 0)
                    {
                        throw ngraph_error("Parameter axis out of the tensor rank.");
                    }
                    else
                    {
                        return static_cast<size_t>(new_axis);
                    }
                }
            }

            std::vector<std::size_t> convert_negative_axis(std::vector<std::int64_t> axes,
                                                           std::size_t tensor_rank)
            {
                std::vector<std::size_t> new_axes;

                for (auto a : axes)
                {
                    new_axes.push_back(convert_negative_axis(a, tensor_rank));
                }

                return new_axes;
            }

        } // namespace  common
    }     // namespace onnx_import
} // namespace ngraph
