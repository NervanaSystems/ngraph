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

#include <memory>

#include <onnx-ml.pb.h>

#include "ngraph/op/convert.hpp"
#include "ngraph/type/element_type.hpp"

#include "cast.hpp"
#include "exceptions.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector cast(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    int64_t target_type = node.get_attribute_value<int64_t>("to");
                    element::Type elem_type;

                    switch (target_type)
                    {
                    case onnx::TensorProto_DataType_BOOL: elem_type = element::boolean; break;
                    case onnx::TensorProto_DataType_DOUBLE: elem_type = element::f64; break;
                    case onnx::TensorProto_DataType_FLOAT16:
                    case onnx::TensorProto_DataType_FLOAT: elem_type = element::f32; break;
                    case onnx::TensorProto_DataType_INT8: elem_type = element::i8; break;
                    case onnx::TensorProto_DataType_INT16: elem_type = element::i16; break;
                    case onnx::TensorProto_DataType_INT32: elem_type = element::i32; break;
                    case onnx::TensorProto_DataType_INT64: elem_type = element::i64; break;
                    case onnx::TensorProto_DataType_UINT8: elem_type = element::u8; break;
                    case onnx::TensorProto_DataType_UINT16: elem_type = element::u16; break;
                    case onnx::TensorProto_DataType_UINT32: elem_type = element::u32; break;
                    case onnx::TensorProto_DataType_UINT64: elem_type = element::u64; break;
                    case onnx::TensorProto_DataType_UNDEFINED: elem_type = element::dynamic; break;
                    default: ASSERT_IS_SUPPORTED(node, false) << "unsupported type";
                    }

                    return {std::make_shared<ngraph::op::Convert>(data, elem_type)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
