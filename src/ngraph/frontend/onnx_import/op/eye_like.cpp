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

#include <onnx-ml.pb.h>

#include "exceptions.hpp"
#include "eye_like.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/frontend/onnx_import/utils/common.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/embedding_lookup.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector eye_like(const Node& node)
                {
                    const auto input = node.get_ng_inputs().at(0);
                    const auto& input_shape = input->get_shape();

                    std::int64_t dtype;
                    element::Type target_type;

                    std::int64_t shift = node.get_attribute_value<std::int64_t>("k", 0);
                    try
                    {
                        dtype = node.get_attribute_value<std::int64_t>("dtype");
                        switch (dtype)
                        {
                        case onnx::TensorProto_DataType_BOOL: target_type = element::boolean; break;
                        case onnx::TensorProto_DataType_DOUBLE: target_type = element::f64; break;
                        case onnx::TensorProto_DataType_FLOAT16:
                        case onnx::TensorProto_DataType_FLOAT: target_type = element::f32; break;
                        case onnx::TensorProto_DataType_INT8: target_type = element::i8; break;
                        case onnx::TensorProto_DataType_INT16: target_type = element::i16; break;
                        case onnx::TensorProto_DataType_INT32: target_type = element::i32; break;
                        case onnx::TensorProto_DataType_INT64: target_type = element::i64; break;
                        case onnx::TensorProto_DataType_UINT8: target_type = element::u8; break;
                        case onnx::TensorProto_DataType_UINT16: target_type = element::u16; break;
                        case onnx::TensorProto_DataType_UINT32: target_type = element::u32; break;
                        case onnx::TensorProto_DataType_UINT64: target_type = element::u64; break;
                        case onnx::TensorProto_DataType_UNDEFINED:
                            target_type = element::dynamic;
                            break;
                        default: ASSERT_IS_SUPPORTED(node, false) << "unsupported type";
                        }
                    }
                    catch (...)
                    {
                        target_type = input->get_element_type();
                    }

                    ASSERT_VALID_ARGUMENT(node, input_shape.size() == 2)
                        << "The provided shape rank: " << input_shape.size()
                        << " is unsupported, only 2D shapes are supported";

                    std::shared_ptr<ngraph::Node> eye_like_matrix =
                        common::shifted_square_identity(input_shape, target_type, shift);

                    return {eye_like_matrix};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
