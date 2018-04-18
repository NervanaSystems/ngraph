/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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

#include <map>
#include <ostream>
#include "ngraph/type/element_type.hpp"
#include "onnx.pb.h"

namespace ngraph
{
    namespace onnx_import
    {
        struct mapping
        {
            static const std::map<onnx::TensorProto_DataType, ngraph::element::Type>
                onnx_to_ng_types;
        };

        const std::map<onnx::TensorProto_DataType, ngraph::element::Type> mapping::onnx_to_ng_types{
            {onnx::TensorProto_DataType_BOOL, ngraph::element::boolean},
            {onnx::TensorProto_DataType_INT8, ngraph::element::i8},
            {onnx::TensorProto_DataType_INT16, ngraph::element::i16},
            {onnx::TensorProto_DataType_INT32, ngraph::element::i32},
            {onnx::TensorProto_DataType_INT64, ngraph::element::i64},
            {onnx::TensorProto_DataType_UINT8, ngraph::element::u8},
            {onnx::TensorProto_DataType_UINT16, ngraph::element::u16},
            {onnx::TensorProto_DataType_UINT32, ngraph::element::u32},
            {onnx::TensorProto_DataType_UINT64, ngraph::element::u64},
            {onnx::TensorProto_DataType_FLOAT16, ngraph::element::f32}, // note 16->32
            {onnx::TensorProto_DataType_FLOAT, ngraph::element::f32},
            {onnx::TensorProto_DataType_DOUBLE, ngraph::element::f64},
            // nGraph incompatible types:
            // onnx::TensorProto_DataType_UNDEFINED
            // onnx::TensorProto_DataType_STRING
            // onnx::TensorProto_DataType_COMPLEX64
            // onnx::TensorProto_DataType_COMPLEX128
        };

    } // namespace onnx_import
} // namespace ngraph
