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

#include <map>
#include <ostream>

#include "mapping.hpp"
#include "ngraph/type/element_type.hpp"
#include "onnx.pb.h"

using namespace ngraph;

element::Type onnx_import::mapping::onnx_to_ng_type(onnx::TensorProto_DataType onnx_type)
{
    switch (onnx_type)
    {
    case onnx::TensorProto_DataType_BOOL: return element::boolean;
    case onnx::TensorProto_DataType_DOUBLE: return element::f64;
    case onnx::TensorProto_DataType_FLOAT16:
        return element::f32; // note 16->32
    case onnx::TensorProto_DataType_FLOAT: return element::f32;
    case onnx::TensorProto_DataType_INT16: return element::i16;
    case onnx::TensorProto_DataType_INT32: return element::i32;
    case onnx::TensorProto_DataType_INT64: return element::i64;
    case onnx::TensorProto_DataType_INT8: return element::i8;
    case onnx::TensorProto_DataType_UINT16: return element::u16;
    case onnx::TensorProto_DataType_UINT32: return element::u32;
    case onnx::TensorProto_DataType_UINT64: return element::u64;
    case onnx::TensorProto_DataType_UINT8: return element::u8;

    case onnx::TensorProto_DataType_STRING: break;
    case onnx::TensorProto_DataType_UNDEFINED: break;
    case onnx::TensorProto_DataType_COMPLEX128: break;
    case onnx::TensorProto_DataType_COMPLEX64: break;
    }

    throw ngraph::ngraph_error("Unrecognized ONNX element type: " +
                               onnx::TensorProto_DataType_Name(onnx_type));
}
