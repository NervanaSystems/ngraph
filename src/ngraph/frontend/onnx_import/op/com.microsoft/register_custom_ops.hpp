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

#pragma once

#include "dequantize_linear.hpp"
#include "onnx.hpp"
#include "quant_conv.hpp"
#include "quantize_linear.hpp"

namespace onnxruntime
{
    namespace ngraph_ep
    {
        void register_custom_ops()
        {
            constexpr const char* ms_domain = "com.microsoft";

            ngraph::onnx_import::register_operator(
                "DequantizeLinear", 9, ms_domain, dequantize_linear);
            ngraph::onnx_import::register_operator("QuantizeLinear", 9, ms_domain, quantize_linear);
            ngraph::onnx_import::register_operator("QLinearConv", 9, ms_domain, quant_conv);
        }
    } // namespace ngraph_ep

} // namespace onnxruntime
