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

#include <cstring>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/reference/dequantize.hpp"
#include "ngraph/runtime/reference/quantize.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Dequantize)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& functors = external_function->get_functors();
                    auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    size_t dequantize_index =
                        mkldnn_emitter->build_dequantization(node, input_desc, result_desc);
                    auto& deps = mkldnn_emitter->get_primitive_deps(dequantize_index);
                    auto functor = [&, dequantize_index](CPURuntimeContext* ctx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, dequantize_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    auto& functors = external_function->get_functors();
                    auto& tensor_data = external_function->get_tensor_data();

                    const ngraph::op::Dequantize* dequantize =
                        static_cast<const ngraph::op::Dequantize*>(node);
                    function<void(CPURuntimeContext*)> functor;

                    auto& arg0_tensor = tensor_data[args[0].get_name()];
                    auto& arg1_tensor = tensor_data[args[1].get_name()];
                    auto& arg2_tensor = tensor_data[args[2].get_name()];
                    auto& out_tensor = tensor_data[out[0].get_name()];
                    auto arg0_shape = args[0].get_shape();
                    auto arg1_shape = args[1].get_shape();
                    auto daxes = dequantize->get_axes();

                    if (args[0].get_element_type() == element::i8)
                    {
                        if (out[0].get_element_type() == element::f32)
                        {
                            functor = [&, arg0_shape, arg1_shape, daxes](CPURuntimeContext* ctx) {
                                ngraph::runtime::reference::dequantize<int8_t>(
                                    static_cast<int8_t*>(arg0_tensor),
                                    static_cast<float*>(arg1_tensor),
                                    static_cast<int8_t*>(arg2_tensor),
                                    static_cast<float*>(out_tensor),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes);
                            };
                        }
                        else if (out[0].get_element_type() == element::f64)
                        {
                            functor = [&, arg0_shape, arg1_shape, daxes](CPURuntimeContext* ctx) {
                                ngraph::runtime::reference::dequantize<int8_t>(
                                    static_cast<int8_t*>(arg0_tensor),
                                    static_cast<double*>(arg1_tensor),
                                    static_cast<int8_t*>(arg2_tensor),
                                    static_cast<double*>(out_tensor),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes);
                            };
                        }
                        else
                        {
                            throw ngraph_error("Unsupported dequantization element type");
                        }
                    }
                    else if (args[0].get_element_type() == element::u8)
                    {
                        if (out[0].get_element_type() == element::f32)
                        {
                            functor = [&, arg0_shape, arg1_shape, daxes](CPURuntimeContext* ctx) {
                                ngraph::runtime::reference::dequantize<uint8_t>(
                                    static_cast<uint8_t*>(arg0_tensor),
                                    static_cast<float*>(arg1_tensor),
                                    static_cast<uint8_t*>(arg2_tensor),
                                    static_cast<float*>(out_tensor),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes);
                            };
                        }
                        else if (out[0].get_element_type() == element::f64)
                        {
                            functor = [&, arg0_shape, arg1_shape, daxes](CPURuntimeContext* ctx) {
                                ngraph::runtime::reference::dequantize<uint8_t>(
                                    static_cast<uint8_t*>(arg0_tensor),
                                    static_cast<double*>(arg1_tensor),
                                    static_cast<uint8_t*>(arg2_tensor),
                                    static_cast<double*>(out_tensor),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes);
                            };
                        }
                        else
                        {
                            throw ngraph_error("Unsupported dequantization element type");
                        }
                    }
                    else
                    {
                        throw ngraph_error("Unsupported input element type");
                    }
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Quantize)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto quantize = static_cast<const ngraph::op::Quantize*>(node);
                    auto& functors = external_function->get_functors();
                    auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    auto scale_const_op =
                        std::dynamic_pointer_cast<ngraph::op::Constant>(quantize->get_argument(1));
                    if (scale_const_op == nullptr)
                    {
                        throw ngraph_error("Quantize scale must be a constant");
                    }
                    auto scale = scale_const_op->get_vector<float>();

                    std::vector<float> scales;
                    scales.push_back(1.0 / scale[0]);
                    size_t quantize_index =
                        mkldnn_emitter->build_quantize_reorder(input_desc, result_desc, scales);
                    auto& deps = mkldnn_emitter->get_primitive_deps(quantize_index);
                    auto functor = [&, quantize_index](CPURuntimeContext* ctx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, quantize_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    auto& functors = external_function->get_functors();
                    auto& tensor_data = external_function->get_tensor_data();

                    const ngraph::op::Quantize* quantize =
                        static_cast<const ngraph::op::Quantize*>(node);
                    function<void(CPURuntimeContext*)> functor;

                    auto& arg0_tensor = tensor_data[args[0].get_name()];
                    auto& arg1_tensor = tensor_data[args[1].get_name()];
                    auto& arg2_tensor = tensor_data[args[2].get_name()];
                    auto& out_tensor = tensor_data[out[0].get_name()];

                    auto arg0_shape = args[0].get_shape();
                    auto arg1_shape = args[1].get_shape();
                    auto daxes = quantize->get_axes();
                    op::Quantize::RoundMode round_mode = quantize->get_round_mode();

                    if (args[0].get_element_type() == element::f32)
                    {
                        if (out[0].get_element_type() == element::i8)
                        {
                            functor = [&, arg0_shape, arg1_shape, daxes, round_mode](
                                CPURuntimeContext* ctx) {
                                ngraph::runtime::reference::quantize<float>(
                                    static_cast<float*>(arg0_tensor),
                                    static_cast<float*>(arg1_tensor),
                                    static_cast<int8_t*>(arg2_tensor),
                                    static_cast<int8_t*>(out_tensor),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes,
                                    round_mode);
                            };
                        }
                        else if (out[0].get_element_type() == element::u8)
                        {
                            functor = [&, arg0_shape, arg1_shape, daxes, round_mode](
                                CPURuntimeContext* ctx) {
                                ngraph::runtime::reference::quantize<float>(
                                    static_cast<float*>(arg0_tensor),
                                    static_cast<float*>(arg1_tensor),
                                    static_cast<uint8_t*>(arg2_tensor),
                                    static_cast<uint8_t*>(out_tensor),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes,
                                    round_mode);
                            };
                        }
                        else
                        {
                            throw ngraph_error("Unsupported quantization element type");
                        }
                    }
                    else if (args[0].get_element_type() == element::f64)
                    {
                        if (out[0].get_element_type() == element::i8)
                        {
                            functor = [&, arg0_shape, arg1_shape, daxes, round_mode](
                                CPURuntimeContext* ctx) {
                                ngraph::runtime::reference::quantize<double>(
                                    static_cast<double*>(arg0_tensor),
                                    static_cast<double*>(arg1_tensor),
                                    static_cast<int8_t*>(arg2_tensor),
                                    static_cast<int8_t*>(out_tensor),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes,
                                    round_mode);
                            };
                        }
                        else if (out[0].get_element_type() == element::u8)
                        {
                            functor = [&, arg0_shape, arg1_shape, daxes, round_mode](
                                CPURuntimeContext* ctx) {
                                ngraph::runtime::reference::quantize<double>(
                                    static_cast<double*>(arg0_tensor),
                                    static_cast<double*>(arg1_tensor),
                                    static_cast<uint8_t*>(arg2_tensor),
                                    static_cast<uint8_t*>(out_tensor),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes,
                                    round_mode);
                            };
                        }
                        else
                        {
                            throw ngraph_error("Unsupported quantization element type");
                        }
                    }
                    else
                    {
                        throw ngraph_error("Unsupported input element type");
                    }

                    functors.emplace_back(functor);
                }
            }

            REGISTER_OP_BUILDER(Dequantize);
            REGISTER_OP_BUILDER(Quantize);
        }
    }
}
