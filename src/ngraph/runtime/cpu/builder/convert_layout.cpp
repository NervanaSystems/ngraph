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

#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/group_conv.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::runtime::cpu::op::ConvertLayout)
            {
                auto& functors = external_function->get_functors();

                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                auto in_shape = args[0].get_shape();
                auto out_name = out[0].get_name();
                auto out_shape = out[0].get_shape();
                if (input_desc.data.format == mkldnn_nchw &&
                    result_desc.data.format == mkldnn_goihw)
                {
                    //becomes a copy
                    input_desc = result_desc;
                }
                else if (input_desc.data.format == mkldnn_nchw && input_desc.data.ndims == 4 &&
                         result_desc.data.ndims == 5 && node->get_users().size() == 1)
                {
                    Shape weights_shape_groups;
                    if (auto gconv = std::dynamic_pointer_cast<ngraph::op::GroupConvolution>(
                            node->get_users()[0]))
                    {
                        weights_shape_groups = gconv->get_weights_dimensions();
                    }
                    else if (auto gconvb =
                                 std::dynamic_pointer_cast<ngraph::op::GroupConvolutionBias>(
                                     node->get_users()[0]))
                    {
                        weights_shape_groups = gconvb->get_weights_dimensions();
                    }
                    else
                    {
                        throw ngraph_error("Incompatible input/output shape in ConvertLayout op");
                    }
                    input_desc = mkldnn::memory::desc(
                        mkldnn::memory::dims(weights_shape_groups.begin(),
                                             weights_shape_groups.end()),
                        mkldnn_utils::get_mkldnn_data_type(args[0].get_element_type()),
                        mkldnn::memory::format::goihw);
                }

                size_t reorder_index = mkldnn_emitter->build_reorder(input_desc, result_desc);

                std::cout << __func__ << std::endl;
                std::cout << "input: " << args[0].get_name() << " " << ngraph::vector_to_string(in_shape) << std::endl;
                std::cout << "output: " << out[0].get_name() << " " << ngraph::vector_to_string(out_shape) << std::endl;
                std::cout << "input desc: " << input_desc.data.format << std::endl;
                std::cout << "result desc: " << result_desc.data.format << std::endl;

                auto& deps = mkldnn_emitter->get_primitive_deps(reorder_index);
                auto functor = [&, input_desc, result_desc, in_shape, out_shape, out_name, reorder_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* ectx) {

                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, reorder_index);
#if 0
                    if (out_name == "ConvertLayout_5933_0") {
                      std::cout << "ConvertLayout_5933_0" << std::endl;
                      std::cout << "### convertlayout input:" << std::endl;
                      for (size_t i = 0; i < 1000; ++i) {
                        std::cout << "i = " << i << std::endl;
                        for (size_t j = 0; j < 16; ++j) {
                          std::cout << "  j = " << j << ": ";
                          for (size_t k = 0; k < 784; ++k) {
                            std::cout << static_cast<float*>(arg_tensor)[i*(16*784)+j*784+k] << " ";
                          }
                          std::cout << std::endl;
                        }
                      }
                      std::cout << "### convertlayout output:" << std::endl;
                      for (size_t i = 0; i < 1000; ++i) {
                        std::cout << "i = " << i << std::endl;
                        for (size_t j = 0; j < 9; ++j) {
                          std::cout << "  j = " << j << ": ";
                          for (size_t k = 0; k < 784; ++k) {
//                          for (size_t k = 0; k < 16; ++k) {
                            std::cout << static_cast<float*>(out_tensor)[i*(9*784)+j*784+k] << " ";
                          }
                          std::cout << std::endl;
                        }
                      }
                    }
#endif
#if 0
                    if (ngraph::shape_size(in_shape) == 50176000) {
                      std::cout << "input desc: " << input_desc.data.format << std::endl;
                      std::cout << "result desc: " << result_desc.data.format << std::endl;
                      std::cout << "### convertlayout input:" << std::endl;
                      for (size_t i = 0; i < 1000; ++i) {
                        std::cout << "i = " << i << std::endl;
                        for (size_t j = 0; j < 256; ++j) {
                          std::cout << "  j = " << j << ": ";
               //           for (size_t k = 0; k < 196; ++k) {
                          for (size_t k = 0; k < 16; ++k) {
                            std::cout << static_cast<float*>(arg_tensor)[i*(256*196)+j*196+k] << " ";
                          }
                          std::cout << std::endl;
                        }
                      }
                    }
                    if (ngraph::shape_size(in_shape) == 50176000) {
                      std::cout << "### convertlayout output:" << std::endl;
                      for (size_t i = 0; i < 1; ++i) {
                        std::cout << "i = " << i << std::endl;
                        for (size_t j = 0; j < 256; ++j) {
                          std::cout << "  j = " << j << ": ";
                          for (size_t k = 0; k < 196; ++k) {
//                          for (size_t k = 0; k < 16; ++k) {
                            std::cout << static_cast<float*>(out_tensor)[i*(256*196)+j*196+k] << " ";
                          }
                          std::cout << std::endl;
                        }
                      }
                    }
#endif
                };
                functors.emplace_back(functor);
            }
        }
    }
}
