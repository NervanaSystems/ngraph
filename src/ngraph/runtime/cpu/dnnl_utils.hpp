//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <dnnl.hpp>
#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/type/element_type.hpp"

#define TENSOR_MAX_DIMS DNNL_MAX_NDIMS
#define FORMAT format_tag
#define FORMAT_KIND format_kind
#define FORMAT_KIND_UNDEF format_kind::undef
#define FORMAT_ANY static_cast<dnnl_format_kind_t>(dnnl::memory::format_kind::any)
#define FORMAT_UNDEF format_tag::undef
#define DATA_UNDEF undef

#define CHANGE_FORMAT

#define BN_FLAG_CLASS normalization_flags

#define PADDING

#define SET_ROUND_MODE

#define QUERY_SCRATCHPAD(op_name, x) dnnl_emitter->query_scratchpad_##op_name(x)
#define QUERY_SCRATCHPAD_2ARGS(op_name, x, y) dnnl_emitter->query_scratchpad_##op_name(x, y)
#define QUERY_SCRATCHPAD_3ARGS(op_name, x, y, z) dnnl_emitter->query_scratchpad_##op_name(x, y, z)
#define QUERY_SCRATCHPAD_4ARGS(op_name, x, y, z, u)                                                \
    dnnl_emitter->query_scratchpad_##op_name(x, y, z, u)

#define ATTR_S                                                                                     \
    dnnl::primitive_attr attr;                                                                     \
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#define GET_SIZE                                                                                   \
    dnnl::memory::desc scratchpad_md = pd.scratchpad_desc();                                       \
    size_t size = scratchpad_md.get_size();                                                        \
    m_max_scratchpad_size = size > m_max_scratchpad_size ? size : m_max_scratchpad_size;           \
    return size;

#define DNNL_ERROR_MESSAGE std::string(e.message)

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace dnnl_utils
            {
                extern dnnl::engine global_cpu_engine;
                dnnl::memory::FORMAT
                    CreateNativeDataFormat(const ngraph::runtime::cpu::LayoutDescriptor& layout);
                dnnl::memory::FORMAT CreateNativeDataFormat(const Shape& shape);
                const std::string& get_dnnl_data_type_string(const ngraph::element::Type& type);
                dnnl::memory::data_type get_dnnl_data_type(const ngraph::element::Type& type);
                const std::string& get_dnnl_format_string(dnnl::memory::FORMAT fmt);

                const dnnl::memory::desc& get_input_dnnl_md(const Node* node, size_t index);
                const dnnl::memory::desc& get_output_dnnl_md(const Node* node, size_t index);

                dnnl::memory::desc create_default_dnnl_md(const Node* node,
                                                          size_t index,
                                                          bool is_output,
                                                          dnnl::memory::FORMAT format);
                bool is_perm_sorted(const Strides& a, const AxisVector& perm);
                bool can_create_dnnl_md(const ngraph::element::Type type);
                bool can_create_dnnl_md(const Shape& dims,
                                        const Strides& strides,
                                        const ngraph::element::Type type);
                dnnl::memory::desc create_blocked_dnnl_md(const Shape& dims,
                                                          const Strides& strides,
                                                          const ngraph::element::Type type);
                dnnl::memory::desc
                    create_blocked_dnnl_md_helper(const dnnl::memory::dims& dim,
                                                  const Strides& strides,
                                                  const dnnl::memory::dims& stride,
                                                  const dnnl::memory::data_type dtype);
                dnnl::memory::desc try_get_named_md(const dnnl_memory_desc_t& md);
                dnnl::memory::desc rotate_blocked_md(const dnnl::memory::desc& in,
                                                     const AxisVector& axis_order);
                dnnl::memory::desc squeeze_blocked_md(const dnnl::memory::desc& in,
                                                      AxisVector& axis_list);
                dnnl::memory::desc expand_blocked_md(const dnnl::memory::desc& in,
                                                     AxisVector& axis_list);

                bool compare_dnnl_formats(dnnl::memory::FORMAT lhs, dnnl::memory::FORMAT rhs);
                bool compare_dnnl_mds(const dnnl::memory::desc& lhs, const dnnl::memory::desc& rhs);
                bool is_dnnl_padded_layout(const dnnl::memory::desc& in,
                                           const AxisVector& axis_list);
                bool is_dnnl_filter_format(dnnl::memory::FORMAT fmt);
                bool is_dnnl_blocked_data_format(dnnl::memory::FORMAT fmt);
                bool can_use_dnnl_batchnorm_fprop(const ngraph::Node* node);
                bool can_use_dnnl_batchnorm_bprop(const ngraph::Node* node);

                bool CPU_BACKEND_API is_bf16_supported();

                //
                // Intel(R) MKL-DNN supports the Winograd algorithm for convolutions with the
                // following sizes:
                // 2D convolution (i.e. spatial depth d=1), kernel sizes kh=3,kw=3. strides sh=sw=1.
                // Inference - Based on convolution sizes, DNNL chooses between two different tile
                // sizes F(2x2, 3x3) or F(4x4, 3x3)(refer to Winograd paper for more informartion on
                // tile sizes). Training - Uses F(4x4, 3x3) winograd.
                //
                dnnl::algorithm get_conv_algo();

                // Placeholder for when "auto" support is added for deconv
                dnnl::algorithm get_deconv_algo();

                bool use_dnnl_kernel(const ngraph::Node* node);
                void assign_dnnl_kernel(Node* node);

                std::map<element::Type, const dnnl::memory::data_type>& get_dnnl_data_type_map();
                std::map<element::Type, const std::string>& get_dnnl_data_type_string_map();
                std::map<dnnl::memory::FORMAT, const std::string>& get_dnnl_format_string_map();
                std::set<dnnl::memory::FORMAT>& get_filter_formats();
                template <typename T>
                bool can_use_dnnl_conv(ngraph::Node* node)
                {
                    auto convolution = static_cast<const T*>(node);
                    auto arg0_rank = node->get_input_shape(0).size();

                    for (size_t s : convolution->get_data_dilation_strides())
                    {
                        if (s != 1)
                            return false;
                    }
                    // DNNL doesnt support negative padding
                    for (auto s : convolution->get_padding_above())
                    {
                        if (s < 0)
                        {
                            return false;
                        }
                    }

                    for (auto s : convolution->get_padding_below())
                    {
                        if (s < 0)
                        {
                            return false;
                        }
                    }

                    if (arg0_rank != 3 && arg0_rank != 4 && arg0_rank != 5)
                    {
                        return false;
                    }
                    // Data
                    if (node->get_input_element_type(0) != element::f32 &&
                        node->get_input_element_type(0) != element::i8 &&
                        node->get_input_element_type(0) != element::u8 &&
                        node->get_input_element_type(0) != element::bf16)
                    {
                        return false;
                    }
                    // Weights
                    if (node->get_input_element_type(1) != element::f32 &&
                        node->get_input_element_type(1) != element::i8 &&
                        node->get_input_element_type(1) != element::bf16)
                    {
                        return false;
                    }
                    // Outputs
                    if (node->get_output_element_type(0) != element::f32 &&
                        node->get_output_element_type(0) != element::i8 &&
                        node->get_output_element_type(0) != element::u8 &&
                        node->get_output_element_type(0) != element::i32 &&
                        node->get_output_element_type(0) != element::bf16)
                    {
                        return false;
                    }

                    // Check if bf16 is supported on the platform
                    if (!is_bf16_supported() && (node->get_input_element_type(0) == element::bf16 ||
                                                 node->get_input_element_type(1) == element::bf16 ||
                                                 node->get_output_element_type(0) == element::bf16))
                    {
                        return false;
                    }

                    return true;
                }

                std::map<dnnl::memory::format_kind, const std::string>&
                    get_dnnl_format_kind_string_map();
                const std::string& get_dnnl_format_kind_string(dnnl::memory::format_kind fmt_kind);
                bool inline compare_dnnl_dims(dnnl_dims_t& arr1, dnnl_dims_t& arr2, size_t size);
                bool compare_dnnl_strides_order(dnnl_dims_t& stride1,
                                                dnnl_dims_t& stride2,
                                                size_t size);
                bool compare_dnnl_md_formats(const dnnl::memory::desc& lhs,
                                             const dnnl::memory::desc& rhs);
                bool dnnl_md_matches_format_tag(const dnnl::memory::desc&,
                                                const dnnl::memory::format_tag&);
                dnnl::memory::desc create_default_dnnl_md_with_strides(const Node* node,
                                                                       size_t index,
                                                                       dnnl::memory::dims& strides,
                                                                       bool is_output);
                bool is_dnnl_desc_blocked_data_format(const dnnl::memory::desc& desc);
            }
        }
    }
}
