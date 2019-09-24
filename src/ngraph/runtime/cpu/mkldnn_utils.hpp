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

#pragma once

#include <mkldnn.hpp>
#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/type/element_type.hpp"

#if MKLDNN_VERSION_MAJOR < 1
#define FORMAT format
#define FORMAT_KIND format
#define FORMAT_KIND_UNDEF mkdnn_format_undef
#define FORMAT_ANY mkldnn_any
#define FORMAT_UNDEF mkldnn_undef
#define DATA_UNDEF data_undef

#define CHANGE_FORMAT                                                                              \
    if (weights_desc.data.format == mkldnn_nchw)                                                   \
    {                                                                                              \
        weights_desc.data.format = mkldnn_oihw;                                                    \
    }                                                                                              \
    if (weights_desc.data.format == mkldnn_ncdhw)                                                  \
    {                                                                                              \
        weights_desc.data.format = mkldnn_oidhw;                                                   \
    }

#define BN_FLAG_CLASS batch_normalization_flag

#define PADDING , mkldnn::padding_kind::zero

#define SET_ROUND_MODE attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);

#define QUERY_SCRATCHPAD(op_name, x)
#define QUERY_SCRATCHPAD_2ARGS(op_name, x, y)
#define QUERY_SCRATCHPAD_3ARGS(op_name, x, y, z)
#define QUERY_SCRATCHPAD_4ARGS(op_name, x, y, z, u)

#define MKLDNN_ERROR_MESSAGE e.message

#else
#define TENSOR_MAX_DIMS MKLDNN_MAX_NDIMS
#define FORMAT format_tag
#define FORMAT_KIND format_kind
#define FORMAT_KIND_UNDEF format_kind::undef
#define FORMAT_ANY static_cast<mkldnn_format_kind_t>(mkldnn::memory::format_kind::any)
#define FORMAT_UNDEF format_tag::undef
#define DATA_UNDEF undef

#define CHANGE_FORMAT

#define BN_FLAG_CLASS normalization_flags

#define PADDING

#define SET_ROUND_MODE

#define QUERY_SCRATCHPAD(op_name, x) mkldnn_emitter->query_scratchpad_##op_name(x)
#define QUERY_SCRATCHPAD_2ARGS(op_name, x, y) mkldnn_emitter->query_scratchpad_##op_name(x, y)
#define QUERY_SCRATCHPAD_3ARGS(op_name, x, y, z) mkldnn_emitter->query_scratchpad_##op_name(x, y, z)
#define QUERY_SCRATCHPAD_4ARGS(op_name, x, y, z, u)                                                \
    mkldnn_emitter->query_scratchpad_##op_name(x, y, z, u)

#define ATTR_S                                                                                     \
    mkldnn::primitive_attr attr;                                                                   \
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);

#define GET_SIZE                                                                                   \
    mkldnn::memory::desc scratchpad_md = pd.scratchpad_desc();                                     \
    size_t size = scratchpad_md.get_size();                                                        \
    m_max_scratchpad_size = size > m_max_scratchpad_size ? size : m_max_scratchpad_size;

#define MKLDNN_ERROR_MESSAGE std::string(e.message)

#endif

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace mkldnn_utils
            {
                extern mkldnn::engine global_cpu_engine;
#ifndef _WIN32
                extern "C" void mkl_serv_free_buffers();
#endif
                mkldnn::memory::FORMAT
                    CreateNativeDataFormat(const ngraph::runtime::cpu::LayoutDescriptor& layout);
                mkldnn::memory::FORMAT CreateNativeDataFormat(const Shape& shape);
                const std::string& get_mkldnn_data_type_string(const ngraph::element::Type& type);
                mkldnn::memory::data_type get_mkldnn_data_type(const ngraph::element::Type& type);
                const std::string& get_mkldnn_format_string(mkldnn::memory::FORMAT fmt);

                const mkldnn::memory::desc& get_input_mkldnn_md(const Node* node, size_t index);
                const mkldnn::memory::desc& get_output_mkldnn_md(const Node* node, size_t index);

                mkldnn::memory::desc create_default_mkldnn_md(const Node* node,
                                                              size_t index,
                                                              bool is_output,
                                                              mkldnn::memory::FORMAT format);
                bool is_perm_sorted(const Strides& a, const AxisVector& perm);
                bool can_create_mkldnn_md(const ngraph::element::Type type);
                bool can_create_mkldnn_md(const Shape& dims,
                                          const Strides& strides,
                                          const ngraph::element::Type type);
                mkldnn::memory::desc create_blocked_mkldnn_md(const Shape& dims,
                                                              const Strides& strides,
                                                              const ngraph::element::Type type);
                mkldnn::memory::desc
                    create_blocked_mkldnn_md_helper(const mkldnn::memory::dims& dim,
                                                    const Strides& strides,
                                                    const mkldnn::memory::dims& stride,
                                                    const mkldnn::memory::data_type dtype);
                mkldnn::memory::desc try_get_named_md(const mkldnn_memory_desc_t& md);
                mkldnn::memory::desc rotate_blocked_md(const mkldnn::memory::desc& in,
                                                       const AxisVector& axis_order);
                mkldnn::memory::desc squeeze_blocked_md(const mkldnn::memory::desc& in,
                                                        AxisVector& axis_list);
                mkldnn::memory::desc expand_blocked_md(const mkldnn::memory::desc& in,
                                                       AxisVector& axis_list);

                bool compare_mkldnn_formats(mkldnn::memory::FORMAT lhs, mkldnn::memory::FORMAT rhs);
                bool compare_mkldnn_mds(const mkldnn::memory::desc& lhs,
                                        const mkldnn::memory::desc& rhs);
                bool is_mkldnn_padded_layout(const mkldnn::memory::desc& in,
                                             const AxisVector& axis_list);
                bool is_mkldnn_filter_format(mkldnn::memory::FORMAT fmt);
                bool is_mkldnn_blocked_data_format(mkldnn::memory::FORMAT fmt);
                bool can_use_mkldnn_batchnorm_fprop(const ngraph::Node* node);
                bool can_use_mkldnn_batchnorm_bprop(const ngraph::Node* node);

                //
                // Intel(R) MKL-DNN supports the Winograd algorithm for convolutions with the
                // following sizes:
                // 2D convolution (i.e. spatial depth d=1), kernel sizes kh=3,kw=3. strides sh=sw=1.
                // Inference - Based on convolution sizes, MKLDNN chooses between two different tile
                // sizes F(2x2, 3x3) or F(4x4, 3x3)(refer to Winograd paper for more informartion on
                // tile sizes). Training - Uses F(4x4, 3x3) winograd.
                //
                mkldnn::algorithm get_conv_algo();

                // Placeholder for when "auto" support is added for deconv
                mkldnn::algorithm get_deconv_algo();

                bool use_mkldnn_kernel(const ngraph::Node* node);
                void assign_mkldnn_kernel(Node* node);

                std::map<element::Type, const mkldnn::memory::data_type>&
                    get_mkldnn_data_type_map();
                std::map<element::Type, const std::string>& get_mkldnn_data_type_string_map();
                std::map<mkldnn::memory::FORMAT, const std::string>& get_mkldnn_format_string_map();
                std::set<mkldnn::memory::FORMAT>& get_filter_formats();
                template <typename T>
                bool can_use_mkldnn_conv(ngraph::Node* node)
                {
                    auto convolution = static_cast<const T*>(node);
                    auto arg0_rank = node->get_input_shape(0).size();

                    for (size_t s : convolution->get_data_dilation_strides())
                    {
                        if (s != 1)
                            return false;
                    }
                    // MKLDNN doesnt support negative padding
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
                    return true;
                }

#if MKLDNN_VERSION_MAJOR >= 1
                std::map<mkldnn::memory::format_kind, const std::string>&
                    get_mkldnn_format_kind_string_map();
                const std::string&
                    get_mkldnn_format_kind_string(mkldnn::memory::format_kind fmt_kind);
                bool inline compare_mkldnn_dims(mkldnn_dims_t& arr1,
                                                mkldnn_dims_t& arr2,
                                                size_t size);
                bool compare_mkldnn_strides_order(mkldnn_dims_t& stride1,
                                                  mkldnn_dims_t& stride2,
                                                  size_t size);
                bool compare_mkldnn_md_formats(const mkldnn::memory::desc& lhs,
                                               const mkldnn::memory::desc& rhs);
                bool mkldnn_md_matches_format_tag(const mkldnn::memory::desc&,
                                                  const mkldnn::memory::format_tag&);
                mkldnn::memory::desc create_default_mkldnn_md_with_strides(
                    const Node* node, size_t index, mkldnn::memory::dims& strides, bool is_output);
                bool is_mkldnn_desc_blocked_data_format(const mkldnn::memory::desc& desc);
#endif
            }
        }
    }
}
