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

#include <mkldnn.hpp>

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace mkldnn_utils
            {
                extern mkldnn::engine global_cpu_engine;

                mkldnn::memory::format
                    CreateNativeDataFormat(const ngraph::runtime::cpu::LayoutDescriptor& layout);
                mkldnn::memory::format CreateNativeDataFormat(const Shape& shape);
                const std::string& get_mkldnn_data_type_string(const ngraph::element::Type& type);
                mkldnn::memory::data_type get_mkldnn_data_type(const ngraph::element::Type& type);
                const std::string& get_mkldnn_format_string(mkldnn::memory::format fmt);

                const mkldnn::memory::desc& get_input_mkldnn_md(const Node* node, size_t index);
                const mkldnn::memory::desc& get_output_mkldnn_md(const Node* node, size_t index);

                mkldnn::memory::desc create_default_mkldnn_md(const Node* node,
                                                              size_t index,
                                                              bool is_output,
                                                              mkldnn::memory::format format);
                bool is_perm_sorted(const Strides& a, const AxisVector& perm);
                bool can_create_mkldnn_md(const Shape& dims,
                                          const Strides& strides,
                                          const ngraph::element::Type type);
                mkldnn::memory::desc create_blocked_mkldnn_md(const Shape& dims,
                                                              const Strides& strides,
                                                              const ngraph::element::Type type);
                mkldnn::memory::desc try_get_named_md(mkldnn_memory_desc_t md);
                mkldnn::memory::desc rotate_blocked_md(const mkldnn::memory::desc& in,
                                                       const AxisVector& axis_order);
                mkldnn::memory::desc squeeze_blocked_md(const mkldnn::memory::desc& in,
                                                        AxisVector& axis_list);
                mkldnn::memory::desc expand_blocked_md(const mkldnn::memory::desc& in,
                                                       AxisVector& axis_list);

                bool compare_mkldnn_formats(mkldnn::memory::format lhs, mkldnn::memory::format rhs);
                bool compare_mkldnn_mds(const mkldnn::memory::desc& lhs,
                                        const mkldnn::memory::desc& rhs);
                bool is_mkldnn_padded_layout(const mkldnn::memory::desc& in,
                                             const AxisVector& axis_list);
                bool is_mkldnn_filter_format(mkldnn::memory::format fmt);
                bool is_mkldnn_blocked_data_format(mkldnn::memory::format fmt);

                bool use_mkldnn_kernel(const ngraph::Node* node);

                std::map<element::Type, const mkldnn::memory::data_type>&
                    get_mkldnn_data_type_map();
                std::map<element::Type, const std::string>& get_mkldnn_data_type_string_map();
                std::map<mkldnn::memory::format, const std::string>& get_mkldnn_format_string_map();
                std::set<mkldnn::memory::format>& get_filter_formats();
            }
        }
    }
}
