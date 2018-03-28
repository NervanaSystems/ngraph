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

#include <mkldnn.hpp>

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

                bool IsMKLDNNOp(ngraph::Node& op);
                mkldnn::memory::format
                    CreateNativeDataFormat(const ngraph::runtime::cpu::LayoutDescriptor& layout);
                const std::string& get_mkldnn_data_type_string(const ngraph::element::Type& type);
                mkldnn::memory::data_type get_mkldnn_data_type(const ngraph::element::Type& type);
                const std::string& get_mkldnn_format_string(mkldnn::memory::format fmt);

                mkldnn::memory::format get_input_mkldnn_format(const Node* node, size_t index);
                mkldnn::memory::format get_output_mkldnn_format(const Node* node, size_t index);
                bool use_mkldnn_kernel(const ngraph::Node* node);
                bool compare_mkldnn_formats(mkldnn::memory::format fmt1,
                                            mkldnn::memory::format fmt2);
                bool is_mkldnn_filter_format(mkldnn::memory::format fmt);
            }
        }
    }
}
