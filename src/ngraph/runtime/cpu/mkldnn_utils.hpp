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
#include "ngraph/types/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace mkldnn_utils
            {
                extern mkldnn::engine global_cpu_engine;

                mkldnn::memory::data_type get_data_type(const ngraph::element::Type& et);

                bool is_mkldnn_op(ngraph::Node& op);

                mkldnn::memory::format
                    create_native_data_format(const ngraph::runtime::cpu::LayoutDescriptor& layout);
            }
        }
    }
}
