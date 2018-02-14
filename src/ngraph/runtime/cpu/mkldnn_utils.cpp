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

#include <typeindex>
#include <typeinfo>
#include <unordered_set>

#include "ngraph/node.hpp"
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/max_pool.hpp"

#include "mkldnn_utils.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace mkldnn_utils
            {
#define TI(x) std::type_index(typeid(x))

                const std::unordered_set<std::type_index> s_op_registry{
                    TI(ngraph::op::AvgPool),
                    TI(ngraph::op::AvgPoolBackprop),
                    TI(ngraph::op::Convolution),
                    TI(ngraph::op::ConvolutionBackpropData),
                    TI(ngraph::op::ConvolutionBackpropFilters),
                    TI(ngraph::op::MaxPool)};

                bool IsMKLDNNOp(ngraph::Node& op)
                {
                    return (s_op_registry.find(TI(op)) != s_op_registry.end());
                }

                mkldnn::memory::format
                    CreateNativeDataFormat(const ngraph::runtime::cpu::LayoutDescriptor& layout)
                {
                    switch (layout.get_shape().size())
                    {
                    case 1: return mkldnn::memory::format::x;
                    case 2: return mkldnn::memory::format::nc;
                    case 4: return mkldnn::memory::format::nchw;
                    default: return mkldnn::memory::format::format_undef;
                    }
                }
            }
        }
    }
}
