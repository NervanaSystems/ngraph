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

#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

#include "ngraph/node.hpp"
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/batch_norm.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/relu.hpp"

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

                static const std::unordered_set<std::type_index> s_op_registry{
                    TI(ngraph::op::AvgPool),
                    TI(ngraph::op::AvgPoolBackprop),
                    TI(ngraph::op::BatchNorm),
                    TI(ngraph::op::Convolution),
                    TI(ngraph::op::ConvolutionBackpropData),
                    TI(ngraph::op::ConvolutionBackpropFilters),
                    TI(ngraph::op::MaxPool),
                    TI(ngraph::op::MaxPoolBackprop),
                    TI(ngraph::op::Relu),
                    TI(ngraph::op::ReluBackprop)};

                static const std::unordered_map<std::string, const mkldnn::memory::data_type>
                    s_data_type_map{{"char", mkldnn::memory::data_type::s8},
                                    {"float", mkldnn::memory::data_type::f32},
                                    {"double", mkldnn::memory::data_type::data_undef},
                                    {"int8_t", mkldnn::memory::data_type::s8},
                                    {"int16_t", mkldnn::memory::data_type::s16},
                                    {"int32_t", mkldnn::memory::data_type::s32},
                                    {"int64_t", mkldnn::memory::data_type::data_undef},
                                    {"uint8_t", mkldnn::memory::data_type::u8},
                                    {"uint16_t", mkldnn::memory::data_type::data_undef},
                                    {"uint32_t", mkldnn::memory::data_type::data_undef},
                                    {"uint64_t", mkldnn::memory::data_type::data_undef}};

                mkldnn::memory::data_type GetDataType(const ngraph::element::Type& et)
                {
                    auto it = s_data_type_map.find(et.c_type_string());
                    if (it == s_data_type_map.end() ||
                        it->second == mkldnn::memory::data_type::data_undef)
                        throw ngraph_error("No MKLDNN data type exists for the given element type");
                    return it->second;
                }

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
