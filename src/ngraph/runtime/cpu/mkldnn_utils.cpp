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
#include <unordered_map>
#include <unordered_set>

#include "ngraph/node.hpp"
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/max_pool.hpp"

#include "mkldnn_utils.hpp"

using namespace mkldnn;
using namespace ngraph;

#define TI(x) std::type_index(typeid(x))

const std::unordered_set<std::type_index> s_op_registry{TI(ngraph::op::AvgPool),
                                                        TI(ngraph::op::AvgPoolBackprop),
                                                        TI(ngraph::op::Convolution),
                                                        TI(ngraph::op::ConvolutionBackpropData),
                                                        TI(ngraph::op::ConvolutionBackpropFilters),
                                                        TI(ngraph::op::MaxPool)};

// Mapping from POD types to MKLDNN data types
// An empty string implies the corresponding MKLDNN data type
// is not supported
static const std::unordered_map<std::string, const std::string> mkldnn_data_type_string_map{
    {"char", "memory::data_type::s8"},
    {"float", "memory::data_type::f32"},
    {"double", ""},
    {"int8_t", "memory::data_type::s8"},
    {"int16_t", "memory::data_type::s16"},
    {"int32_t", "memory::data_type::s32"},
    {"int64_t", ""},
    {"uint8_t", "memory::data_type::u8"},
    {"uint16_t", ""},
    {"uint32_t", ""},
    {"uint64_t", ""}};

static const std::unordered_map<std::string, memory::data_type> mkldnn_data_type_map{
    {"char", memory::data_type::s8},
    {"float", memory::data_type::f32},
    {"double", memory::data_type::data_undef},
    {"int8_t", memory::data_type::s8},
    {"int16_t", memory::data_type::s16},
    {"int32_t", memory::data_type::s32},
    {"int64_t", memory::data_type::data_undef},
    {"uint8_t", memory::data_type::u8},
    {"uint16_t", memory::data_type::data_undef},
    {"uint32_t", memory::data_type::data_undef},
    {"uint64_t", memory::data_type::data_undef}};

bool runtime::cpu::mkldnn_utils::IsMKLDNNOp(ngraph::Node& op)
{
    return (s_op_registry.find(TI(op)) != s_op_registry.end());
}

mkldnn::memory::format runtime::cpu::mkldnn_utils::CreateNativeDataFormat(
    const ngraph::runtime::cpu::LayoutDescriptor& layout)
{
    switch (layout.get_shape().size())
    {
    case 1: return mkldnn::memory::format::x;
    case 2: return mkldnn::memory::format::nc;
    case 4: return mkldnn::memory::format::nchw;
    default: return mkldnn::memory::format::format_undef;
    }
}

const std::string& runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(const std::string& type)
{
    auto it = mkldnn_data_type_string_map.find(type);
    if (it == mkldnn_data_type_string_map.end() || it->second.empty())
        throw ngraph_error("No MKLDNN data type exists for the given element type");
    return it->second;
}

mkldnn::memory::data_type runtime::cpu::mkldnn_utils::get_mkldnn_data_type(const std::string& type)
{
    auto it = mkldnn_data_type_map.find(type);
    if (it == mkldnn_data_type_map.end() || it->second == memory::data_type::data_undef)
    {
        throw ngraph_error("No MKLDNN data type exists for the given element type");
    }
    return it->second;
}
