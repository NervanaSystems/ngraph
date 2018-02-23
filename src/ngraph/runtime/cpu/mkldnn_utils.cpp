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
#include <unordered_set>

#include "ngraph/node.hpp"
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/batch_norm.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/relu.hpp"
#include "ngraph/types/element_type.hpp"

#include "mkldnn_utils.hpp"

using namespace mkldnn;
using namespace ngraph;

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

// Mapping from POD types to MKLDNN data types
static const std::map<element::Type, const mkldnn::memory::data_type> s_mkldnn_data_type_map{
    {element::boolean, mkldnn::memory::data_type::s8},
    {element::f32, mkldnn::memory::data_type::f32},
    {element::f64, mkldnn::memory::data_type::data_undef},
    {element::i8, mkldnn::memory::data_type::s8},
    {element::i16, mkldnn::memory::data_type::s16},
    {element::i32, mkldnn::memory::data_type::s32},
    {element::i64, mkldnn::memory::data_type::data_undef},
    {element::u8, mkldnn::memory::data_type::u8},
    {element::u16, mkldnn::memory::data_type::data_undef},
    {element::u32, mkldnn::memory::data_type::data_undef},
    {element::u64, mkldnn::memory::data_type::data_undef}};

static const std::map<element::Type, const std::string> s_mkldnn_data_type_string_map{
    {element::boolean, "mkldnn::memory::data_type::s8"},
    {element::f32, "mkldnn::memory::data_type::f32"},
    {element::f64, "mkldnn::memory::data_type::data_undef"},
    {element::i8, "mkldnn::memory::data_type::s8"},
    {element::i16, "mkldnn::memory::data_type::s16"},
    {element::i32, "mkldnn::memory::data_type::s32"},
    {element::i64, "mkldnn::memory::data_type::data_undef"},
    {element::u8, "mkldnn::memory::data_type::u8"},
    {element::u16, "mkldnn::memory::data_type::data_undef"},
    {element::u32, "mkldnn::memory::data_type::data_undef"},
    {element::u64, "mkldnn::memory::data_type::data_undef"}};

// TODO (jbobba): Add the rest of memory formats to this map as well
static const std::map<memory::format, const std::string> s_mkldnn_format_string_map{
    {memory::format::format_undef, "memory::format::format_undef"},
    {memory::format::any, "memory::format::any"},
    {memory::format::blocked, "memory::format::blocked"},
    {memory::format::x, "memory::format::x"},
    {memory::format::nc, "memory::format::nc"},
    {memory::format::nchw, "memory::format::nchw"},
    {memory::format::nhwc, "memory::format::nhwc"},
    {memory::format::chwn, "memory::format::chwn"},
    {memory::format::nChw8c, "memory::format::nChw8c"},
    {memory::format::nChw16c, "memory::format::nChw16c"},
    {memory::format::oi, "memory::format::oi"},
    {memory::format::io, "memory::format::io"},
    {memory::format::oihw, "memory::format::oihw"},
    {memory::format::ihwo, "memory::format::ihwo"},
    {memory::format::hwio, "memory::format::hwio"},
    {memory::format::oIhw8i, "memory::format::oIhw8i"},
    {memory::format::oIhw16i, "memory::format::oIhw16i"},
    {memory::format::OIhw8i8o, "memory::format::OIhw8i8o"},
    {memory::format::OIhw16i16o, "memory::format::OIhw16i16o"},
    {memory::format::OIhw8o8i, "memory::format::OIhw8o8i"},
    {memory::format::OIhw16o16i, "memory::format::OIhw16o16i"},
    {memory::format::Oihw8o, "memory::format::Oihw8o"},
    {memory::format::Oihw16o, "memory::format::Oihw16o"},
    {memory::format::Ohwi8o, "memory::format::Ohwi8o"},
    {memory::format::Ohwi16o, "memory::format::Ohwi16o"},
    {memory::format::OhIw16o4i, "memory::format::OhIw16o4i"},
};

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

const std::string&
    runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(const ngraph::element::Type& type)
{
    auto it = s_mkldnn_data_type_string_map.find(type);
    if (it == s_mkldnn_data_type_string_map.end() || it->second.empty())
        throw ngraph_error("No MKLDNN data type exists for the given element type");
    return it->second;
}

mkldnn::memory::data_type
    runtime::cpu::mkldnn_utils::get_mkldnn_data_type(const ngraph::element::Type& type)
{
    auto it = s_mkldnn_data_type_map.find(type);
    if (it == s_mkldnn_data_type_map.end() || it->second == memory::data_type::data_undef)
    {
        throw ngraph_error("No MKLDNN data type exists for the given element type");
    }
    return it->second;
}

const std::string& runtime::cpu::mkldnn_utils::get_mkldnn_format_string(memory::format fmt)
{
    auto it = s_mkldnn_format_string_map.find(fmt);
    if (it == s_mkldnn_format_string_map.end())
        throw ngraph_error("No MKLDNN format exists for the given format type " +
                           std::to_string(fmt));
    return it->second;
}
