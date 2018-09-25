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

#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_set>

#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/type/element_type.hpp"

#include "mkldnn_utils.hpp"

using namespace mkldnn;
using namespace ngraph;
using namespace std;

#define TI(x) std::type_index(typeid(x))

static const std::unordered_set<std::type_index> s_op_registry{
    TI(ngraph::op::Add),
    TI(ngraph::op::AvgPool),
    TI(ngraph::op::AvgPoolBackprop),
    TI(ngraph::op::BatchNorm),
    TI(ngraph::op::BatchNormBackprop),
    TI(ngraph::op::Concat),
    TI(ngraph::op::Convolution),
    TI(ngraph::op::ConvolutionBackpropData),
    TI(ngraph::op::ConvolutionBackpropFilters),
    TI(ngraph::op::ConvolutionBias),
    TI(ngraph::op::ConvolutionRelu),
    TI(ngraph::op::ConvolutionBiasBackpropFiltersBias),
    TI(ngraph::op::MaxPool),
    TI(ngraph::op::MaxPoolBackprop),
    TI(ngraph::op::Relu),
    TI(ngraph::op::ReluBackprop),
    TI(ngraph::op::Reshape)};

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
    {memory::format::ncdhw, "memory::format::ndhwc"},
    {memory::format::ncdhw, "memory::format::ndhwc"},
    {memory::format::nCdhw16c, "memory::format::nCdhw16c"},
    {memory::format::oi, "memory::format::oi"},
    {memory::format::io, "memory::format::io"},
    {memory::format::oihw, "memory::format::oihw"},
    {memory::format::ihwo, "memory::format::ihwo"},
    {memory::format::hwio, "memory::format::hwio"},
    // TODO (nishant): Uncomment after the next release of mkl-dnn"
    //{memory::format::dhwio, "memory::format::dhwio"},
    {memory::format::oidhw, "memory::format::oidhw"},
    {memory::format::OIdhw16i16o, "memory::format::OIdhw16i16o"},
    {memory::format::OIdhw16o16i, "memory::format::OIdhw16o16i"},
    {memory::format::Oidhw16o, "memory::format::Oidhw16o"},
    {memory::format::Odhwi16o, "memory::format::Odhwi16o"},
    {memory::format::oIhw8i, "memory::format::oIhw8i"},
    {memory::format::oIhw16i, "memory::format::oIhw16i"},
    {memory::format::OIhw8i8o, "memory::format::OIhw8i8o"},
    {memory::format::OIhw16i16o, "memory::format::OIhw16i16o"},
    {memory::format::IOhw16o16i, "memory::format::IOhw16o16i"},
    {memory::format::OIhw8o8i, "memory::format::OIhw8o8i"},
    {memory::format::OIhw16o16i, "memory::format::OIhw16o16i"},
    {memory::format::Oihw8o, "memory::format::Oihw8o"},
    {memory::format::Oihw16o, "memory::format::Oihw16o"},
    {memory::format::Ohwi8o, "memory::format::Ohwi8o"},
    {memory::format::Ohwi16o, "memory::format::Ohwi16o"},
    {memory::format::OhIw16o4i, "memory::format::OhIw16o4i"},
    {memory::format::tnc, "memory::format::tnc"},
    {memory::format::ldsnc, "memory::format::ldsnc"},
    {memory::format::ldigo, "memory::format::ldigo"},
    {memory::format::ldgo, "memory::format::ldgo"},
    {memory::format::ldgo, "memory::format::Goihw8g"},
    {memory::format::ldgo, "memory::format::Goihw16g"},
};

static const std::set<memory::format> s_filter_formats{
    memory::format::oihw,
    memory::format::ihwo,
    memory::format::hwio,
    // TODO (nishant): Uncomment after the next release of mkl-dnn"
    // memory::format::dhwio,
    memory::format::oidhw,
    memory::format::OIdhw16i16o,
    memory::format::OIdhw16o16i,
    memory::format::Oidhw16o,
    memory::format::Odhwi16o,
    // memory::format::oIhw8i,             // These currently map to nChw8c and nChw16c
    // memory::format::oIhw16i,
    memory::format::OIhw8i8o,
    memory::format::OIhw16i16o,
    memory::format::IOhw16o16i,
    memory::format::OIhw8o8i,
    memory::format::OIhw16o16i,
    memory::format::Oihw8o,
    memory::format::Oihw16o,
    memory::format::Ohwi8o,
    memory::format::Ohwi16o,
    memory::format::OhIw16o4i};

bool runtime::cpu::mkldnn_utils::IsMKLDNNOp(ngraph::Node& op)
{
    return (s_op_registry.find(TI(op)) != s_op_registry.end());
}

mkldnn::memory::format runtime::cpu::mkldnn_utils::CreateNativeDataFormat(
    const ngraph::runtime::cpu::LayoutDescriptor& layout)
{
    return CreateNativeDataFormat(layout.get_shape());
}

mkldnn::memory::format runtime::cpu::mkldnn_utils::CreateNativeDataFormat(const Shape& shape)
{
    switch (shape.size())
    {
    case 1: return mkldnn::memory::format::x;
    case 2: return mkldnn::memory::format::nc;
    case 4: return mkldnn::memory::format::nchw;
    case 5: return mkldnn::memory::format::ncdhw;
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
    if (it == s_mkldnn_data_type_map.end())
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

const mkldnn::memory::desc& runtime::cpu::mkldnn_utils::get_input_mkldnn_md(const Node* node,
                                                                            size_t index)
{
    auto cpu_tvl = dynamic_pointer_cast<runtime::cpu::LayoutDescriptor>(
        node->get_inputs()[index].get_output().get_tensor_ptr()->get_tensor_layout());
    return cpu_tvl->get_mkldnn_md();
}

const mkldnn::memory::desc& runtime::cpu::mkldnn_utils::get_output_mkldnn_md(const Node* node,
                                                                             size_t index)
{
    auto tvl = node->get_output_tensor_ptr(index)->get_tensor_layout();
    return dynamic_cast<runtime::cpu::LayoutDescriptor&>(*tvl).get_mkldnn_md();
}

mkldnn::memory::desc runtime::cpu::mkldnn_utils::create_default_mkldnn_md(
    const Node* node,
    size_t index,
    bool output = false,
    mkldnn::memory::format format = mkldnn::memory::format::any)
{
    Shape shape;
    mkldnn::memory::data_type et;
    if (output)
    {
        shape = node->get_output_shape(index);
        et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(node->get_output_element_type(index));
    }
    else
    {
        shape = node->get_input_shape(index);
        et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(node->get_input_element_type(index));
    }

    return memory::desc(memory::dims(shape.begin(), shape.end()), et, format);
}

bool runtime::cpu::mkldnn_utils::can_create_mkldnn_md(const Shape& dims,
                                                      const Strides& strides,
                                                      const ngraph::element::Type type)
{
    auto it = s_mkldnn_data_type_map.find(type);
    if (dims.size() == 0)
    {
        return false;
    }
    if (it == s_mkldnn_data_type_map.end() || it->second == mkldnn::memory::data_type::data_undef)
    {
        return false;
    }
    if (dims.size() > TENSOR_MAX_DIMS)
    {
        return false;
    }
    if (shape_size(dims) == 0)
    {
        return false;
    }
    return true;
}

bool runtime::cpu::mkldnn_utils::is_perm_sorted(const Strides& a, const AxisVector& perm)
{
    for (size_t i = 0; i < a.size() - 1; i++)
    {
        if (a[perm[i]] < a[perm[i + 1]])
            return false;
    }
    return true;
}

mkldnn::memory::desc runtime::cpu::mkldnn_utils::create_blocked_mkldnn_md(
    const Shape& dims, const Strides& strides, const ngraph::element::Type type)
{
    memory::dims dim(dims.begin(), dims.end());
    memory::dims stride(strides.begin(), strides.end());
    memory::data_type dtype = get_mkldnn_data_type(type);

    if (dims.size() == 1)
    {
        return memory::desc(dim, dtype, memory::format::x);
    }
    if (dims.size() == 2)
    {
        if (is_perm_sorted(strides, {0, 1}))
        {
            return memory::desc(dim, dtype, memory::format::nc);
        }
    }

    if (dims.size() == 4)
    {
        if (is_perm_sorted(strides, {0, 1, 2, 3}))
        {
            return memory::desc(dim, dtype, memory::format::nchw);
        }
        if (is_perm_sorted(strides, {0, 2, 3, 1}))
        {
            return memory::desc(dim, dtype, memory::format::nhwc);
        }
    }

    if (dims.size() == 5)
    {
        if (is_perm_sorted(strides, {0, 1, 2, 3, 4}))
        {
            return memory::desc(dim, dtype, memory::format::ncdhw);
        }
        if (is_perm_sorted(strides, {0, 2, 3, 4, 1}))
        {
            return memory::desc(dim, dtype, memory::format::ndhwc);
        }
    }

    mkldnn_memory_desc_t md;
    md.primitive_kind = mkldnn_memory;
    md.ndims = static_cast<int>(dim.size());
    md.format = mkldnn_blocked;
    md.data_type = mkldnn::memory::convert_to_c(dtype);

    for (size_t i = 0; i < dim.size(); i++)
    {
        md.layout_desc.blocking.block_dims[i] = 1;
        md.layout_desc.blocking.strides[1][i] = 1;
        md.layout_desc.blocking.strides[0][i] = stride[i];
        md.layout_desc.blocking.padding_dims[i] = dim[i];
        md.layout_desc.blocking.offset_padding_to_data[i] = 0;
        md.dims[i] = dim[i];
    }
    md.layout_desc.blocking.offset_padding = 0;

    return memory::desc(md);
}

memory::desc runtime::cpu::mkldnn_utils::rotate_blocked_md(const memory::desc& in,
                                                           AxisVector& axis_order)
{
    mkldnn_memory_desc_t md;
    md.primitive_kind = in.data.primitive_kind;
    md.ndims = in.data.ndims;
    md.format = mkldnn_blocked;
    md.data_type = in.data.data_type;

    for (size_t i = 0; i < in.data.ndims; i++)
    {
        md.layout_desc.blocking.block_dims[i] =
            in.data.layout_desc.blocking.block_dims[axis_order[i]];
        md.layout_desc.blocking.strides[1][i] =
            in.data.layout_desc.blocking.strides[1][axis_order[i]];
        md.layout_desc.blocking.strides[0][i] =
            in.data.layout_desc.blocking.strides[0][axis_order[i]];
        md.layout_desc.blocking.padding_dims[i] =
            in.data.layout_desc.blocking.padding_dims[axis_order[i]];
        md.layout_desc.blocking.offset_padding_to_data[i] =
            in.data.layout_desc.blocking.offset_padding_to_data[axis_order[i]];
        md.dims[i] = in.data.dims[axis_order[i]];
    }
    md.layout_desc.blocking.offset_padding = in.data.layout_desc.blocking.offset_padding;

    auto out_md = memory::desc(md);

    auto get_named_md = [](const mkldnn_memory_desc_t& blk, const mkldnn_memory_format_t format) {
        mkldnn_memory_desc_t named_md;
        // Could throw an exception if named `format` is not compatible with `md.dims`
        error::wrap_c_api(
            mkldnn_memory_desc_init(&named_md, blk.ndims, blk.dims, blk.data_type, format), "");
        return memory::desc(named_md);
    };

    auto compare_named_md = [&](const mkldnn_memory_desc_t& blk,
                                const mkldnn_memory_format_t format,
                                const memory::desc& out) {
        try
        {
            auto named_md = get_named_md(blk, format);
            if (compare_mkldnn_mds(named_md, out))
            {
                return true;
            }
        }
        catch (const mkldnn::error&)
        {
            // Cannot create the named descriptor compatible with `in` desc
            return false;
        }
        return false;
    };

#define CANONICALIZE_MD(X)                                                                         \
    if (compare_named_md(md, X, out_md))                                                           \
        return get_named_md(md, X);
    switch (md.ndims)
    {
    case 1: CANONICALIZE_MD(mkldnn_x); break;
    case 2: CANONICALIZE_MD(mkldnn_nc); break;
    case 4:
        CANONICALIZE_MD(mkldnn_nchw);
        CANONICALIZE_MD(mkldnn_nhwc);
        CANONICALIZE_MD(mkldnn_nChw8c);
        CANONICALIZE_MD(mkldnn_nChw16c);
        break;
    case 5:
        CANONICALIZE_MD(mkldnn_ncdhw);
        CANONICALIZE_MD(mkldnn_ndhwc);
        CANONICALIZE_MD(mkldnn_nCdhw16c);
        break;
    default:;
    }
    return out_md;
}

bool runtime::cpu::mkldnn_utils::use_mkldnn_kernel(const ngraph::Node* node)
{
    auto op_annotations = static_cast<const ngraph::op::Op*>(node)->get_op_annotations();
    return (op_annotations &&
            static_pointer_cast<ngraph::runtime::cpu::CPUOpAnnotations>(op_annotations)
                ->is_mkldnn_op());
}

bool runtime::cpu::mkldnn_utils::compare_mkldnn_formats(mkldnn::memory::format lhs,
                                                        mkldnn::memory::format rhs)
{
    std::set<mkldnn::memory::format> similar_4d_formats{mkldnn::memory::format::nchw,
                                                        mkldnn::memory::format::oihw};
    if ((lhs == rhs) || (similar_4d_formats.find(lhs) != similar_4d_formats.end() &&
                         similar_4d_formats.find(rhs) != similar_4d_formats.end()))
    {
        return true;
    }
    return false;
}

bool runtime::cpu::mkldnn_utils::compare_mkldnn_mds(const mkldnn::memory::desc& lhs,
                                                    const mkldnn::memory::desc& rhs)
{
    return (memory::primitive_desc(lhs, runtime::cpu::mkldnn_utils::global_cpu_engine) ==
            memory::primitive_desc(rhs, runtime::cpu::mkldnn_utils::global_cpu_engine));
}

bool runtime::cpu::mkldnn_utils::is_mkldnn_filter_format(mkldnn::memory::format fmt)
{
    if (s_filter_formats.find(fmt) != s_filter_formats.end())
    {
        return true;
    }
    return false;
}

bool runtime::cpu::mkldnn_utils::is_mkldnn_blocked_data_format(mkldnn::memory::format fmt)
{
    if (fmt == memory::format::nChw8c || fmt == memory::format::nChw16c)
    {
        return true;
    }
    return false;
}
