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
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/type/element_type.hpp"

#include "mkldnn_utils.hpp"

using namespace mkldnn;
using namespace ngraph;
using namespace std;

#if defined(MKLDNN_VERSION_MAJOR) && defined(MKLDNN_VERSION_MINOR) && defined(MKLDNN_VERSION_PATCH)
/** Intel(R) MKL-DNN Version type */
/* typedef struct {
    int    major;
    int    minor;
    int    patch;
    const char *hash;
} mkldnn_version_t; */
static const mkldnn_version_t* get_mkldnn_version()
{
    return mkldnn_version();
}
#endif

#if MKLDNN_VERSION_MAJOR < 1
#define DATA_UNDEF data_undef
#else
#define DATA_UNDEF undef
#endif

// for both versions
const std::string&
    runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string(const ngraph::element::Type& type)
{
    auto it = get_mkldnn_data_type_string_map().find(type);
    if (it == get_mkldnn_data_type_string_map().end() || it->second.empty())
    {
        throw ngraph_error("No MKLDNN data type exists for the given element type" +
                           type.c_type_string());
    }
    return it->second;
}

mkldnn::memory::data_type
    runtime::cpu::mkldnn_utils::get_mkldnn_data_type(const ngraph::element::Type& type)
{
    auto it = get_mkldnn_data_type_map().find(type);
    if (it == get_mkldnn_data_type_map().end())
    {
        throw ngraph_error("No MKLDNN data type exists for the given element type" +
                           type.c_type_string());
    }
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

bool runtime::cpu::mkldnn_utils::can_create_mkldnn_md(const ngraph::element::Type type)
{
    auto it = get_mkldnn_data_type_map().find(type);
    if (it == get_mkldnn_data_type_map().end() ||
        it->second == mkldnn::memory::data_type::DATA_UNDEF)
    {
        return false;
    }
    return true;
}

bool runtime::cpu::mkldnn_utils::can_create_mkldnn_md(const Shape& dims,
                                                      const Strides& strides,
                                                      const ngraph::element::Type type)
{
    auto it = get_mkldnn_data_type_map().find(type);
    if (dims.size() == 0)
    {
        return false;
    }
    if (it == get_mkldnn_data_type_map().end() ||
        it->second == mkldnn::memory::data_type::DATA_UNDEF)
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
    if (dims.size() > TENSOR_MAX_DIMS || strides.size() > TENSOR_MAX_DIMS)
    {
        throw ngraph_error("In create_blocked_mkldnn_md: Dimensions (dims, stride): (" +
                           std::to_string(dims.size()) + ", " + std::to_string(strides.size()) +
                           ") exceed maximum supported by MKLDNN " +
                           std::to_string(TENSOR_MAX_DIMS));
    }

    if (dims.size() != strides.size())
    {
        throw ngraph_error("In create_blocked_mkldnn_md: Rank mismatch between shape and strides " +
                           std::to_string(dims.size()) + " " + std::to_string(strides.size()));
    }

    memory::dims dim(dims.begin(), dims.end());
    memory::dims stride(strides.begin(), strides.end());
    memory::data_type dtype = get_mkldnn_data_type(type);

    return create_blocked_mkldnn_md_helper(dim, strides, stride, dtype);
}

bool runtime::cpu::mkldnn_utils::is_mkldnn_filter_format(mkldnn::memory::FORMAT fmt)
{
    if (get_filter_formats().find(fmt) != get_filter_formats().end())
    {
        return true;
    }
    return false;
}

bool runtime::cpu::mkldnn_utils::is_mkldnn_blocked_data_format(mkldnn::memory::FORMAT fmt)
{
    if (fmt == memory::FORMAT::nChw8c || fmt == memory::FORMAT::nChw16c)
    {
        return true;
    }
    return false;
}

bool runtime::cpu::mkldnn_utils::use_mkldnn_kernel(const ngraph::Node* node)
{
    if (auto* op_node = dynamic_cast<const ngraph::op::Op*>(node))
    {
        auto op_annotations = op_node->get_op_annotations();
        return (op_annotations &&
                static_pointer_cast<ngraph::runtime::cpu::CPUOpAnnotations>(op_annotations)
                    ->is_mkldnn_op());
    }

    return false;
}

void runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(Node* node)
{
    auto ngraph_op = static_cast<ngraph::op::Op*>(node);
    auto op_annotations = std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
    op_annotations->set_mkldnn_op(true);
    ngraph_op->set_op_annotations(op_annotations);
}

bool runtime::cpu::mkldnn_utils::can_use_mkldnn_batchnorm_fprop(const ngraph::Node* node)
{
    auto input_rank = node->get_input_shape(2).size();
    auto input_element_type = node->get_input_element_type(2);

    if (((input_rank == 4 || input_rank == 5) && input_element_type == element::f32))
    {
        return true;
    }
    else
    {
        return false;
    }
}

mkldnn::algorithm runtime::cpu::mkldnn_utils::get_deconv_algo()
{
    // Note: there is no deconvolution_auto, so for now will return direct
    // TODO:
    return mkldnn::algorithm::deconvolution_direct;
}

mkldnn::algorithm runtime::cpu::mkldnn_utils::get_conv_algo()
{
#if defined(NGRAPH_ENABLE_CPU_CONV_AUTO) && defined(MKLDNN_VERSION_MAJOR) &&                       \
    defined(MKLDNN_VERSION_MINOR) && defined(MKLDNN_VERSION_PATCH)
    auto mkldnn_version = get_mkldnn_version();
    if ((mkldnn_version->major == 0 && mkldnn_version->minor >= 18 && mkldnn_version->patch >= 0) ||
        mkldnn_version->major >= 1)
    {
        return mkldnn::algorithm::convolution_auto;
    }
#endif
    return mkldnn::algorithm::convolution_direct;
}

bool runtime::cpu::mkldnn_utils::can_use_mkldnn_batchnorm_bprop(const ngraph::Node* node)
{
    auto input_rank = node->get_input_shape(2).size();
    auto input_element_type = node->get_input_element_type(2);
    auto delta_rank = node->get_input_shape(5).size();
    auto delta_element_type = node->get_input_element_type(5);

    if (((input_rank == 4 && delta_rank == 4) || (input_rank == 5 && delta_rank == 5)) &&
        (input_element_type == element::f32) && (delta_element_type == element::f32))
    {
        return true;
    }
    else
    {
        return false;
    }
}

#if MKLDNN_VERSION_MAJOR < 1
std::map<element::Type, const mkldnn::memory::data_type>&
    runtime::cpu::mkldnn_utils::get_mkldnn_data_type_map()
{
    // Mapping from POD types to MKLDNN data types
    static std::map<element::Type, const mkldnn::memory::data_type> s_mkldnn_data_type_map = {
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
        {element::u64, mkldnn::memory::data_type::data_undef},
    };
    return s_mkldnn_data_type_map;
}

std::map<element::Type, const std::string>&
    runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string_map()
{
    static std::map<element::Type, const std::string> s_mkldnn_data_type_string_map{
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
    return s_mkldnn_data_type_string_map;
}

std::map<memory::format, const std::string>&
    runtime::cpu::mkldnn_utils::get_mkldnn_format_string_map()
{
    static std::map<memory::format, const std::string> s_mkldnn_format_string_map{
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
        {memory::format::ncdhw, "memory::format::ncdhw"},
        {memory::format::ndhwc, "memory::format::ndhwc"},
        {memory::format::nCdhw8c, "memory::format::nCdhw8c"},
        {memory::format::nCdhw16c, "memory::format::nCdhw16c"},
        {memory::format::oi, "memory::format::oi"},
        {memory::format::io, "memory::format::io"},
        {memory::format::oihw, "memory::format::oihw"},
        {memory::format::ihwo, "memory::format::ihwo"},
        {memory::format::hwio, "memory::format::hwio"},
        {memory::format::dhwio, "memory::format::dhwio"},
        {memory::format::oidhw, "memory::format::oidhw"},
        {memory::format::OIdhw8i8o, "memory::format::OIdhw8i8o"},
        {memory::format::OIdhw8o8i, "memory::format::OIdhw8o8i"},
        {memory::format::Odhwi8o, "memory::format::Odhwi8o"},
        {memory::format::OIdhw16i16o, "memory::format::OIdhw16i16o"},
        {memory::format::OIdhw16o16i, "memory::format::OIdhw16o16i"},
        {memory::format::Oidhw16o, "memory::format::Oidhw16o"},
        {memory::format::Odhwi16o, "memory::format::Odhwi16o"},
        {memory::format::oIhw8i, "memory::format::oIhw8i"},
        {memory::format::oIhw16i, "memory::format::oIhw16i"},
        {memory::format::oIdhw8i, "memory::format::oIdhw8i"},
        {memory::format::oIdhw16i, "memory::format::oIdhw16i"},
        {memory::format::OIhw8i8o, "memory::format::OIhw8i8o"},
        {memory::format::OIhw16i16o, "memory::format::OIhw16i16o"},
        {memory::format::OIhw8o8i, "memory::format::OIhw8o8i"},
        {memory::format::OIhw16o16i, "memory::format::OIhw16o16i"},
        {memory::format::IOhw16o16i, "memory::format::IOhw16o16i"},
        {memory::format::OIhw8i16o2i, "memory::format::OIhw8i16o2i"},
        {memory::format::OIdhw8i16o2i, "memory::format::OIdhw8i16o2i"},
        {memory::format::OIhw8o16i2o, "memory::format::OIhw8o16i2o"},
        {memory::format::OIhw4i16o4i, "memory::format::OIhw4i16o4i"},
        {memory::format::Oihw8o, "memory::format::Oihw8o"},
        {memory::format::Oihw16o, "memory::format::Oihw16o"},
        {memory::format::Ohwi8o, "memory::format::Ohwi8o"},
        {memory::format::Ohwi16o, "memory::format::Ohwi16o"},
        {memory::format::OhIw16o4i, "memory::format::OhIw16o4i"},
        {memory::format::goihw, "memory::format::goihw"},
        {memory::format::hwigo, "memory::format::hwigo"},
        {memory::format::gOIdhw8i8o, "memory::format::gOIdhw8i8o"},
        {memory::format::gOIdhw8o8i, "memory::format::gOIdhw8o8i"},
        {memory::format::gOdhwi8o, "memory::format::gOdhwi8o"},
        {memory::format::gOIhw8i8o, "memory::format::gOIhw8i8o"},
        {memory::format::gOIhw16i16o, "memory::format::gOIhw16i16o"},
        {memory::format::gOIhw8i16o2i, "memory::format::gOIhw8i16o2i"},
        {memory::format::gOIdhw8i16o2i, "memory::format::gOIdhw8i16o2i"},
        {memory::format::gOIhw8o16i2o, "memory::format::gOIhw8o16i2o"},
        {memory::format::gOIhw4i16o4i, "memory::format::gOIhw4i16o4i"},
        {memory::format::gOihw8o, "memory::format::gOihw8o"},
        {memory::format::gOihw16o, "memory::format::gOihw16o"},
        {memory::format::gOhwi8o, "memory::format::gOhwi8o"},
        {memory::format::gOhwi16o, "memory::format::gOhwi16o"},
        {memory::format::Goihw8g, "memory::format::Goihw8g"},
        {memory::format::Goihw16g, "memory::format::Goihw16g"},
        {memory::format::gOIhw8o8i, "memory::format::gOIhw8o8i"},
        {memory::format::gOIhw16o16i, "memory::format::gOIhw16o16i"},
        {memory::format::gIOhw16o16i, "memory::format::gIOhw16o16i"},
        {memory::format::gOhIw16o4i, "memory::format::gOhIw16o4i"},
        {memory::format::goidhw, "memory::format::goidhw"},
        {memory::format::gOIdhw16i16o, "memory::format::gOIdhw16i16o"},
        {memory::format::gOIdhw16o16i, "memory::format::gOIdhw16o16i"},
        {memory::format::gOidhw16o, "memory::format::gOidhw16o"},
        {memory::format::gOdhwi16o, "memory::format::gOdhwi16o"},
        {memory::format::ntc, "memory::format::ntc"},
        {memory::format::tnc, "memory::format::tnc"},
        {memory::format::ldsnc, "memory::format::ldsnc"},
        {memory::format::ldigo, "memory::format::ldigo"},
        {memory::format::ldgoi, "memory::format::ldgoi"},
        {memory::format::ldgo, "memory::format::ldgo"},
        {memory::format::wino_fmt, "memory::format::wino_fmt"},
        {memory::format::format_last, "memory::format::format_last"},
    };
    return s_mkldnn_format_string_map;
}

std::set<memory::format>& runtime::cpu::mkldnn_utils::get_filter_formats()
{
    static std::set<memory::format> s_filter_formats{
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
    return s_filter_formats;
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

const std::string& runtime::cpu::mkldnn_utils::get_mkldnn_format_string(memory::format fmt)
{
    auto it = get_mkldnn_format_string_map().find(fmt);
    if (it == get_mkldnn_format_string_map().end())
        throw ngraph_error("No MKLDNN format exists for the given format type " +
                           std::to_string(fmt));
    return it->second;
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

    if (shape == Shape{})
    {
        shape = Shape{1};
    }
    return memory::desc(memory::dims(shape.begin(), shape.end()), et, format);
}

// MKLDNN kernel selection sometimes relies on named layouts like "mkldnn_nchw"
// Try and convert a blocked layout into a named layout
memory::desc runtime::cpu::mkldnn_utils::try_get_named_md(const mkldnn_memory_desc_t& md)
{
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
    case 3:
        CANONICALIZE_MD(mkldnn_tnc);
        CANONICALIZE_MD(mkldnn_ntc);
        break;
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

memory::desc runtime::cpu::mkldnn_utils::rotate_blocked_md(const memory::desc& in,
                                                           const AxisVector& axis_order)
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

    return try_get_named_md(md);
}

memory::desc runtime::cpu::mkldnn_utils::squeeze_blocked_md(const memory::desc& in,
                                                            AxisVector& axis_list)
{
    if (in.data.ndims <= axis_list.size())
    {
        throw ngraph_error("Squeezing too many axes: input " + to_string(in.data.ndims) +
                           " , removing " + to_string(axis_list.size()));
    }
    for (auto axis : axis_list)
    {
        if (in.data.dims[axis] != 1)
        {
            throw ngraph_error("Cannot squeeze axis on non unit-size, axis: " + to_string(axis) +
                               " size: " + to_string(in.data.dims[axis]));
        }
    }

    mkldnn_memory_desc_t md;
    md.primitive_kind = in.data.primitive_kind;
    md.ndims = in.data.ndims - static_cast<int>(axis_list.size());
    md.format = mkldnn_blocked;
    md.data_type = in.data.data_type;

    size_t k = 0;
    for (size_t i = 0, j = 0; i < in.data.ndims; i++)
    {
        if (k < axis_list.size() && i == axis_list[k])
        {
            k++;
            continue;
        }

        md.layout_desc.blocking.block_dims[j] = in.data.layout_desc.blocking.block_dims[i];
        md.layout_desc.blocking.strides[1][j] = in.data.layout_desc.blocking.strides[1][i];
        md.layout_desc.blocking.strides[0][j] = in.data.layout_desc.blocking.strides[0][i];
        md.layout_desc.blocking.padding_dims[j] = in.data.layout_desc.blocking.padding_dims[i];
        md.layout_desc.blocking.offset_padding_to_data[j] =

            in.data.layout_desc.blocking.offset_padding_to_data[i];
        md.dims[j] = in.data.dims[i];
        j++;
    }
    md.layout_desc.blocking.offset_padding = in.data.layout_desc.blocking.offset_padding;

    return try_get_named_md(md);
}

memory::desc runtime::cpu::mkldnn_utils::expand_blocked_md(const memory::desc& in,
                                                           AxisVector& axis_list)
{
    mkldnn_memory_desc_t md;
    md.primitive_kind = in.data.primitive_kind;
    md.ndims = in.data.ndims + static_cast<int>(axis_list.size());
    md.format = mkldnn_blocked;
    md.data_type = in.data.data_type;

    size_t k = 0;
    for (size_t i = 0, j = 0; j < md.ndims; j++)
    {
        if (k < axis_list.size() && j == axis_list[k])
        {
            k++;
            md.dims[j] = 1;
            md.layout_desc.blocking.block_dims[j] = 1;
            md.layout_desc.blocking.padding_dims[j] = 1;
            md.layout_desc.blocking.offset_padding_to_data[j] = 0;
            if (i > 0)
            {
                md.layout_desc.blocking.strides[1][j] =
                    in.data.layout_desc.blocking.strides[1][i - 1];
                md.layout_desc.blocking.strides[0][j] =
                    in.data.layout_desc.blocking.strides[0][i - 1];
            }
            else
            {
                md.layout_desc.blocking.strides[1][j] =
                    in.data.layout_desc.blocking.strides[0][in.data.ndims - 1];
                size_t nelems = 1;
                for (size_t idx = 0; idx < in.data.ndims; idx++)
                    nelems *= in.data.layout_desc.blocking.padding_dims[idx];
                md.layout_desc.blocking.strides[0][j] = nelems;
            }
        }
        else
        {
            md.dims[j] = in.data.dims[i];
            md.layout_desc.blocking.strides[1][j] = in.data.layout_desc.blocking.strides[1][i];
            md.layout_desc.blocking.strides[0][j] = in.data.layout_desc.blocking.strides[0][i];
            md.layout_desc.blocking.block_dims[j] = in.data.layout_desc.blocking.block_dims[i];
            md.layout_desc.blocking.padding_dims[j] = in.data.layout_desc.blocking.padding_dims[i];
            md.layout_desc.blocking.offset_padding_to_data[j] =
                in.data.layout_desc.blocking.offset_padding_to_data[i];
            i++;
        }
    }
    md.layout_desc.blocking.offset_padding = in.data.layout_desc.blocking.offset_padding;

    return try_get_named_md(md);
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
    return (memory::primitive_desc(lhs, runtime::cpu::executor::global_cpu_engine) ==
            memory::primitive_desc(rhs, runtime::cpu::executor::global_cpu_engine));
}

bool runtime::cpu::mkldnn_utils::is_mkldnn_padded_layout(const mkldnn::memory::desc& in,
                                                         const AxisVector& axis_list)
{
    for (size_t i = 0; i < in.data.ndims; i++)
    {
        if (std::find(axis_list.begin(), axis_list.end(), i) == axis_list.end())
        {
            continue;
        }
        if (in.data.layout_desc.blocking.padding_dims[i] != in.data.dims[i])
        {
            return true;
        }
        if (in.data.layout_desc.blocking.offset_padding_to_data[i] != 0)
        {
            return true;
        }
    }

    if (in.data.layout_desc.blocking.offset_padding != 0)
    {
        return true;
    }

    return false;
}

mkldnn::memory::desc runtime::cpu::mkldnn_utils::create_blocked_mkldnn_md_helper(
    const mkldnn::memory::dims& dim,
    const Strides& strides,
    const mkldnn::memory::dims& stride,
    const mkldnn::memory::data_type dtype)
{
    if (dim.size() == 1)
    {
        return memory::desc(dim, dtype, memory::format::x);
    }
    if (dim.size() == 2)
    {
        if (is_perm_sorted(strides, {0, 1}))
        {
            return memory::desc(dim, dtype, memory::format::nc);
        }
    }
    if (dim.size() == 3)
    {
        if (is_perm_sorted(strides, {0, 1, 2}))
        {
            return memory::desc(dim, dtype, memory::format::tnc);
        }
        if (is_perm_sorted(strides, {1, 0, 2}))
        {
            return memory::desc(dim, dtype, memory::format::ntc);
        }
    }
    if (dim.size() == 4)
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
    if (dim.size() == 5)
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
#else
std::map<element::Type, const mkldnn::memory::data_type>&
    runtime::cpu::mkldnn_utils::get_mkldnn_data_type_map()
{
    // Mapping from POD types to MKLDNN data types
    static std::map<element::Type, const mkldnn::memory::data_type> s_mkldnn_data_type_map = {
        {element::boolean, mkldnn::memory::data_type::s8},
        {element::bf16, mkldnn::memory::data_type::bf16},
        {element::f16, mkldnn::memory::data_type::f16},
        {element::f32, mkldnn::memory::data_type::f32},
        {element::f64, mkldnn::memory::data_type::undef},
        {element::i8, mkldnn::memory::data_type::s8},
        {element::i16, mkldnn::memory::data_type::undef},
        {element::i32, mkldnn::memory::data_type::s32},
        {element::i64, mkldnn::memory::data_type::undef},
        {element::u8, mkldnn::memory::data_type::u8},
        {element::u16, mkldnn::memory::data_type::undef},
        {element::u32, mkldnn::memory::data_type::undef},
        {element::u64, mkldnn::memory::data_type::undef},
    };
    return s_mkldnn_data_type_map;
}

std::map<element::Type, const std::string>&
    runtime::cpu::mkldnn_utils::get_mkldnn_data_type_string_map()
{
    static std::map<element::Type, const std::string> s_mkldnn_data_type_string_map{
        {element::boolean, "mkldnn::memory::data_type::s8"},
        {element::bf16, "mkldnn::memory::data_type::bf16"},
        {element::f16, "mkldnn::memory::data_type::f16"},
        {element::f32, "mkldnn::memory::data_type::f32"},
        {element::f64, "mkldnn::memory::data_type::undef"},
        {element::i8, "mkldnn::memory::data_type::s8"},
        {element::i16, "mkldnn::memory::data_type::undef"},
        {element::i32, "mkldnn::memory::data_type::s32"},
        {element::i64, "mkldnn::memory::data_type::undef"},
        {element::u8, "mkldnn::memory::data_type::u8"},
        {element::u16, "mkldnn::memory::data_type::undef"},
        {element::u32, "mkldnn::memory::data_type::undef"},
        {element::u64, "mkldnn::memory::data_type::undef"}};
    return s_mkldnn_data_type_string_map;
}

std::map<memory::format_kind, const std::string>&
    runtime::cpu::mkldnn_utils::get_mkldnn_format_kind_string_map()
{
    static std::map<memory::format_kind, const std::string> s_mkldnn_format_kind_string_map{
        {memory::format_kind::undef, "memory::format_kind::undef"},
        {memory::format_kind::any, "memory::format_kind::any"},
        {memory::format_kind::blocked, "memory::format_kind::blocked"},
        {memory::format_kind::wino, "memory::format_kind::wino"},
        {memory::format_kind::packed, "memory::format_kind::packed"},
    };
    return s_mkldnn_format_kind_string_map;
}

std::map<memory::format_tag, const std::string>&
    runtime::cpu::mkldnn_utils::get_mkldnn_format_string_map()
{
    static std::map<memory::format_tag, const std::string> s_mkldnn_format_tag_string_map{
        {memory::format_tag::undef, "memory::format_tag::undef"},
        {memory::format_tag::any, "memory::format_tag::any"},
        // Plain formats
        {memory::format_tag::a, "memory::format_tag::a"},
        {memory::format_tag::ab, "memory::format_tag::ab"},
        {memory::format_tag::abc, "memory::format_tag::abc"},
        {memory::format_tag::abcd, "memory::format_tag::abcd"},
        {memory::format_tag::abcde, "memory::format_tag::abcde"},
        {memory::format_tag::abcdef, "memory::format_tag::abcdef"},
        // Permuted plain formats
        {memory::format_tag::abdec, "memory::format_tag::abdec"},
        {memory::format_tag::acb, "memory::format_tag::acb"},
        {memory::format_tag::acbde, "memory::format_tag::acbde"},
        {memory::format_tag::acdb, "memory::format_tag::acdb"},
        {memory::format_tag::acdeb, "memory::format_tag::acdeb"},
        {memory::format_tag::ba, "memory::format_tag::ba"},
        {memory::format_tag::bac, "memory::format_tag::bac"},
        {memory::format_tag::bacd, "memory::format_tag::bacd"},
        //{memory::format_tag::bca, "memory::format_tag::bca"},
        {memory::format_tag::bcda, "memory::format_tag::bcda"},
        //{memory::format_tag::bcdea, "memory::format_tag::bcdea"},
        {memory::format_tag::cba, "memory::format_tag::cba"},
        {memory::format_tag::cdba, "memory::format_tag::cdba"},
        {memory::format_tag::cdeba, "memory::format_tag::cdeba"},
        {memory::format_tag::decab, "memory::format_tag::decab"},
        // Opaque blocked formats
        {memory::format_tag::Abc16a, "memory::format_tag::Abc16a"},
        {memory::format_tag::ABc16a16b, "memory::format_tag::ABc16a16b"},
        {memory::format_tag::aBc16b, "memory::format_tag::aBc16b"},
        {memory::format_tag::ABc16b16a, "memory::format_tag::ABc16b16a"},
        {memory::format_tag::Abc4a, "memory::format_tag::Abc4a"},
        {memory::format_tag::aBc4b, "memory::format_tag::aBc4b"},
        {memory::format_tag::ABc4b16a4b, "memory::format_tag::ABc4b16a4b"},
        {memory::format_tag::ABc4b4a, "memory::format_tag::ABc4b4a"},
        {memory::format_tag::ABc8a16b2a, "memory::format_tag::ABc8a16b2a"},
        {memory::format_tag::ABc8a8b, "memory::format_tag::ABc8a8b"},
        {memory::format_tag::aBc8b, "memory::format_tag::aBc8b"},
        {memory::format_tag::ABc8b16a2b, "memory::format_tag::ABc8b16a2b"},
        //{memory::format_tag::BAc8a16b2a, "memory::format_tag::BAc8a16b2a"},
        {memory::format_tag::ABc8b8a, "memory::format_tag::ABc8b8a"},
        {memory::format_tag::Abcd16a, "memory::format_tag::Abcd16a"},
        {memory::format_tag::ABcd16a16b, "memory::format_tag::ABcd16a16b"},
        {memory::format_tag::ABcd32a32b, "memory::format_tag::ABcd32a32b"},
        {memory::format_tag::aBcd16b, "memory::format_tag::aBcd16b"},
        {memory::format_tag::ABcd16b16a, "memory::format_tag::ABcd16b16a"},
        {memory::format_tag::aBCd16b16c, "memory::format_tag::aBCd16b16c"},
        {memory::format_tag::aBCd16c16b, "memory::format_tag::aBCd16c16b"},
        {memory::format_tag::Abcd4a, "memory::format_tag::Abcd4a"},
        {memory::format_tag::aBcd4b, "memory::format_tag::aBcd4b"},
        {memory::format_tag::ABcd4b16a4b, "memory::format_tag::ABcd4b16a4b"},
        {memory::format_tag::ABcd4b4a, "memory::format_tag::ABcd4b4a"},
        {memory::format_tag::aBCd4c16b4c, "memory::format_tag::aBCd4c16b4c"},
        {memory::format_tag::aBCd4c4b, "memory::format_tag::aBCd4c4b"},
        {memory::format_tag::ABcd8a16b2a, "memory::format_tag::ABcd8a16b2a"},
        {memory::format_tag::ABcd8a8b, "memory::format_tag::ABcd8a8b"},
        {memory::format_tag::aBcd8b, "memory::format_tag::aBcd8b"},
        {memory::format_tag::ABcd8b16a2b, "memory::format_tag::ABcd8b16a2b"},
        {memory::format_tag::aBCd8b16c2b, "memory::format_tag::aBCd8b16c2b"},
        //{memory::format_tag::BAcd8a16b2a, "memory::format_tag::BAcd8a16b2a"},
        {memory::format_tag::ABcd8b8a, "memory::format_tag::ABcd8b8a"},
        {memory::format_tag::aBCd8b8c, "memory::format_tag::aBCd8b8c"},
        {memory::format_tag::aBCd8c16b2c, "memory::format_tag::aBCd8c16b2c"},
        //{memory::format_tag::ABcde8a16b2a, "memory::format_tag::ABcde8a16b2a"},
        //{memory::format_tag::aCBd8b16c2b, "memory::format_tag::aCBd8b16c2b"},
        {memory::format_tag::aBCd8c8b, "memory::format_tag::aBCd8c8b"},
        {memory::format_tag::Abcde16a, "memory::format_tag::Abcde16a"},
        {memory::format_tag::ABcde16a16b, "memory::format_tag::ABcde16a16b"},
        //{memory::format_tag::BAcde8a16b2a, "memory::format_tag::BAcde8a16b2a"},
        {memory::format_tag::aBcde16b, "memory::format_tag::aBcde16b"},
        {memory::format_tag::ABcde16b16a, "memory::format_tag::ABcde16b16a"},
        {memory::format_tag::aBCde16b16c, "memory::format_tag::aBCde16b16c"},
        {memory::format_tag::aBCde16c16b, "memory::format_tag::aBCde16c16b"},
        {memory::format_tag::aBCde16c16b, "memory::format_tag::aBCde16c16b"},
        {memory::format_tag::Abcde4a, "memory::format_tag::Abcde4a"},
        {memory::format_tag::aBcde4b, "memory::format_tag::aBcde4b"},
        {memory::format_tag::ABcde4b4a, "memory::format_tag::ABcde4b4a"},
        {memory::format_tag::aBCde4b4c, "memory::format_tag::aBCde4b4c"},
        {memory::format_tag::aBCde4c16b4c, "memory::format_tag::aBCde4c16b4c"},
        {memory::format_tag::aBCde4c4b, "memory::format_tag::aBCde4c4b"},
        {memory::format_tag::Abcde8a, "memory::format_tag::Abcde8a"},
        {memory::format_tag::ABcde8a8b, "memory::format_tag::ABcde8a8b"},
        //{memory::format_tag::BAcde16b16a, "memory::format_tag::BAcde16b16a"},
        {memory::format_tag::aBcde8b, "memory::format_tag::aBcde8b"},
        {memory::format_tag::ABcde8b16a2b, "memory::format_tag::ABcde8b16a2b"},
        {memory::format_tag::aBCde8b16c2b, "memory::format_tag::aBCde8b16c2b"},
        //{memory::format_tag::aCBde8b16c2b, "memory::format_tag::aCBde8b16c2b"},
        {memory::format_tag::ABcde8b8a, "memory::format_tag::ABcde8b8a"},
        {memory::format_tag::aBCde8b8c, "memory::format_tag::aBCde8b8c"},
        {memory::format_tag::ABcd4a8b8a4b, "memory::format_tag::ABcd4a8b8a4b"},
        {memory::format_tag::ABcd2a8b8a2b, "memory::format_tag::ABcd2a8b8a2b"},
        {memory::format_tag::aBCde4b8c8b4c, "memory::format_tag::aBCde4b8c8b4c"},
        {memory::format_tag::aBCde2b8c8b2c, "memory::format_tag::aBCde2b8c8b2c"},
        {memory::format_tag::aBCde8c16b2c, "memory::format_tag::aBCde8c16b2c"},
        {memory::format_tag::aBCde8c8b, "memory::format_tag::aBCde8c8b"},
        {memory::format_tag::aBcdef16b, "memory::format_tag::aBcdef16b"},
        {memory::format_tag::aBCdef16b16c, "memory::format_tag::aBCdef16b16c"},
        {memory::format_tag::aBCdef16c16b, "memory::format_tag::aBCdef16c16b"},
        {memory::format_tag::aBcdef4b, "memory::format_tag::aBcdef4b"},
        {memory::format_tag::aBCdef4c4b, "memory::format_tag::aBCdef4c4b"},
        {memory::format_tag::aBCdef8b8c, "memory::format_tag::aBCdef8b8c"},
        {memory::format_tag::aBCdef8c16b2c, "memory::format_tag::aBCdef8c16b2c"},
        //{memory::format_tag::aBCdef8b16c2b, "memory::format_tag::aBCdef8b16c2b"},
        //{memory::format_tag::aCBdef8b16c2b, "memory::format_tag::aCBdef8b16c2b"},
        {memory::format_tag::aBCdef8c8b, "memory::format_tag::aBCdef8c8b"},
        {memory::format_tag::aBdc16b, "memory::format_tag::aBdc16b"},
        {memory::format_tag::aBdc4b, "memory::format_tag::aBdc4b"},
        {memory::format_tag::aBdc8b, "memory::format_tag::aBdc8b"},
        {memory::format_tag::aBdec16b, "memory::format_tag::aBdec16b"},
        {memory::format_tag::aBdec32b, "memory::format_tag::aBdec32b"},
        {memory::format_tag::aBdec4b, "memory::format_tag::aBdec4b"},
        {memory::format_tag::aBdec8b, "memory::format_tag::aBdec8b"},
        {memory::format_tag::aBdefc16b, "memory::format_tag::aBdefc16b"},
        {memory::format_tag::aCBdef16c16b, "memory::format_tag::aCBdef16c16b"},
        {memory::format_tag::aBdefc4b, "memory::format_tag::aBdefc4b"},
        {memory::format_tag::aBdefc8b, "memory::format_tag::aBdefc8b"},
        {memory::format_tag::Abcdef16a, "memory::format_tag::Abcdef16a"},
        {memory::format_tag::Acb16a, "memory::format_tag::Acb16a"},
        {memory::format_tag::Acb4a, "memory::format_tag::Acb4a"},
        {memory::format_tag::Acb8a, "memory::format_tag::Acb8a"},
        {memory::format_tag::aCBd16b16c, "memory::format_tag::aCBd16b16c"},
        {memory::format_tag::aCBd16c16b, "memory::format_tag::aCBd16c16b"},
        {memory::format_tag::aCBde16b16c, "memory::format_tag::aCBde16b16c"},
        {memory::format_tag::aCBde16c16b, "memory::format_tag::aCBde16c16b"},
        {memory::format_tag::Acdb16a, "memory::format_tag::Acdb16a"},
        {memory::format_tag::Acdb32a, "memory::format_tag::Acdb32a"},
        {memory::format_tag::Acdb4a, "memory::format_tag::Acdb4a"},
        {memory::format_tag::Acdb8a, "memory::format_tag::Acdb8a"},
        {memory::format_tag::Acdeb16a, "memory::format_tag::Acdeb16a"},
        {memory::format_tag::Acdeb4a, "memory::format_tag::Acdeb4a"},
        {memory::format_tag::Acdeb8a, "memory::format_tag::Acdeb8a"},
        {memory::format_tag::BAc16a16b, "memory::format_tag::BAc16a16b"},
        {memory::format_tag::BAc16b16a, "memory::format_tag::BAc16b16a"},
        {memory::format_tag::BAcd16a16b, "memory::format_tag::BAcd16a16b"},
        {memory::format_tag::BAcd16b16a, "memory::format_tag::BAcd16b16a"},
        //{memory::format_tag::last, "memory::format_tag::last"},
        // Aliases
        {memory::format_tag::x, "memory::format_tag::x"},
        {memory::format_tag::nc, "memory::format_tag::nc"},
        {memory::format_tag::cn, "memory::format_tag::cn"},
        {memory::format_tag::ncw, "memory::format_tag::ncw"},
        {memory::format_tag::nwc, "memory::format_tag::nwc"},
        {memory::format_tag::nchw, "memory::format_tag::nchw"},
        {memory::format_tag::nhwc, "memory::format_tag::nhwc"},
        {memory::format_tag::chwn, "memory::format_tag::chwn"},
        {memory::format_tag::ncdhw, "memory::format_tag::ncdhw"},
        {memory::format_tag::ndhwc, "memory::format_tag::ndhwc"},
        {memory::format_tag::oi, "memory::format_tag::oi"},
        {memory::format_tag::io, "memory::format_tag::io"},
        {memory::format_tag::oiw, "memory::format_tag::oiw"},
        //{memory::format_tag::owi, "memory::format_tag::owi"},
        {memory::format_tag::wio, "memory::format_tag::wio"},
        //{memory::format_tag::iwo, "memory::format_tag::iwo"},
        {memory::format_tag::oihw, "memory::format_tag::oihw"},
        {memory::format_tag::hwio, "memory::format_tag::hwio"},
        //{memory::format_tag::ohwi, "memory::format_tag::ohwi"},
        {memory::format_tag::ihwo, "memory::format_tag::ihwo"},
        {memory::format_tag::iohw, "memory::format_tag::iohw"},
        {memory::format_tag::oidhw, "memory::format_tag::oidhw"},
        {memory::format_tag::dhwio, "memory::format_tag::dhwio"},
        //{memory::format_tag::odhwi, "memory::format_tag::odhwi"},
        //{memory::format_tag::idhwo, "memory::format_tag::idhwo"},
        {memory::format_tag::goiw, "memory::format_tag::goiw"},
        {memory::format_tag::goihw, "memory::format_tag::goihw"},
        {memory::format_tag::hwigo, "memory::format_tag::hwigo"},
        {memory::format_tag::giohw, "memory::format_tag::giohw"},
        {memory::format_tag::goidhw, "memory::format_tag::goidhw"},
        {memory::format_tag::tnc, "memory::format_tag::tnc"},
        {memory::format_tag::ntc, "memory::format_tag::ntc"},
        {memory::format_tag::ldnc, "memory::format_tag::ldnc"},
        {memory::format_tag::ldigo, "memory::format_tag::ldigo"},
        {memory::format_tag::ldgoi, "memory::format_tag::ldgoi"},
        {memory::format_tag::ldgo, "memory::format_tag::ldgo"},
        {memory::format_tag::nCdhw16c, "memory::format_tag::nCdhw16c"},
        {memory::format_tag::nCdhw4c, "memory::format_tag::nCdhw4c"},
        {memory::format_tag::nCdhw8c, "memory::format_tag::nCdhw8c"},
        {memory::format_tag::nChw16c, "memory::format_tag::nChw16c"},
        {memory::format_tag::nChw4c, "memory::format_tag::nChw4c"},
        {memory::format_tag::nChw8c, "memory::format_tag::nChw8c"},
        {memory::format_tag::nCw16c, "memory::format_tag::nCw16c"},
        {memory::format_tag::nCw4c, "memory::format_tag::nCw4c"},
        {memory::format_tag::nCw8c, "memory::format_tag::nCw8c"},
        {memory::format_tag::NCw16n16c, "memory::format_tag::NCw16n16c"},
        {memory::format_tag::NCdhw16n16c, "memory::format_tag::NCdhw16n16c"},
        {memory::format_tag::NChw16n16c, "memory::format_tag::NChw16n16c"},
        {memory::format_tag::NChw32n32c, "memory::format_tag::NChw32n32c"},
        {memory::format_tag::IOw16o16i, "memory::format_tag::IOw16o16i"},
        {memory::format_tag::IOw16i16o, "memory::format_tag::IOw16i16o"},
        {memory::format_tag::OIw16i16o, "memory::format_tag::OIw16i16o"},
        {memory::format_tag::OIw16i16o, "memory::format_tag::OIw16i16o"},
        {memory::format_tag::Oiw16o, "memory::format_tag::Oiw16o"},
        {memory::format_tag::OIw4i16o4i, "memory::format_tag::OIw4i16o4i"},
        {memory::format_tag::OIw4i4o, "memory::format_tag::OIw4i4o"},
        {memory::format_tag::Oiw4o, "memory::format_tag::Oiw4o"},
        {memory::format_tag::OIw8i16o2i, "memory::format_tag::OIw8i16o2i"},
        {memory::format_tag::OIw8i8o, "memory::format_tag::OIw8i8o"},
        {memory::format_tag::OIw8o16i2o, "memory::format_tag::OIw8o16i2o"},
        //{memory::format_tag::IOw8o16i2o, "memory::format_tag::IOw8o16i2o"},
        {memory::format_tag::OIw8o8i, "memory::format_tag::OIw8o8i"},
        {memory::format_tag::Owi16o, "memory::format_tag::Owi16o"},
        {memory::format_tag::Owi4o, "memory::format_tag::Owi4o"},
        {memory::format_tag::Owi8o, "memory::format_tag::Owi8o"},
        {memory::format_tag::IOhw16i16o, "memory::format_tag::IOhw16i16o"},
        {memory::format_tag::IOhw16o16i, "memory::format_tag::IOhw16o16i"},
        {memory::format_tag::Ohwi16o, "memory::format_tag::Ohwi16o"},
        {memory::format_tag::Ohwi32o, "memory::format_tag::Ohwi32o"},
        {memory::format_tag::Ohwi4o, "memory::format_tag::Ohwi4o"},
        {memory::format_tag::Ohwi8o, "memory::format_tag::Ohwi8o"},
        {memory::format_tag::OIhw16i16o, "memory::format_tag::OIhw16i16o"},
        {memory::format_tag::OIhw16o16i, "memory::format_tag::OIhw16o16i"},
        {memory::format_tag::Oihw16o, "memory::format_tag::Oihw16o"},
        {memory::format_tag::OIhw4i16o4i, "memory::format_tag::OIhw4i16o4i"},
        {memory::format_tag::OIhw4i4o, "memory::format_tag::OIhw4i4o"},
        {memory::format_tag::Oihw4o, "memory::format_tag::Oihw4o"},
        {memory::format_tag::OIhw8i16o2i, "memory::format_tag::OIhw8i16o2i"},
        {memory::format_tag::OIhw8i8o, "memory::format_tag::OIhw8i8o"},
        //{memory::format_tag::OIhw8o16i2o, "memory::format_tag::OIhw8o16i2o"},
        //{memory::format_tag::IOhw8o16i2o, "memory::format_tag::IOhw8o16i2o"},
        {memory::format_tag::OIhw8o8i, "memory::format_tag::OIhw8o8i"},
        {memory::format_tag::Odhwi16o, "memory::format_tag::Odhwi16o"},
        {memory::format_tag::Odhwi4o, "memory::format_tag::Odhwi4o"},
        {memory::format_tag::Odhwi8o, "memory::format_tag::Odhwi8o"},
        {memory::format_tag::OIdhw16i16o, "memory::format_tag::OIdhw16i16o"},
        {memory::format_tag::OIdhw16o16i, "memory::format_tag::OIdhw16o16i"},
        {memory::format_tag::Oidhw16o, "memory::format_tag::Oidhw16o"},
        {memory::format_tag::OIdhw4i4o, "memory::format_tag::OIdhw4i4o"},
        {memory::format_tag::Oidhw4o, "memory::format_tag::Oidhw4o"},
        {memory::format_tag::OIdhw8i16o2i, "memory::format_tag::OIdhw8i16o2i"},
        {memory::format_tag::OIdhw8i8o, "memory::format_tag::OIdhw8i8o"},
        //{memory::format_tag::OIdhw8o16i2o, "memory::format_tag::OIdhw8o16i2o"},
        //{memory::format_tag::IOdhw8o16i2o, "memory::format_tag::IOdhw8o16i2o"},
        {memory::format_tag::OIdhw8o8i, "memory::format_tag::OIdhw8o8i"},
        {memory::format_tag::IOdhw16i16o, "memory::format_tag::IOdhw16i16o"},
        //{memory::format_tag::Goiw16g, "memory::format_tag::Goiw16g"},
        {memory::format_tag::gIOw16o16i, "memory::format_tag::gIOw16o16i"},
        {memory::format_tag::gIOw16i16o, "memory::format_tag::gIOw16i16o"},
        {memory::format_tag::gOIw16i16o, "memory::format_tag::gOIw16i16o"},
        {memory::format_tag::gOIw16o16i, "memory::format_tag::gOIw16o16i"},
        {memory::format_tag::gOiw16o, "memory::format_tag::gOiw16o"},
        {memory::format_tag::gOIw4i16o4i, "memory::format_tag::gOIw4i16o4i"},
        {memory::format_tag::gOIw4i4o, "memory::format_tag::gOIw4i4o"},
        {memory::format_tag::gOiw4o, "memory::format_tag::gOiw4o"},
        {memory::format_tag::gOIw8i16o2i, "memory::format_tag::gOIw8i16o2i"},
        {memory::format_tag::gOIw8i8o, "memory::format_tag::gOIw8i8o"},
        {memory::format_tag::gOIw8o16i2o, "memory::format_tag::gOIw8o16i2o"},
        //{memory::format_tag::gIOw8o16i2o, "memory::format_tag::gIOw8o16i2o"},
        {memory::format_tag::gOIw8o8i, "memory::format_tag::gOIw8o8i"},
        {memory::format_tag::gOwi16o, "memory::format_tag::gOwi16o"},
        {memory::format_tag::gOwi4o, "memory::format_tag::gOwi4o"},
        {memory::format_tag::gOwi8o, "memory::format_tag::gOwi8o"},
        {memory::format_tag::gIOhw16i16o, "memory::format_tag::gIOhw16i16o"},
        {memory::format_tag::gIOhw16o16i, "memory::format_tag::gIOhw16o16i"},
        {memory::format_tag::gOhwi16o, "memory::format_tag::gOhwi16o"},
        {memory::format_tag::gOhwi32o, "memory::format_tag::gOhwi32o"},
        {memory::format_tag::gOhwi4o, "memory::format_tag::gOhwi4o"},
        {memory::format_tag::gOhwi8o, "memory::format_tag::gOhwi8o"},
        {memory::format_tag::Goihw16g, "memory::format_tag::Goihw16g"},
        {memory::format_tag::gOIhw16i16o, "memory::format_tag::gOIhw16i16o"},
        {memory::format_tag::gOIhw16o16i, "memory::format_tag::gOIhw16o16i"},
        {memory::format_tag::gOihw16o, "memory::format_tag::gOihw16o"},
        {memory::format_tag::gOIhw2i8o4i, "memory::format_tag::gOIhw2i8o4i"},
        {memory::format_tag::gOIhw4i16o4i, "memory::format_tag::gOIhw4i16o4i"},
        {memory::format_tag::gOIhw4i4o, "memory::format_tag::gOIhw4i4o"},
        {memory::format_tag::gOIhw4o4i, "memory::format_tag::gOIhw4o4i"},
        {memory::format_tag::gOihw4o, "memory::format_tag::gOihw4o"},
        {memory::format_tag::Goihw8g, "memory::format_tag::Goihw8g"},
        {memory::format_tag::gOIhw8i16o2i, "memory::format_tag::gOIhw8i16o2i"},
        {memory::format_tag::gOIhw8i8o, "memory::format_tag::gOIhw8i8o"},
        {memory::format_tag::gOIhw8o16i2o, "memory::format_tag::gOIhw8o16i2o"},
        //{memory::format_tag::gIOhw8o16i2o, "memory::format_tag::gIOhw8o16i2o"},
        {memory::format_tag::gOIhw8o8i, "memory::format_tag::gOIhw8o8i"},
        {memory::format_tag::OIhw4o8i8o4i, "memory::format_tag::OIhw4o8i8o4i"},
        {memory::format_tag::OIhw2o8i8o2i, "memory::format_tag::OIhw2o8i8o2i"},
        {memory::format_tag::gOIhw4o8i8o4i, "memory::format_tag::gOIhw4o8i8o4i"},
        {memory::format_tag::gOIhw2o8i8o2i, "memory::format_tag::gOIhw2o8i8o2i"},
        {memory::format_tag::gIOdhw16i16o, "memory::format_tag::gIOdhw16i16o"},
        {memory::format_tag::gOdhwi16o, "memory::format_tag::gOdhwi16o"},
        {memory::format_tag::gOdhwi4o, "memory::format_tag::gOdhwi4o"},
        {memory::format_tag::gOdhwi8o, "memory::format_tag::gOdhwi8o"},
        {memory::format_tag::gOIdhw16i16o, "memory::format_tag::gOIdhw16i16o"},
        {memory::format_tag::gOIdhw16o16i, "memory::format_tag::gOIdhw16o16i"},
        {memory::format_tag::gOidhw16o, "memory::format_tag::gOidhw16o"},
        {memory::format_tag::gOIdhw4i4o, "memory::format_tag::gOIdhw4i4o"},
        {memory::format_tag::gOidhw4o, "memory::format_tag::gOidhw4o"},
        {memory::format_tag::gOIdhw8i16o2i, "memory::format_tag::gOIdhw8i16o2i"},
        {memory::format_tag::gOIdhw8i8o, "memory::format_tag::gOIdhw8i8o"},
        //{memory::format_tag::gOIdhw8o16i2o, "memory::format_tag::gOIdhw8o16i2o"},
        //{memory::format_tag::gIOdhw8o16i2o, "memory::format_tag::gIOdhw8o16i2o"},
        {memory::format_tag::gOIdhw8o8i, "memory::format_tag::gOIdhw8o8i"},
        {memory::format_tag::Goidhw16g, "memory::format_tag::Goidhw16g"},
    };
    return s_mkldnn_format_tag_string_map;
}

std::set<memory::format_tag>& runtime::cpu::mkldnn_utils::get_filter_formats()
{
    static std::set<memory::format_tag> s_filter_format_tags{
        memory::format_tag::oihw,
        memory::format_tag::ihwo,
        memory::format_tag::hwio,
        // TODO (nishant): Uncomment after the next release of mkl-dnn"
        // memory::format_tag::dhwio,
        memory::format_tag::oidhw,
        memory::format_tag::OIdhw16i16o,
        memory::format_tag::OIdhw16o16i,
        memory::format_tag::Oidhw16o,
        memory::format_tag::Odhwi16o,
        // memory::format_tag::oIhw8i,             // These currently map to nChw8c and nChw16c
        // memory::format_tag::oIhw16i,
        memory::format_tag::OIhw8i8o,
        memory::format_tag::OIhw16i16o,
        memory::format_tag::IOhw16o16i,
        memory::format_tag::OIhw8o8i,
        memory::format_tag::OIhw16o16i,
        // memory::format_tag::Oihw8o,
        memory::format_tag::Oihw16o,
        memory::format_tag::Ohwi8o,
        // memory::format_tag::OhIw16o4i,
        memory::format_tag::Ohwi16o};
    return s_filter_format_tags;
}

mkldnn::memory::format_tag runtime::cpu::mkldnn_utils::CreateNativeDataFormat(
    const ngraph::runtime::cpu::LayoutDescriptor& layout)
{
    return CreateNativeDataFormat(layout.get_shape());
}

mkldnn::memory::format_tag runtime::cpu::mkldnn_utils::CreateNativeDataFormat(const Shape& shape)
{
    switch (shape.size())
    {
    case 1: return mkldnn::memory::format_tag::x;
    case 2: return mkldnn::memory::format_tag::nc;
    case 4: return mkldnn::memory::format_tag::nchw;
    case 5: return mkldnn::memory::format_tag::ncdhw;
    default: return mkldnn::memory::format_tag::undef;
    }
}

const std::string& runtime::cpu::mkldnn_utils::get_mkldnn_format_string(memory::format_tag fmt)
{
    auto it = get_mkldnn_format_string_map().find(fmt);
    if (it == get_mkldnn_format_string_map().end())
        throw ngraph_error("No MKLDNN format_tag exists for the given format_tag type " /* +
                           std::to_string(fmt)*/);
    return it->second;
}

const std::string&
    runtime::cpu::mkldnn_utils::get_mkldnn_format_kind_string(memory::format_kind fmt_kind)
{
    auto it = get_mkldnn_format_kind_string_map().find(fmt_kind);
    if (it == get_mkldnn_format_kind_string_map().end())
        throw ngraph_error("No MKLDNN format_kind exists for the given format_kind type " /*+
                           std::to_string(fmt_kind)*/);
    return it->second;
}

mkldnn::memory::desc runtime::cpu::mkldnn_utils::create_default_mkldnn_md(
    const Node* node,
    size_t index,
    bool output = false,
    mkldnn::memory::format_tag format_tag = mkldnn::memory::format_tag::any)
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

    if (shape == Shape{})
    {
        shape = Shape{1};
    }
    return memory::desc(memory::dims(shape.begin(), shape.end()), et, format_tag);
}

mkldnn::memory::desc runtime::cpu::mkldnn_utils::create_default_mkldnn_md_with_strides(
    const Node* node, size_t index, mkldnn::memory::dims& strides, bool output = false)
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

    if (shape == Shape{})
    {
        shape = Shape{1};
    }
    return memory::desc(memory::dims(shape.begin(), shape.end()), et, strides);
}

mkldnn::memory::desc runtime::cpu::mkldnn_utils::create_blocked_mkldnn_md_helper(
    const mkldnn::memory::dims& dim,
    const Strides& strides,
    const mkldnn::memory::dims& stride,
    const mkldnn::memory::data_type dtype)
{
    return memory::desc(dim, dtype, stride);
}

// MKLDNN kernel selection sometimes relies on named layouts like "mkldnn_nchw"
// Try and convert a blocked layout into a named layout
memory::desc runtime::cpu::mkldnn_utils::try_get_named_md(const mkldnn_memory_desc_t& md)
{
    auto out_md = memory::desc(md);

    auto get_named_md = [](const mkldnn_memory_desc_t& blk, const mkldnn_format_tag_t format) {
        mkldnn_memory_desc_t named_md;
        // Could throw an exception if named `format` is not compatible with `md.dims`
        error::wrap_c_api(
            mkldnn_memory_desc_init_by_tag(&named_md, blk.ndims, blk.dims, blk.data_type, format),
            "");

        return memory::desc(named_md);
    };

    auto compare_named_md = [&](const mkldnn_memory_desc_t& blk,
                                const mkldnn_format_tag_t format,
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
    case 3:
        CANONICALIZE_MD(mkldnn_tnc);
        CANONICALIZE_MD(mkldnn_ntc);
        break;
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

memory::desc runtime::cpu::mkldnn_utils::rotate_blocked_md(const memory::desc& in,
                                                           const AxisVector& axis_order)
{
    mkldnn_memory_desc_t md;
    md.ndims = in.data.ndims;
    md.format_kind = mkldnn_blocked;
    md.data_type = in.data.data_type;
    md.format_desc.blocking.inner_nblks = in.data.format_desc.blocking.inner_nblks;

    for (size_t i = 0; i < in.data.ndims; i++)
    {
        md.padded_dims[i] = in.data.padded_dims[axis_order[i]];
        md.padded_offsets[i] = in.data.padded_offsets[axis_order[i]];
        md.dims[i] = in.data.dims[axis_order[i]];
        md.format_desc.blocking.strides[i] = in.data.format_desc.blocking.strides[axis_order[i]];
    }

    for (size_t i = 0; i < in.data.format_desc.blocking.inner_nblks; i++)
    {
        md.format_desc.blocking.inner_blks[i] = in.data.format_desc.blocking.inner_blks[i];
        md.format_desc.blocking.inner_idxs[i] =
            axis_order[in.data.format_desc.blocking.inner_idxs[i]];
    }

    md.offset0 = in.data.offset0;
    md.extra.flags = in.data.extra.flags;
    md.extra.scale_adjust = in.data.extra.scale_adjust;

    return try_get_named_md(md);
}

memory::desc runtime::cpu::mkldnn_utils::squeeze_blocked_md(const memory::desc& in,
                                                            AxisVector& axis_list)
{
    if (in.data.ndims <= axis_list.size())
    {
        throw ngraph_error("Squeezing too many axes: input " + to_string(in.data.ndims) +
                           " , removing " + to_string(axis_list.size()));
    }
    for (auto axis : axis_list)
    {
        if (in.data.dims[axis] != 1)
        {
            throw ngraph_error("Cannot squeeze axis on non unit-size, axis: " + to_string(axis) +
                               " size: " + to_string(in.data.dims[axis]));
        }
    }

    mkldnn_memory_desc_t md;
    md.ndims = in.data.ndims - static_cast<int>(axis_list.size());
    md.format_kind = mkldnn_blocked;
    md.data_type = in.data.data_type;
    md.format_desc.blocking.inner_nblks = in.data.format_desc.blocking.inner_nblks;

    size_t k = 0;
    for (size_t i = 0, j = 0; i < in.data.ndims; i++)
    {
        if (k < axis_list.size() && i == axis_list[k])
        {
            k++;
            continue;
        }

        md.format_desc.blocking.strides[j] = in.data.format_desc.blocking.strides[i];
        md.padded_dims[j] = in.data.padded_dims[i];
        md.padded_offsets[j] = in.data.padded_offsets[i];
        md.dims[j] = in.data.dims[i];

        j++;
    }

    std::sort(axis_list.begin(), axis_list.end());
    auto get_axis_after_squeeze = [&](size_t index) -> int {
        if (std::find(axis_list.begin(), axis_list.end(), index) != axis_list.end())
        {
            return -1;
        }
        for (auto axis : axis_list)
        {
            if (axis < index)
            {
                index--;
            }
            else
            {
                return index;
            }
        }
        return index;
    };

    for (size_t i = 0; i < in.data.format_desc.blocking.inner_nblks; i++)
    {
        md.format_desc.blocking.inner_blks[i] = in.data.format_desc.blocking.inner_blks[i];
        md.format_desc.blocking.inner_idxs[i] =
            get_axis_after_squeeze(in.data.format_desc.blocking.inner_idxs[i]);
    }
    md.offset0 = in.data.offset0;
    md.extra.flags = in.data.extra.flags;
    md.extra.scale_adjust = in.data.extra.scale_adjust;

    return try_get_named_md(md);
}

memory::desc runtime::cpu::mkldnn_utils::expand_blocked_md(const memory::desc& in,
                                                           AxisVector& axis_list)
{
    mkldnn_memory_desc_t md;
    md.ndims = in.data.ndims + static_cast<int>(axis_list.size());
    md.format_kind = mkldnn_blocked;
    md.data_type = in.data.data_type;
    md.format_desc.blocking.inner_nblks = in.data.format_desc.blocking.inner_nblks;

    size_t k = 0;
    for (size_t i = 0, j = 0; j < md.ndims; j++)
    {
        if (k < axis_list.size() && j == axis_list[k])
        {
            k++;
            md.dims[j] = 1;
            md.padded_dims[j] = 1;
            md.padded_offsets[j] = 0;
            if (i > 0)
            {
                md.format_desc.blocking.strides[j] = in.data.format_desc.blocking.strides[i - 1];
            }
            else
            {
                size_t nelems = 1;
                for (size_t idx = 0; idx < in.data.ndims; idx++)
                    nelems *= in.data.padded_dims[idx];
                md.format_desc.blocking.strides[j] = nelems;
            }
        }
        else
        {
            md.format_desc.blocking.strides[j] = in.data.format_desc.blocking.strides[i];
            md.padded_dims[j] = in.data.padded_dims[i];
            md.padded_offsets[j] = in.data.padded_offsets[i];
            md.dims[j] = in.data.dims[i];

            i++;
        }
    }

    std::vector<size_t> input_axis_after_expand_list;
    for (size_t i = 0; i < md.ndims; i++)
    {
        if (std::find(axis_list.begin(), axis_list.end(), i) == axis_list.end())
        {
            input_axis_after_expand_list.push_back(i);
        }
    }

    for (size_t i = 0; i < in.data.format_desc.blocking.inner_nblks; i++)
    {
        md.format_desc.blocking.inner_blks[i] = in.data.format_desc.blocking.inner_blks[i];
        md.format_desc.blocking.inner_idxs[i] =
            input_axis_after_expand_list[in.data.format_desc.blocking.inner_idxs[i]];
    }

    md.offset0 = in.data.offset0;
    md.extra.flags = in.data.extra.flags;
    md.extra.scale_adjust = in.data.extra.scale_adjust;

    return try_get_named_md(md);
}

bool runtime::cpu::mkldnn_utils::compare_mkldnn_formats(mkldnn::memory::format_tag lhs,
                                                        mkldnn::memory::format_tag rhs)
{
    return lhs == rhs;
}

bool runtime::cpu::mkldnn_utils::compare_mkldnn_mds(const mkldnn::memory::desc& lhs,
                                                    const mkldnn::memory::desc& rhs)
{
    mkldnn_memory_desc_t md1 = lhs.data, md2 = rhs.data;

    if (md1.format_kind != md2.format_kind)
    {
        return false;
    }

    if (md1.format_kind != static_cast<mkldnn_format_kind_t>(mkldnn::memory::format_kind::blocked))
    {
        // mkldnn not implemented yet
        return false;
    }

    if (md1.offset0 != md2.offset0 || md1.extra.flags != md2.extra.flags ||
        // scale_adjust can only be 0.5 or 1.
        std::fabs(md1.extra.scale_adjust - md2.extra.scale_adjust) > 0.1f)
    {
        return false;
    }

    auto blk1 = md1.format_desc.blocking;
    auto blk2 = md2.format_desc.blocking;

    if (md1.ndims != md2.ndims || !compare_mkldnn_dims(md1.dims, md2.dims, md1.ndims) ||
        !compare_mkldnn_dims(md1.padded_dims, md2.padded_dims, md1.ndims) ||
        !compare_mkldnn_dims(md1.padded_offsets, md2.padded_offsets, md1.ndims) ||
        !compare_mkldnn_dims(blk1.strides, blk2.strides, md1.ndims))
    {
        return false;
    }

    if (blk1.inner_nblks != blk2.inner_nblks ||
        !compare_mkldnn_dims(blk1.inner_blks, blk2.inner_blks, blk1.inner_nblks) ||
        !compare_mkldnn_dims(blk1.inner_idxs, blk2.inner_idxs, blk1.inner_nblks))
    {
        return false;
    }

    return true;
}

bool inline runtime::cpu::mkldnn_utils::compare_mkldnn_dims(mkldnn_dims_t& arr1,
                                                            mkldnn_dims_t& arr2,
                                                            size_t size)
{
    for (auto i = 0; i < size; i++)
    {
        if (arr1[i] != arr2[i])
        {
            return false;
        }
    }
    return true;
}

bool runtime::cpu::mkldnn_utils::compare_mkldnn_strides_order(mkldnn_dims_t& strides1,
                                                              mkldnn_dims_t& strides2,
                                                              size_t size)
{
    std::vector<size_t> indices1(size, 0), indices2(size, 0);
    for (size_t i = 0; i < size; i++)
    {
        indices1[i] = i;
        indices2[i] = i;
    }
    std::sort(indices1.begin(), indices1.begin(), [&](const size_t& n1, const size_t& n2) {
        return strides1[n1] < strides1[n2];
    });
    std::sort(indices2.begin(), indices2.begin(), [&](const size_t& n1, const size_t& n2) {
        return strides2[n1] < strides2[n2];
    });

    for (auto i = 0; i < size; i++)
    {
        if (indices1[i] != indices2[i])
        {
            return false;
        }
    }
    return true;
}

bool runtime::cpu::mkldnn_utils::compare_mkldnn_md_formats(const mkldnn::memory::desc& lhs,
                                                           const mkldnn::memory::desc& rhs)
{
    mkldnn_memory_desc_t md1 = lhs.data, md2 = rhs.data;

    if (md1.format_kind != md2.format_kind)
    {
        return false;
    }

    if (md1.format_kind != static_cast<mkldnn_format_kind_t>(mkldnn::memory::format_kind::blocked))
    {
        // mkldnn not implemented yet
        return false;
    }

    if (md1.ndims != md2.ndims)
    {
        return false;
    }

    auto blk1 = md1.format_desc.blocking;
    auto blk2 = md2.format_desc.blocking;

    if (blk1.inner_nblks != blk2.inner_nblks ||
        !compare_mkldnn_dims(blk1.inner_blks, blk2.inner_blks, blk1.inner_nblks) ||
        !compare_mkldnn_dims(blk1.inner_idxs, blk2.inner_idxs, blk1.inner_nblks))
    {
        return false;
    }

    return compare_mkldnn_strides_order(blk1.strides, blk2.strides, md1.ndims);
}

bool runtime::cpu::mkldnn_utils::mkldnn_md_matches_format_tag(const mkldnn::memory::desc& desc,
                                                              const mkldnn::memory::format_tag& fmt)
{
    auto format_tag_to_kind = [](mkldnn::memory::format_tag tag) {
        if (tag == mkldnn::memory::format_tag::undef)
        {
            return mkldnn::memory::format_kind::undef;
        }
        else if (tag == mkldnn::memory::format_tag::any)
        {
            return mkldnn::memory::format_kind::any;
        }
        else
        {
            return mkldnn::memory::format_kind::blocked;
        }
    };

    mkldnn_memory_desc_t md = desc.data;
    if (md.format_kind != static_cast<mkldnn_format_kind_t>(format_tag_to_kind(fmt)))
    {
        return false;
    }

    mkldnn_memory_desc_t named_md;
    try
    {
        // Could throw an exception if named `format` is not compatible with `md.dims`
        error::wrap_c_api(
            mkldnn_memory_desc_init_by_tag(
                &named_md, md.ndims, md.dims, md.data_type, static_cast<mkldnn_format_tag_t>(fmt)),
            "");
    }
    catch (const mkldnn::error&)
    {
        // Cannot create the named descriptor compatible with `md` desc
        return false;
    }

    if (md.format_kind != static_cast<mkldnn_format_kind_t>(mkldnn::memory::format_kind::blocked))
    {
        // mkldnn not implemented yet
        return false;
    }

    auto blk = md.format_desc.blocking;
    auto named_blk = named_md.format_desc.blocking;

    if (blk.inner_nblks != named_blk.inner_nblks ||
        !compare_mkldnn_dims(blk.inner_blks, named_blk.inner_blks, blk.inner_nblks) ||
        !compare_mkldnn_dims(blk.inner_idxs, named_blk.inner_idxs, blk.inner_nblks) ||
        !compare_mkldnn_dims(blk.strides, named_blk.strides, md.ndims))
    {
        return false;
    }

    return true;
}

bool runtime::cpu::mkldnn_utils::is_mkldnn_padded_layout(const mkldnn::memory::desc& in,
                                                         const AxisVector& axis_list)
{
    for (size_t i = 0; i < in.data.ndims; i++)
    {
        if (std::find(axis_list.begin(), axis_list.end(), i) == axis_list.end())
        {
            continue;
        }
        if (in.data.padded_dims[i] != in.data.dims[i])
        {
            return true;
        }
        if (in.data.padded_offsets[i] != 0)
        {
            return true;
        }
    }

    return false;
}

bool runtime::cpu::mkldnn_utils::is_mkldnn_desc_blocked_data_format(
    const mkldnn::memory::desc& desc)
{
    auto blk = desc.data.format_desc.blocking;
// TODO for v0.x, we just check if nChw8c or nChw16c, should we do the same here?
#if 0
	// Check if nChw8c or nChw16c
	if (desc.data.ndims != 4 || blk.inner_nblks != 1 ||
		(blk.inner_blks[0] != 8 && blk.inner_blks[0] != 16) ||
	    blk.inner_idxs[0] != 1)
	{
		return false;
	}
	std::vector<size_t> perm{0, 1, 2, 3};
    for (size_t i = 0; i < 3; i++)
    {
        if (blk.strides[i] < blk.strides[i + 1]])
            return false;
    }
	return true;
#endif
    return blk.inner_nblks != 0;
}
#endif
