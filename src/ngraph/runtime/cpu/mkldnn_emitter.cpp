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

#include <memory>
#include <string>

#include "mkldnn_emitter.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_concat.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/deconv.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;
using namespace ngraph::op;
using namespace ngraph::runtime::cpu;

MKLDNNEmitter::~MKLDNNEmitter()
{
    for (auto p : m_mkldnn_primitives)
        delete p;
#ifndef _WIN32
    //To avoid memory leak in mkldnn, release any buffers that are not free'd yet.
    //https://software.intel.com/en-us/mkl-linux-developer-guide-avoiding-memory-leaks-in-intel-mkl
    //mkl_free_buffers() is not exposed at this point, hence using mkl_serv_free_buffers()
    mkldnn_utils::mkl_serv_free_buffers();
#endif
}

std::vector<mkldnn::primitive*>& MKLDNNEmitter::get_mkldnn_primitives()
{
    return m_mkldnn_primitives;
}

const std::vector<mkldnn::primitive*>& MKLDNNEmitter::get_mkldnn_primitives_cg() const
{
    return m_mkldnn_primitives_cg;
}

const std::vector<char*>& MKLDNNEmitter::get_mkldnn_workspaces()
{
    return m_workspace_bufs;
}

size_t MKLDNNEmitter::insert_primitive(mkldnn::primitive* primitive)
{
    m_mkldnn_primitives.emplace_back(primitive);
    return (m_mkldnn_primitives.size() - 1);
}

size_t MKLDNNEmitter::insert_workspace(std::unique_ptr<MKLDNNWorkspace>& workspace)
{
    m_workspace_bufs.push_back(workspace.get()->buf);
    m_workspaces.push_back(std::move(workspace));
    return (m_workspaces.size() - 1);
}

size_t MKLDNNEmitter::reserve_workspace()
{
    m_workspaces_size++;
    return m_workspaces_size - 1;
}

void MKLDNNEmitter::reserve_descriptor_space(size_t count)
{
    m_mkldnn_descriptors_size += count;
}

size_t MKLDNNEmitter::get_mkldnn_descriptors_size()
{
    return m_mkldnn_descriptors_size;
}

size_t MKLDNNEmitter::insert_workspace(std::vector<char*>& mkldnn_workspaces,
                                       std::unique_ptr<MKLDNNWorkspace>& workspace)
{
    mkldnn_workspaces.push_back(workspace.get()->buf);
    m_workspaces.push_back(std::move(workspace));
    return (mkldnn_workspaces.size() - 1);
}

const std::vector<size_t>& MKLDNNEmitter::get_primitive_deps(size_t index) const
{
    return m_primitive_deps.at(index);
}

const std::vector<size_t>& MKLDNNEmitter::get_primitive_deps_cg(size_t index) const
{
    return m_primitive_deps_cg.at(index);
}

std::vector<size_t>& MKLDNNEmitter::get_primitive_deps(size_t index)
{
    return m_primitive_deps.at(index);
}

mkldnn::memory::desc MKLDNNEmitter::build_memory_descriptor(const TensorViewWrapper& tvw,
                                                            mkldnn::memory::format fmt) const
{
    if (fmt == mkldnn::memory::format::blocked)
    {
        throw ngraph_error("Cannot created blocked descriptor.");
    }
    return mkldnn::memory::desc(
        mkldnn::memory::dims(tvw.get_shape().begin(), tvw.get_shape().end()),
        mkldnn_utils::get_mkldnn_data_type(tvw.get_element_type()),
        fmt);
}

mkldnn::memory::desc MKLDNNEmitter::build_memory_descriptor(const ngraph::Shape& shape,
                                                            const ngraph::element::Type& et,
                                                            mkldnn::memory::format fmt) const
{
    if (fmt == mkldnn::memory::format::blocked)
    {
        throw ngraph_error("Cannot created blocked descriptor");
    }
    return mkldnn::memory::desc(mkldnn::memory::dims(shape.begin(), shape.end()),
                                mkldnn_utils::get_mkldnn_data_type(et),
                                fmt);
}

mkldnn::memory::desc
    MKLDNNEmitter::build_blocked_memory_descriptor(const mkldnn::memory::dims& dim,
                                                   const mkldnn::memory::dims& strides,
                                                   mkldnn::memory::data_type dtype) const
{
    mkldnn_memory_desc_t md;
    md.primitive_kind = mkldnn_memory;
    md.ndims = static_cast<int>(dim.size());
    md.format = mkldnn_blocked;
    md.data_type = mkldnn::memory::convert_to_c(dtype);

    for (size_t i = 0; i < dim.size(); i++)
    {
        md.layout_desc.blocking.block_dims[i] = 1;
        md.layout_desc.blocking.strides[1][i] = 1;
        md.layout_desc.blocking.strides[0][i] = strides[i];
        md.layout_desc.blocking.padding_dims[i] = dim[i];
        md.layout_desc.blocking.offset_padding_to_data[i] = 0;
        md.dims[i] = dim[i];
    }
    md.layout_desc.blocking.offset_padding = 0;

    return mkldnn::memory::desc(md);
}

size_t MKLDNNEmitter::build_memory_primitive(const mkldnn::memory::desc& desc)
{
    size_t index =
        insert_primitive(new mkldnn::memory({desc, executor::global_cpu_engine}, nullptr));
    return index;
}

void MKLDNNEmitter::build_memory_primitive(const mkldnn::memory::desc& desc, size_t index)
{
    m_mkldnn_primitives[index] = new mkldnn::memory({desc, executor::global_cpu_engine}, nullptr);
}

void MKLDNNEmitter::build_memory_primitive(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                           const mkldnn::memory::desc& desc,
                                           size_t index)
{
    mkldnn_primitives[index] = new mkldnn::memory({desc, executor::global_cpu_engine}, nullptr);
}

size_t MKLDNNEmitter::build_quantize_reorder(const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& result_desc,
                                             const std::vector<float>& scales)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);
    mkldnn::primitive_attr attr;
    attr.set_output_scales(0, scales);
    attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
    auto reorder_desc = mkldnn::reorder::primitive_desc({input_desc, executor::global_cpu_engine},
                                                        {result_desc, executor::global_cpu_engine},
                                                        attr);
    size_t primitive_index = insert_primitive(new mkldnn::reorder(
        reorder_desc, *m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]));

    NGRAPH_CHECK(m_primitive_deps.find(primitive_index) == m_primitive_deps.end(),
                 "Dependencies already created for node");

    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
}

void MKLDNNEmitter::build_quantize_reorder(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                           const mkldnn::memory::desc& input_desc,
                                           const mkldnn::memory::desc& result_desc,
                                           const std::vector<float>& scales,
                                           const std::vector<size_t>& deps,
                                           size_t quantize_index,
                                           const int mask)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, input_desc, input_index);
    size_t result_index = deps[1];
    build_memory_primitive(mkldnn_primitives, result_desc, result_index);

    mkldnn::primitive_attr attr;
    attr.set_output_scales(mask, scales);
    attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
    auto reorder_desc = mkldnn::reorder::primitive_desc({input_desc, executor::global_cpu_engine},
                                                        {result_desc, executor::global_cpu_engine},
                                                        attr);
    mkldnn_primitives[quantize_index] = new mkldnn::reorder(
        reorder_desc, *mkldnn_primitives[input_index], *mkldnn_primitives[result_index]);
}

size_t MKLDNNEmitter::build_dequantization(const ngraph::Node* node,
                                           const mkldnn::memory::desc& input_desc,
                                           const mkldnn::memory::desc& result_desc)
{
    auto dequantize = static_cast<const ngraph::op::Dequantize*>(node);
    auto scale_const_op =
        std::dynamic_pointer_cast<ngraph::op::Constant>(dequantize->get_argument(1));
    std::vector<float> scale = {1.0f};
    if (scale_const_op != nullptr)
    {
        scale = scale_const_op->get_vector<float>();
    }
    std::vector<float> scales;
    scales.push_back(scale[0]);
    size_t dequantize_index = 0;
    dequantize_index = this->build_quantize_reorder(input_desc, result_desc, scales);
    return dequantize_index;
}

mkldnn::memory::format MKLDNNEmitter::query_convolution_forward_weight_format(
    const mkldnn::memory::desc& input_data_desc,
    const mkldnn::memory::desc& weights_desc_any,
    const mkldnn::memory::desc& result_desc,
    const ngraph::Strides& filter_strides,
    const ngraph::Strides& window_dilation_strides_adjusted,
    const ngraph::CoordinateDiff& padding_below,
    const ngraph::CoordinateDiff& padding_above)

{
    mkldnn::memory::dims mkldnn_filter_strides(filter_strides.begin(), filter_strides.end());
    mkldnn::memory::dims mkldnn_dilated_strides(window_dilation_strides_adjusted.begin(),
                                                window_dilation_strides_adjusted.end());
    mkldnn::memory::dims mkldnn_padding_below(padding_below.begin(), padding_below.end());
    mkldnn::memory::dims mkldnn_padding_above(padding_above.begin(), padding_above.end());

    mkldnn::algorithm convolution_algo = mkldnn_utils::get_conv_algo();
    mkldnn::engine cpu_engine(mkldnn::engine::cpu, 0);
    mkldnn::convolution_forward::desc conv_desc_layout(
        mkldnn::prop_kind::forward_inference,
        convolution_algo,
        input_data_desc,
        weights_desc_any, // this needs to be in default format
        result_desc,
        mkldnn_filter_strides,
        mkldnn_dilated_strides,
        mkldnn_padding_below,
        mkldnn_padding_above,
        mkldnn::padding_kind::zero);

    mkldnn::convolution_forward::primitive_desc prim_desc(conv_desc_layout, cpu_engine);
    return static_cast<mkldnn::memory::format>(
        prim_desc.weights_primitive_desc().desc().data.format);
}

void MKLDNNEmitter::build_deconvolutionbias_forward(
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    const mkldnn::deconvolution_forward::desc& deconv_desc,
    const std::vector<size_t>& deps,
    size_t deconv_index,
    const mkldnn::memory::desc& weights_desc)
{
    size_t weights_index = deps[0];
    build_memory_primitive(mkldnn_primitives, weights_desc, weights_index);
    size_t delta_index = deps[1];
    build_memory_primitive(mkldnn_primitives, deconv_desc.data.src_desc, delta_index);
    size_t bias_index = deps[2];
    build_memory_primitive(mkldnn_primitives, deconv_desc.data.bias_desc, bias_index);
    size_t result_index = deps[3];
    build_memory_primitive(mkldnn_primitives, deconv_desc.data.dst_desc, result_index);

    try
    {
        mkldnn_primitives[deconv_index] =
            new mkldnn::deconvolution_forward({deconv_desc, executor::global_cpu_engine},
                                              *mkldnn_primitives[delta_index],
                                              *mkldnn_primitives[weights_index],
                                              *mkldnn_primitives[bias_index],
                                              *mkldnn_primitives[result_index]);
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not create mkldnn deconvolution_forward " + e.message);
    }
}

void MKLDNNEmitter::build_convolution_backward_weights_bias(
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    const mkldnn::convolution_backward_weights::desc& bwd_desc,
    const mkldnn::convolution_forward::desc& fwd_desc,
    const std::vector<size_t>& deps,
    size_t conv_index)
{
    size_t src_index = deps[0];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.src_desc, src_index);
    size_t diff_dst_index = deps[1];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.diff_dst_desc, diff_dst_index);
    size_t diff_weights_index = deps[2];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.diff_weights_desc, diff_weights_index);
    size_t diff_bias_index = deps[3];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.diff_bias_desc, diff_bias_index);

    mkldnn::convolution_forward::primitive_desc fwd_pd{fwd_desc, executor::global_cpu_engine};

    mkldnn::convolution_backward_weights::primitive_desc bwd_pd{
        bwd_desc, executor::global_cpu_engine, fwd_pd};

    mkldnn_primitives[conv_index] =
        new mkldnn::convolution_backward_weights(bwd_pd,
                                                 *mkldnn_primitives[src_index],
                                                 *mkldnn_primitives[diff_dst_index],
                                                 *mkldnn_primitives[diff_weights_index],
                                                 *mkldnn_primitives[diff_bias_index]);
}

void MKLDNNEmitter::build_convolution_backward_weights(
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    const mkldnn::convolution_backward_weights::desc& bwd_desc,
    const mkldnn::convolution_forward::desc& fwd_desc,
    const std::vector<size_t>& deps,
    size_t conv_index)
{
    size_t src_index = deps[0];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.src_desc, src_index);
    size_t diff_dst_index = deps[1];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.diff_dst_desc, diff_dst_index);
    size_t diff_weights_index = deps[2];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.diff_weights_desc, diff_weights_index);

    mkldnn_primitives[conv_index] = new mkldnn::convolution_backward_weights(
        {bwd_desc,
         executor::global_cpu_engine,
         // Forward primitive descriptor corresponding to this backward weights descriptor
         {fwd_desc, executor::global_cpu_engine}},
        *mkldnn_primitives[src_index],
        *mkldnn_primitives[diff_dst_index],
        *mkldnn_primitives[diff_weights_index]);
}

void MKLDNNEmitter::build_convolution_backward_data(
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    const mkldnn::convolution_backward_data::desc& bwd_desc,
    const mkldnn::convolution_forward::desc& fwd_desc,
    const std::vector<size_t>& deps,
    size_t conv_index)
{
    size_t weights_index = deps[0];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.weights_desc, weights_index);
    size_t diff_dst_index = deps[1];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.diff_dst_desc, diff_dst_index);
    size_t diff_src_index = deps[2];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.diff_src_desc, diff_src_index);

    mkldnn_primitives[conv_index] = new mkldnn::convolution_backward_data(
        {bwd_desc,
         executor::global_cpu_engine,
         // Forward primitive descriptor corresponding to this backward data descriptor
         {fwd_desc, executor::global_cpu_engine}},
        *mkldnn_primitives[diff_dst_index],
        *mkldnn_primitives[weights_index],
        *mkldnn_primitives[diff_src_index]);
}

void MKLDNNEmitter::build_pooling_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                          const mkldnn::pooling_forward::desc& pool_desc,
                                          const std::vector<size_t>& deps,
                                          size_t pool_index)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, pool_desc.data.src_desc, input_index);
    size_t result_index = deps[1];
    build_memory_primitive(mkldnn_primitives, pool_desc.data.dst_desc, result_index);

    mkldnn_primitives[pool_index] =
        new mkldnn::pooling_forward({pool_desc, executor::global_cpu_engine},
                                    *mkldnn_primitives[input_index],
                                    *mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_pooling_backward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                           const mkldnn::pooling_backward::desc& pool_desc,
                                           const mkldnn::pooling_forward::desc& pool_fwd_desc,
                                           const std::vector<size_t>& deps,
                                           size_t pool_index)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, pool_desc.data.diff_dst_desc, input_index);
    size_t result_index = deps[1];
    build_memory_primitive(mkldnn_primitives, pool_desc.data.diff_src_desc, result_index);

    auto pool_fwd_pd =
        mkldnn::pooling_forward::primitive_desc(pool_fwd_desc, executor::global_cpu_engine);
    auto pool_pd = mkldnn::pooling_backward::primitive_desc(
        pool_desc, executor::global_cpu_engine, pool_fwd_pd);

    mkldnn_primitives[pool_index] = new mkldnn::pooling_backward(
        pool_pd, *mkldnn_primitives[input_index], *mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_max_pooling_backward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                               std::vector<char*>& mkldnn_workspaces,
                                               const mkldnn::pooling_backward::desc& bwd_pool_desc,
                                               const mkldnn::pooling_forward::desc& fwd_pool_desc,
                                               const mkldnn::memory::desc& fprop_src_desc,
                                               std::vector<size_t>& fdeps,
                                               std::vector<size_t>& bdeps,
                                               size_t fwd_pool_index,
                                               size_t bwd_pool_index)
{
    size_t fprop_src_index = fdeps[0];
    build_memory_primitive(mkldnn_primitives, fprop_src_desc, fprop_src_index);
    size_t diff_dst_index = bdeps[0];
    build_memory_primitive(mkldnn_primitives, bwd_pool_desc.data.diff_dst_desc, diff_dst_index);
    size_t diff_src_index = fdeps[1];
    build_memory_primitive(mkldnn_primitives, bwd_pool_desc.data.diff_src_desc, diff_src_index);
    bdeps[2] = diff_src_index;

    mkldnn::pooling_forward::primitive_desc fwd_pd{fwd_pool_desc, executor::global_cpu_engine};

    size_t ws_index = fdeps[2];
    build_memory_primitive(mkldnn_primitives, fwd_pd.workspace_primitive_desc().desc(), ws_index);
    bdeps[1] = ws_index;

    // Allocate workspace
    // TODO (jbobba): Might need to align memory
    auto ws = std::unique_ptr<MKLDNNWorkspace>(
        new MKLDNNWorkspace(fwd_pd.workspace_primitive_desc().get_size()));
    auto ws_buf_index = insert_workspace(mkldnn_workspaces, ws);
    fdeps[3] = ws_buf_index;
    bdeps[3] = ws_buf_index;

    mkldnn_primitives[fwd_pool_index] = new mkldnn::pooling_forward(
        fwd_pd,
        *mkldnn_primitives[fprop_src_index],
        *mkldnn_primitives
            [diff_src_index], // HACK - Uses diff_src buffer. Safe since diff_src > fprop_dst
        *mkldnn_primitives[ws_index]);

    mkldnn_primitives[bwd_pool_index] =
        new mkldnn::pooling_backward({bwd_pool_desc, executor::global_cpu_engine, fwd_pd},
                                     *mkldnn_primitives[diff_dst_index],
                                     *mkldnn_primitives[ws_index],
                                     *mkldnn_primitives[diff_src_index]);
}

void MKLDNNEmitter::build_max_pooling_with_indices_forward(
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    const mkldnn::pooling_forward::desc& max_pool_desc,
    const std::vector<size_t>& deps,
    size_t max_pool_index)
{
    size_t src_index = deps[0];
    build_memory_primitive(mkldnn_primitives, max_pool_desc.data.src_desc, src_index);
    size_t dst_index = deps[1];
    build_memory_primitive(mkldnn_primitives, max_pool_desc.data.dst_desc, dst_index);

    mkldnn::pooling_forward::primitive_desc fwd_pd{max_pool_desc, executor::global_cpu_engine};

    size_t ws_index = deps[2];
    build_memory_primitive(mkldnn_primitives, fwd_pd.workspace_primitive_desc().desc(), ws_index);

    mkldnn_primitives[max_pool_index] = new mkldnn::pooling_forward(fwd_pd,
                                                                    *mkldnn_primitives[src_index],
                                                                    *mkldnn_primitives[dst_index],
                                                                    *mkldnn_primitives[ws_index]);
}

void MKLDNNEmitter::build_max_pooling_with_indices_backward(
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    const mkldnn::pooling_backward::desc& bwd_pool_desc,
    const mkldnn::pooling_forward::desc& fwd_pool_desc,
    const std::vector<size_t>& deps,
    size_t max_pool_index)
{
    size_t diff_dst_index = deps[0];
    build_memory_primitive(mkldnn_primitives, bwd_pool_desc.data.diff_dst_desc, diff_dst_index);
    size_t diff_src_index = deps[2];
    build_memory_primitive(mkldnn_primitives, bwd_pool_desc.data.diff_src_desc, diff_src_index);

    mkldnn::pooling_forward::primitive_desc fwd_pd{fwd_pool_desc, executor::global_cpu_engine};

    size_t fprop_ws_index = deps[1];
    build_memory_primitive(
        mkldnn_primitives, fwd_pd.workspace_primitive_desc().desc(), fprop_ws_index);

    mkldnn_primitives[max_pool_index] =
        new mkldnn::pooling_backward({bwd_pool_desc, executor::global_cpu_engine, fwd_pd},
                                     *mkldnn_primitives[diff_dst_index],
                                     *mkldnn_primitives[fprop_ws_index],
                                     *mkldnn_primitives[diff_src_index]);
}

size_t MKLDNNEmitter::build_reorder(const mkldnn::memory::desc& input_desc,
                                    const mkldnn::memory::desc& result_desc)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index = 0;
    try
    {
        primitive_index = insert_primitive(new mkldnn::reorder(*m_mkldnn_primitives[input_index],
                                                               *m_mkldnn_primitives[result_index]));

        NGRAPH_CHECK(m_primitive_deps.find(primitive_index) == m_primitive_deps.end(),
                     "Dependencies already created for node");

        m_primitive_deps[primitive_index] = {input_index, result_index};
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not create mkldnn primitive " + e.message);
    }

    return primitive_index;
}

void MKLDNNEmitter::build_reorder(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                  const mkldnn::memory::desc& input_desc,
                                  const mkldnn::memory::desc& result_desc,
                                  const std::vector<size_t>& deps,
                                  size_t reorder_index)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, input_desc, input_index);
    size_t result_index = deps[1];
    build_memory_primitive(mkldnn_primitives, result_desc, result_index);

    mkldnn_primitives[reorder_index] =
        new mkldnn::reorder(*mkldnn_primitives[input_index], *mkldnn_primitives[result_index]);
}

mkldnn::lrn_forward::desc MKLDNNEmitter::get_lrn_forward_desc(const ngraph::Node* node)
{
    const ngraph::op::LRN* lrn = static_cast<const ngraph::op::LRN*>(node);

    auto alpha = static_cast<float>(lrn->get_alpha());
    auto beta = static_cast<float>(lrn->get_beta());
    auto bias = static_cast<float>(lrn->get_bias());
    auto nsize = static_cast<int>(lrn->get_nsize());

    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);

    return mkldnn::lrn_forward::desc(mkldnn::prop_kind::forward_scoring,
                                     mkldnn::algorithm::lrn_across_channels,
                                     input_desc,
                                     nsize,
                                     alpha,
                                     beta,
                                     bias);
}

void MKLDNNEmitter::build_lrn_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                      const mkldnn::lrn_forward::desc& lrn_desc,
                                      const std::vector<size_t>& deps,
                                      size_t lrn_index)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, lrn_desc.data.data_desc, input_index);
    size_t result_index = deps[1];
    build_memory_primitive(mkldnn_primitives, lrn_desc.data.data_desc, result_index);

    auto lrn_prim_desc = mkldnn::lrn_forward::primitive_desc(lrn_desc, executor::global_cpu_engine);

    mkldnn_primitives[lrn_index] = new mkldnn::lrn_forward(
        lrn_prim_desc, *mkldnn_primitives[input_index], *mkldnn_primitives[result_index]);
}

mkldnn::eltwise_forward::desc MKLDNNEmitter::get_relu_forward_desc(const ngraph::Node* node)
{
    const float negative_slope = 0.0f;

    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);

    return mkldnn::eltwise_forward::desc(
        mkldnn::prop_kind::forward, mkldnn::algorithm::eltwise_relu, input_desc, negative_slope);
}

void MKLDNNEmitter::build_relu_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                       const mkldnn::eltwise_forward::desc& relu_desc,
                                       const std::vector<size_t>& deps,
                                       size_t relu_index)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, relu_desc.data.data_desc, input_index);
    size_t result_index = deps[1];
    build_memory_primitive(mkldnn_primitives, relu_desc.data.data_desc, result_index);

    mkldnn_primitives[relu_index] =
        new mkldnn::eltwise_forward({relu_desc, executor::global_cpu_engine},
                                    *mkldnn_primitives[input_index],
                                    *mkldnn_primitives[result_index]);
}

mkldnn::eltwise_backward::desc MKLDNNEmitter::get_relu_backward_desc(const ngraph::Node* node)
{
    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

    const float negative_slope = 0.0f;
    return mkldnn::eltwise_backward::desc(
        mkldnn::algorithm::eltwise_relu, result_desc, input_desc, negative_slope);
}

void MKLDNNEmitter::build_relu_backward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                        const mkldnn::eltwise_backward::desc& bwd_desc,
                                        const mkldnn::eltwise_forward::desc& fwd_desc,
                                        const std::vector<size_t>& deps,
                                        size_t relu_index)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.data_desc, input_index);
    size_t delta_index = deps[1];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.diff_data_desc, delta_index);
    size_t result_index = deps[2];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.data_desc, result_index);

    /* create forward relu primitive descriptor*/
    auto relu_pd = mkldnn::eltwise_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    /* create backward relu primitive_descriptor */
    auto relu_bwd_pd =
        mkldnn::eltwise_backward::primitive_desc(bwd_desc, executor::global_cpu_engine, relu_pd);

    mkldnn_primitives[relu_index] = new mkldnn::eltwise_backward(relu_bwd_pd,
                                                                 *mkldnn_primitives[input_index],
                                                                 *mkldnn_primitives[delta_index],
                                                                 *mkldnn_primitives[result_index]);
}

mkldnn::eltwise_forward::desc MKLDNNEmitter::get_sigmoid_forward_desc(const ngraph::Node* node,
                                                                      bool backward_op)
{
    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);

    if (backward_op)
    {
        return mkldnn::eltwise_forward::desc(
            mkldnn::prop_kind::forward, mkldnn::algorithm::eltwise_logistic, input_desc, 0, 0);
    }
    else
    {
        return mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward_training,
                                             mkldnn::algorithm::eltwise_logistic,
                                             input_desc,
                                             0,
                                             0);
    }
}

void MKLDNNEmitter::build_sigmoid_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                          const mkldnn::eltwise_forward::desc& sigmoid_desc,
                                          const std::vector<size_t>& deps,
                                          size_t sigmoid_index)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, sigmoid_desc.data.data_desc, input_index);
    size_t result_index = deps[1];
    build_memory_primitive(mkldnn_primitives, sigmoid_desc.data.data_desc, result_index);

    mkldnn_primitives[sigmoid_index] =
        new mkldnn::eltwise_forward({sigmoid_desc, executor::global_cpu_engine},
                                    *mkldnn_primitives[input_index],
                                    *mkldnn_primitives[result_index]);
}

mkldnn::eltwise_backward::desc MKLDNNEmitter::get_sigmoid_backward_desc(const ngraph::Node* node)
{
    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);

    return mkldnn::eltwise_backward::desc(
        mkldnn::algorithm::eltwise_logistic, delta_desc, input_desc, 0, 0);
}

void MKLDNNEmitter::build_sigmoid_backward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                           const mkldnn::eltwise_backward::desc& bwd_desc,
                                           const mkldnn::eltwise_forward::desc& fwd_desc,
                                           const std::vector<size_t>& deps,
                                           size_t sigmoid_index)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.data_desc, input_index);
    size_t delta_index = deps[1];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.diff_data_desc, delta_index);
    size_t result_index = deps[2];
    build_memory_primitive(mkldnn_primitives, bwd_desc.data.data_desc, result_index);

    // sigmoid forward primitive desc
    mkldnn::eltwise_forward::primitive_desc sigmoid_fwd_pd =
        mkldnn::eltwise_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    mkldnn_primitives[sigmoid_index] =
        new mkldnn::eltwise_backward({bwd_desc, executor::global_cpu_engine, sigmoid_fwd_pd},
                                     *mkldnn_primitives[input_index],
                                     *mkldnn_primitives[delta_index],
                                     *mkldnn_primitives[result_index]);
}

mkldnn::sum::primitive_desc MKLDNNEmitter::get_elementwise_add_desc(const ngraph::Node* node)
{
    std::vector<float> scale_vector(2, 1);
    std::vector<mkldnn::memory::primitive_desc> inputs_pd;

    auto input0_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
    auto input1_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
    inputs_pd.push_back(mkldnn::memory::primitive_desc(
        input0_data_desc, ngraph::runtime::cpu::executor::global_cpu_engine));
    inputs_pd.push_back(mkldnn::memory::primitive_desc(
        input1_data_desc, ngraph::runtime::cpu::executor::global_cpu_engine));

    // elementwise sum primtive descriptor
    mkldnn::sum::primitive_desc sum_pd =
        mkldnn::sum::primitive_desc(result_desc, scale_vector, inputs_pd);
    return sum_pd;
}

void MKLDNNEmitter::build_elementwise_add(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                          const mkldnn::sum::primitive_desc& sum_pd,
                                          const std::vector<size_t>& deps,
                                          size_t add_index)
{
    std::vector<mkldnn::memory::primitive::at> inputs_primitive;

    size_t input0_data_index = deps[0];
    build_memory_primitive(
        mkldnn_primitives, sum_pd.dst_primitive_desc().desc(), input0_data_index);
    size_t input1_data_index = deps[1];
    build_memory_primitive(
        mkldnn_primitives, sum_pd.dst_primitive_desc().desc(), input1_data_index);
    size_t result_index = deps[2];
    build_memory_primitive(mkldnn_primitives, sum_pd.dst_primitive_desc().desc(), result_index);

    inputs_primitive.push_back(*mkldnn_primitives[input0_data_index]);
    inputs_primitive.push_back(*mkldnn_primitives[input1_data_index]);

    // sum primitive
    mkldnn_primitives[add_index] =
        new mkldnn::sum(sum_pd, inputs_primitive, *mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_batchnorm_forward(
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    const mkldnn::batch_normalization_forward::desc& batchnorm_desc,
    const mkldnn::memory::desc& weights_desc,
    bool bn_training_flag,
    const std::vector<size_t>& deps,
    size_t batchnorm_index,
    const mkldnn::post_ops& pops)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, batchnorm_desc.data.data_desc, input_index);

    mkldnn::primitive_attr bn_attr;
    bn_attr.set_post_ops(pops);

    auto use_global_stats = batchnorm_desc.data.flags & 0x1U;
    if (bn_training_flag && !use_global_stats)
    {
        size_t weights_index = deps[1];
        build_memory_primitive(mkldnn_primitives, weights_desc, weights_index);
        size_t result_index = deps[2];
        build_memory_primitive(mkldnn_primitives, batchnorm_desc.data.data_desc, result_index);
        size_t mean_index = deps[3];
        build_memory_primitive(mkldnn_primitives, batchnorm_desc.data.mean_desc, mean_index);
        size_t variance_index = deps[4];
        build_memory_primitive(
            mkldnn_primitives, batchnorm_desc.data.variance_desc, variance_index);

        mkldnn_primitives[batchnorm_index] = new mkldnn::batch_normalization_forward(
            {batchnorm_desc, bn_attr, executor::global_cpu_engine},
            mkldnn::primitive::at(*mkldnn_primitives[input_index]),
            mkldnn::primitive::at(*mkldnn_primitives[weights_index]),
            static_cast<mkldnn::memory>(*mkldnn_primitives[result_index]),
            *mkldnn_primitives[mean_index],
            *mkldnn_primitives[variance_index]);
    }
    else
    {
        size_t weights_index = deps[3];
        build_memory_primitive(mkldnn_primitives, weights_desc, weights_index);
        size_t result_index = deps[4];
        build_memory_primitive(mkldnn_primitives, batchnorm_desc.data.data_desc, result_index);
        size_t mean_index = deps[1];
        build_memory_primitive(mkldnn_primitives, batchnorm_desc.data.mean_desc, mean_index);
        size_t variance_index = deps[2];
        build_memory_primitive(
            mkldnn_primitives, batchnorm_desc.data.variance_desc, variance_index);

        mkldnn_primitives[batchnorm_index] = new mkldnn::batch_normalization_forward(
            {batchnorm_desc, bn_attr, executor::global_cpu_engine},
            mkldnn::primitive::at(*mkldnn_primitives[input_index]),
            mkldnn::primitive::at(*mkldnn_primitives[mean_index]),
            mkldnn::primitive::at(*mkldnn_primitives[variance_index]),
            mkldnn::primitive::at(*mkldnn_primitives[weights_index]),
            static_cast<mkldnn::memory>(*mkldnn_primitives[result_index]));
    }
}

mkldnn::batch_normalization_backward::desc
    MKLDNNEmitter::get_batchnorm_backward_desc(const ngraph::Node* node)
{
    const ngraph::op::BatchNormTrainingBackprop* batchnorm =
        static_cast<const ngraph::op::BatchNormTrainingBackprop*>(node);
    auto eps = batchnorm->get_eps_value();

    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 5);

    return mkldnn::batch_normalization_backward::desc(
        mkldnn::prop_kind::backward,
        delta_desc,
        input_desc,
        eps,
        mkldnn::batch_normalization_flag::use_scale_shift);
}

void MKLDNNEmitter::build_batchnorm_backward(
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    const mkldnn::batch_normalization_backward::desc& batchnorm_desc,
    const mkldnn::memory::desc& weights_desc,
    const mkldnn::memory::desc& dweights_desc,
    const std::vector<size_t>& deps,
    size_t batchnorm_index)
{
    size_t weights_index = deps[0];
    build_memory_primitive(mkldnn_primitives, weights_desc, weights_index);
    size_t input_index = deps[1];
    build_memory_primitive(mkldnn_primitives, batchnorm_desc.data.data_desc, input_index);
    size_t mean_index = deps[2];
    build_memory_primitive(mkldnn_primitives, batchnorm_desc.data.mean_desc, mean_index);
    size_t variance_index = deps[3];
    build_memory_primitive(mkldnn_primitives, batchnorm_desc.data.variance_desc, variance_index);
    size_t delta_index = deps[4];
    build_memory_primitive(mkldnn_primitives, batchnorm_desc.data.diff_data_desc, delta_index);
    size_t dinput_index = deps[5];
    build_memory_primitive(mkldnn_primitives, batchnorm_desc.data.data_desc, dinput_index);
    size_t dweights_index = deps[6];
    build_memory_primitive(mkldnn_primitives, dweights_desc, dweights_index);

    mkldnn_primitives[batchnorm_index] = new mkldnn::batch_normalization_backward(
        {batchnorm_desc,
         executor::global_cpu_engine,
         {{mkldnn::prop_kind::forward_training,
           batchnorm_desc.data.data_desc,
           static_cast<double>(batchnorm_desc.data.batch_norm_epsilon),
           mkldnn::batch_normalization_flag::use_scale_shift},
          executor::global_cpu_engine}},
        *mkldnn_primitives[input_index],
        *mkldnn_primitives[mean_index],
        *mkldnn_primitives[variance_index],
        *mkldnn_primitives[delta_index],
        *mkldnn_primitives[weights_index],
        *mkldnn_primitives[dinput_index],
        *mkldnn_primitives[dweights_index]);
}

void MKLDNNEmitter::build_rnn_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                      std::vector<char*>& mkldnn_workspaces,
                                      const mkldnn::rnn_forward::desc& rnn_desc,
                                      std::vector<size_t>& deps,
                                      size_t rnn_index)
{
    size_t src_layer_index = deps[0];
    build_memory_primitive(mkldnn_primitives, rnn_desc.data.src_layer_desc, src_layer_index);
    size_t src_iter_index = deps[1];
    build_memory_primitive(mkldnn_primitives, rnn_desc.data.src_iter_desc, src_iter_index);
    size_t weights_layer_index = deps[2];
    build_memory_primitive(
        mkldnn_primitives, rnn_desc.data.weights_layer_desc, weights_layer_index);

    size_t weights_iter_index = m_primitive_deps[rnn_index][3];
    build_memory_primitive(mkldnn_primitives, rnn_desc.data.weights_iter_desc, weights_iter_index);
    size_t bias_index = deps[4];
    build_memory_primitive(mkldnn_primitives, rnn_desc.data.bias_desc, bias_index);
    size_t dst_layer_index = deps[5];
    build_memory_primitive(mkldnn_primitives, rnn_desc.data.dst_layer_desc, dst_layer_index);
    size_t dst_iter_index = deps[6];
    build_memory_primitive(mkldnn_primitives, rnn_desc.data.dst_iter_desc, dst_iter_index);

    auto rnn_layer_prim_desc =
        mkldnn::rnn_forward::primitive_desc(rnn_desc, executor::global_cpu_engine);
    size_t workspace_index = deps[7];
    build_memory_primitive(
        mkldnn_primitives, rnn_layer_prim_desc.workspace_primitive_desc().desc(), workspace_index);
    auto workspace = std::unique_ptr<MKLDNNWorkspace>(
        new MKLDNNWorkspace(rnn_layer_prim_desc.workspace_primitive_desc().get_size()));
    auto workspace_buf_index = insert_workspace(mkldnn_workspaces, workspace);
    deps[8] = workspace_buf_index;

    mkldnn_primitives[rnn_index] =
        new mkldnn::rnn_forward(rnn_layer_prim_desc,
                                mkldnn::primitive::at(*mkldnn_primitives[src_layer_index]),
                                mkldnn::primitive::at(*mkldnn_primitives[src_iter_index]),
                                mkldnn::primitive::at(*mkldnn_primitives[weights_layer_index]),
                                mkldnn::primitive::at(*mkldnn_primitives[weights_iter_index]),
                                mkldnn::primitive::at(*mkldnn_primitives[bias_index]),
                                static_cast<mkldnn::memory>(*mkldnn_primitives[dst_layer_index]),
                                static_cast<mkldnn::memory>(*mkldnn_primitives[dst_iter_index]),
                                static_cast<mkldnn::memory>(*mkldnn_primitives[workspace_index]));
}

void MKLDNNEmitter::build_concat(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                 const mkldnn::concat::primitive_desc& concat_pd,
                                 const std::vector<mkldnn::memory::desc>& inputs_data_desc,
                                 const std::vector<size_t>& deps,
                                 size_t concat_index)
{
    std::vector<mkldnn::memory::primitive::at> inputs_primitive;
    std::vector<mkldnn::memory::primitive_desc> inputs_pd;

    for (size_t i = 0; i < inputs_data_desc.size(); i++)
    {
        inputs_pd.push_back(mkldnn::memory::primitive_desc(
            inputs_data_desc[i], runtime::cpu::executor::global_cpu_engine));
    }

    for (size_t i = 0; i < inputs_data_desc.size(); i++)
    {
        size_t inputs_data_index = deps[i];
        build_memory_primitive(mkldnn_primitives, inputs_data_desc[i], inputs_data_index);
        inputs_primitive.push_back(*mkldnn_primitives[inputs_data_index]);
    }
    size_t result_index = deps[inputs_data_desc.size()];
    build_memory_primitive(mkldnn_primitives, concat_pd.dst_primitive_desc().desc(), result_index);

    // concat primitive
    mkldnn_primitives[concat_index] =
        new mkldnn::concat(concat_pd, inputs_primitive, *mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_slice(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                const mkldnn::memory::desc& input_desc,
                                const mkldnn::memory::desc& result_desc,
                                const ngraph::Coordinate& lower_bounds,
                                const ngraph::Shape& result_shape,
                                const std::vector<size_t>& deps,
                                size_t slice_index)
{
    std::vector<size_t> in_out_index;
    mkldnn::memory::primitive_desc input_pd =
        mkldnn::memory::primitive_desc(input_desc, runtime::cpu::executor::global_cpu_engine);
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, input_desc, input_index);

    auto dims = mkldnn::memory::dims(result_shape.begin(), result_shape.end());
    auto offsets = mkldnn::memory::dims(lower_bounds.begin(), lower_bounds.end());
    auto view_pd = mkldnn::view::primitive_desc(input_pd, dims, offsets).dst_primitive_desc();

    mkldnn::memory::primitive_desc result_pd =
        mkldnn::memory::primitive_desc(result_desc, runtime::cpu::executor::global_cpu_engine);
    size_t result_index = deps[1];
    build_memory_primitive(mkldnn_primitives, result_desc, result_index);

    // reorder primitive descriptor
    mkldnn::reorder::primitive_desc reorder_pd =
        mkldnn::reorder::primitive_desc(view_pd, result_pd);
    // reorder primitive
    mkldnn_primitives[slice_index] = new mkldnn::reorder(
        reorder_pd, *mkldnn_primitives[input_index], *mkldnn_primitives[result_index]);
}

mkldnn::softmax_forward::desc MKLDNNEmitter::get_softmax_forward_desc(const ngraph::Node* node)
{
    auto softmax = static_cast<const ngraph::op::Softmax*>(node);

    auto axes = softmax->get_axes();
    if (axes.size() != 1)
    {
        throw ngraph_error("MKLDNN supports softmax only across single axis");
    }
    int softmax_axis = static_cast<int>(*(axes.begin()));

    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);

    return mkldnn::softmax_forward::desc(
        mkldnn::prop_kind::forward_scoring, input_desc, softmax_axis);
}

void MKLDNNEmitter::build_softmax_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                          const mkldnn::softmax_forward::desc& softmax_desc,
                                          const std::vector<size_t>& deps,
                                          size_t softmax_index)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, softmax_desc.data.data_desc, input_index);
    size_t result_index = deps[1];
    build_memory_primitive(mkldnn_primitives, softmax_desc.data.data_desc, result_index);

    mkldnn_primitives[softmax_index] =
        new mkldnn::softmax_forward({softmax_desc, executor::global_cpu_engine},
                                    *mkldnn_primitives[input_index],
                                    *mkldnn_primitives[result_index]);
}

mkldnn::eltwise_forward::desc MKLDNNEmitter::get_leaky_relu_desc(const ngraph::Node* node)
{
    auto alpha = static_cast<const ngraph::op::CPULeakyRelu*>(node)->get_alpha();

    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);

    return mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward_training,
                                         mkldnn::algorithm::eltwise_relu,
                                         input_desc,
                                         alpha,
                                         0.0f);
}

void MKLDNNEmitter::build_leaky_relu(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                     const mkldnn::eltwise_forward::desc& leaky_relu_desc,
                                     const std::vector<size_t>& deps,
                                     size_t leaky_relu_index)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, leaky_relu_desc.data.data_desc, input_index);
    size_t result_index = deps[1];
    build_memory_primitive(mkldnn_primitives, leaky_relu_desc.data.data_desc, result_index);

    mkldnn_primitives[leaky_relu_index] =
        new mkldnn::eltwise_forward({leaky_relu_desc, executor::global_cpu_engine},
                                    *mkldnn_primitives[input_index],
                                    *mkldnn_primitives[result_index]);
}

mkldnn::eltwise_forward::desc MKLDNNEmitter::get_bounded_relu_desc(const ngraph::Node* node)
{
    auto alpha = static_cast<const ngraph::op::BoundedRelu*>(node)->get_alpha();

    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);

    return mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward_training,
                                         mkldnn::algorithm::eltwise_bounded_relu,
                                         input_desc,
                                         alpha,
                                         0.0f);
}

void MKLDNNEmitter::build_bounded_relu(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                       const mkldnn::eltwise_forward::desc& bounded_relu_desc,
                                       const std::vector<size_t>& deps,
                                       size_t bounded_relu_index)
{
    size_t input_index = deps[0];
    build_memory_primitive(mkldnn_primitives, bounded_relu_desc.data.data_desc, input_index);
    size_t result_index = deps[1];
    build_memory_primitive(mkldnn_primitives, bounded_relu_desc.data.data_desc, result_index);

    mkldnn_primitives[bounded_relu_index] =
        new mkldnn::eltwise_forward({bounded_relu_desc, executor::global_cpu_engine},
                                    *mkldnn_primitives[input_index],
                                    *mkldnn_primitives[result_index]);
}

size_t MKLDNNEmitter::convolution_forward_init(bool with_bias)
{
    size_t size = m_mkldnn_primitives.size();
    if (with_bias)
    {
        // Inputs, Weights, Bias, Results, Conv
        m_mkldnn_primitives.resize(size + 5, nullptr);
        m_primitive_deps[m_mkldnn_primitives.size() - 1] = {size, size + 1, size + 2, size + 3};
    }
    else
    {
        // Inputs, Weights, Results, Conv
        m_mkldnn_primitives.resize(size + 4, nullptr);
        m_primitive_deps[m_mkldnn_primitives.size() - 1] = {size, size + 1, size + 2};
    }
    return m_mkldnn_primitives.size() - 1;
}

size_t MKLDNNEmitter::inner_product_forward_init(bool with_bias)
{
    size_t size = m_mkldnn_primitives.size();
    if (with_bias)
    {
        // Inputs, Weights, Bias, Results, inner_product
        m_mkldnn_primitives.resize(size + 5, nullptr);
        m_primitive_deps[m_mkldnn_primitives.size() - 1] = {size, size + 1, size + 2, size + 3};
    }
    else
    {
        // Inputs, Weights, Results, inner_product
        m_mkldnn_primitives.resize(size + 4, nullptr);
        m_primitive_deps[m_mkldnn_primitives.size() - 1] = {size, size + 1, size + 2};
    }
    return m_mkldnn_primitives.size() - 1;
}

size_t MKLDNNEmitter::reserve_primitive_space(size_t count, bool new_workspace)
{
    size_t size = m_mkldnn_primitives.size();
    m_mkldnn_primitives.resize(size + count, nullptr);
    for (auto i = 0; i < count - 1; i++)
    {
        m_primitive_deps[m_mkldnn_primitives.size() - 1].push_back(size + i);
    }
    if (new_workspace)
    {
        m_primitive_deps[m_mkldnn_primitives.size() - 1].push_back(0);
    }
    return m_mkldnn_primitives.size() - 1;
}

size_t MKLDNNEmitter::reserve_primitive_space_cg(size_t count, bool new_workspace)
{
    size_t size = m_mkldnn_primitives_cg.size();
    m_mkldnn_primitives_cg.resize(size + count, nullptr);
    for (auto i = 0; i < count - 1; i++)
    {
        m_primitive_deps_cg[m_mkldnn_primitives_cg.size() - 1].push_back(size + i);
    }
    if (new_workspace)
    {
        m_primitive_deps_cg[m_mkldnn_primitives_cg.size() - 1].push_back(0);
    }
    return m_mkldnn_primitives_cg.size() - 1;
}

size_t MKLDNNEmitter::build_quantized_inner_product_forward(
    const mkldnn::memory::desc& input_data_desc,
    const mkldnn::memory::desc& weights_desc,
    const mkldnn::memory::desc& bias_desc,
    const mkldnn::memory::desc& result_desc,
    const float scale,
    const mkldnn::post_ops& pops)
{
    size_t input_data_index = build_memory_primitive(input_data_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t bias_index = build_memory_primitive(bias_desc);
    size_t result_index = build_memory_primitive(result_desc);
    std::vector<float> output_scale;
    output_scale.push_back(scale);
    // mkldnn inner_product attr
    mkldnn::primitive_attr ip_attr;
    ip_attr.set_post_ops(pops);
    /* Specify the rounding mode */
    ip_attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
    /* Specify the scales array and corresponding mask */
    ip_attr.set_output_scales(0, output_scale);
    // mkldnn inner_product
    size_t ip_index =
        insert_primitive(new mkldnn::inner_product_forward({{
                                                                mkldnn::prop_kind::forward_scoring,
                                                                input_data_desc,
                                                                weights_desc,
                                                                bias_desc,
                                                                result_desc,
                                                            },
                                                            ip_attr,
                                                            executor::global_cpu_engine},
                                                           *m_mkldnn_primitives[input_data_index],
                                                           *m_mkldnn_primitives[weights_index],
                                                           *m_mkldnn_primitives[bias_index],
                                                           *m_mkldnn_primitives[result_index]));
    m_primitive_deps[ip_index] = {input_data_index, weights_index, bias_index, result_index};
    return ip_index;
}

size_t MKLDNNEmitter::build_quantized_inner_product_forward(
    const mkldnn::memory::desc& input_data_desc,
    const mkldnn::memory::desc& weights_desc,
    const mkldnn::memory::desc& result_desc,
    const float scale,
    const mkldnn::post_ops& pops)
{
    size_t input_data_index = build_memory_primitive(input_data_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t result_index = build_memory_primitive(result_desc);
    std::vector<float> output_scale;
    output_scale.push_back(scale);
    // mkldnn inner_product attr
    mkldnn::primitive_attr ip_attr;
    ip_attr.set_post_ops(pops);
    /* Specify the rounding mode */
    ip_attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
    /* Specify the scales array and corresponding mask */
    ip_attr.set_output_scales(0, output_scale);
    // mkldnn inner_product
    size_t ip_index = insert_primitive(new mkldnn::inner_product_forward(
        {{
             mkldnn::prop_kind::forward_scoring, input_data_desc, weights_desc, result_desc,
         },
         ip_attr,
         executor::global_cpu_engine},
        *m_mkldnn_primitives[input_data_index],
        *m_mkldnn_primitives[weights_index],
        *m_mkldnn_primitives[result_index]));
    m_primitive_deps[ip_index] = {input_data_index, weights_index, result_index};
    return ip_index;
}
