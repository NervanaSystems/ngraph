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
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_concat.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
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

const std::vector<mkldnn::primitive*>& MKLDNNEmitter::get_mkldnn_primitives() const
{
    return m_mkldnn_primitives;
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

const std::vector<size_t>& MKLDNNEmitter::get_primitive_deps(size_t index) const
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

void MKLDNNEmitter::build_quantize_reorder(const mkldnn::memory::desc& input_desc,
                                           const mkldnn::memory::desc& result_desc,
                                           const std::vector<float>& scales,
                                           const Node* node)
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

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {input_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
}

void MKLDNNEmitter::build_quantize_reorder(const mkldnn::memory::desc& input_desc,
                                           const mkldnn::memory::desc& result_desc,
                                           const std::vector<float>& scales,
                                           size_t quantize_index,
                                           const int mask)
{
    size_t input_index = m_primitive_deps[quantize_index][0];
    build_memory_primitive(input_desc, input_index);
    size_t result_index = m_primitive_deps[quantize_index][1];
    build_memory_primitive(result_desc, result_index);

    mkldnn::primitive_attr attr;
    attr.set_output_scales(mask, scales);
    attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
    auto reorder_desc = mkldnn::reorder::primitive_desc({input_desc, executor::global_cpu_engine},
                                                        {result_desc, executor::global_cpu_engine},
                                                        attr);
    m_mkldnn_primitives[quantize_index] = new mkldnn::reorder(
        reorder_desc, *m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_dequantization(const ngraph::Node* node,
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

    build_quantize_reorder(input_desc, result_desc, scales, node);
}

void MKLDNNEmitter::build_quantized_max_pool(const ngraph::Node* node)
{
    auto qmax_pool = static_cast<const ngraph::op::QuantizedMaxPool*>(node);
    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

    this->build_pooling_forward(mkldnn::algorithm::pooling_max,
                                input_desc,
                                result_desc,
                                qmax_pool->get_window_movement_strides(),
                                qmax_pool->get_window_shape(),
                                qmax_pool->get_padding_below(),
                                qmax_pool->get_padding_above(),
                                node);
}

void MKLDNNEmitter::build_quantized_avg_pool(const ngraph::Node* node)
{
    auto qavg_pool = static_cast<const ngraph::op::QuantizedAvgPool*>(node);
    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

    this->build_pooling_forward((qavg_pool->get_include_padding_in_avg_computation()
                                     ? mkldnn::algorithm::pooling_avg_include_padding
                                     : mkldnn::algorithm::pooling_avg_exclude_padding),
                                input_desc,
                                result_desc,
                                qavg_pool->get_window_movement_strides(),
                                qavg_pool->get_window_shape(),
                                qavg_pool->get_padding_below(),
                                qavg_pool->get_padding_above(),
                                node);
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
        mkldnn::prop_kind::forward,
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

void MKLDNNEmitter::build_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                              const mkldnn::memory::desc& weights_desc,
                                              const mkldnn::memory::desc& result_desc,
                                              const ngraph::Strides& strides,
                                              const ngraph::Strides& dilation_strides,
                                              const ngraph::CoordinateDiff& padding_below,
                                              const ngraph::CoordinateDiff& padding_above,
                                              const Node* node,
                                              const mkldnn::post_ops& pops)
{
    size_t input_data_index = build_memory_primitive(input_data_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t result_index = build_memory_primitive(result_desc);

    mkldnn::primitive_attr conv_attr;
    conv_attr.set_post_ops(pops);
    mkldnn::algorithm convolution_algo = mkldnn_utils::get_conv_algo();
    size_t conv_index = 0;
    try
    {
        auto conv_prim = new mkldnn::convolution_forward(
            {{mkldnn::prop_kind::forward,
              convolution_algo,
              input_data_desc,
              weights_desc,
              result_desc,
              mkldnn::memory::dims(strides.begin(), strides.end()),
              mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
              mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
              mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
              mkldnn::padding_kind::zero},

             conv_attr,
             executor::global_cpu_engine},
            *m_mkldnn_primitives[input_data_index],
            *m_mkldnn_primitives[weights_index],
            *m_mkldnn_primitives[result_index]);

        conv_index = insert_primitive(conv_prim);

        NGRAPH_ASSERT(m_primitive_deps.find(conv_index) == m_primitive_deps.end() &&
                      m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
            << "Primitive already created for node " << node->description();

        m_primitive_deps[conv_index] = {input_data_index, weights_index, result_index};
        m_node_primitive_idx_map[node] = conv_index;
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not create mkldnn convolution " + e.message);
    }
}

void MKLDNNEmitter::build_quantized_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                                        const mkldnn::memory::desc& weights_desc,
                                                        const mkldnn::memory::desc& result_desc,
                                                        const ngraph::Strides& strides,
                                                        const ngraph::Strides& dilation_strides,
                                                        const ngraph::CoordinateDiff& padding_below,
                                                        const ngraph::CoordinateDiff& padding_above,
                                                        const float scale,
                                                        const Node* node,
                                                        const mkldnn::post_ops& pops)
{
    size_t input_data_index = build_memory_primitive(input_data_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t result_index = build_memory_primitive(result_desc);
    std::vector<float> output_scale;
    output_scale.push_back(scale);
    mkldnn::primitive_attr conv_attr;
    conv_attr.set_post_ops(pops);
    /* Specify the rounding mode */
    conv_attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
    /* Specify the scales array and corresponding mask */
    conv_attr.set_output_scales(0, output_scale);
    mkldnn::algorithm convolution_algo = mkldnn_utils::get_conv_algo();
    size_t conv_index = insert_primitive(new mkldnn::convolution_forward(
        {{mkldnn::prop_kind::forward,
          convolution_algo,
          input_data_desc,
          weights_desc,
          result_desc,
          mkldnn::memory::dims(strides.begin(), strides.end()),
          mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
          mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
          mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
          mkldnn::padding_kind::zero},
         conv_attr,
         executor::global_cpu_engine},
        *m_mkldnn_primitives[input_data_index],
        *m_mkldnn_primitives[weights_index],
        *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(conv_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[conv_index] = {input_data_index, weights_index, result_index};
    m_node_primitive_idx_map[node] = conv_index;
}

void MKLDNNEmitter::build_quantized_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                                        const mkldnn::memory::desc& weights_desc,
                                                        const mkldnn::memory::desc& bias_desc,
                                                        const mkldnn::memory::desc& result_desc,
                                                        const ngraph::Strides& strides,
                                                        const ngraph::Strides& dilation_strides,
                                                        const ngraph::CoordinateDiff& padding_below,
                                                        const ngraph::CoordinateDiff& padding_above,
                                                        const float scale,
                                                        const Node* node,
                                                        const mkldnn::post_ops& pops)
{
    size_t input_data_index = build_memory_primitive(input_data_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t bias_index = build_memory_primitive(bias_desc);
    size_t result_index = build_memory_primitive(result_desc);
    std::vector<float> output_scale;
    output_scale.push_back(scale);
    mkldnn::primitive_attr conv_attr;
    conv_attr.set_post_ops(pops);
    /* Specify the rounding mode */
    conv_attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
    /* Specify the scales array and corresponding mask */
    conv_attr.set_output_scales(0, output_scale);
    mkldnn::algorithm convolution_algo = mkldnn_utils::get_conv_algo();
    size_t conv_index = insert_primitive(new mkldnn::convolution_forward(
        {{mkldnn::prop_kind::forward,
          convolution_algo,
          input_data_desc,
          weights_desc,
          bias_desc,
          result_desc,
          mkldnn::memory::dims(strides.begin(), strides.end()),
          mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
          mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
          mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
          mkldnn::padding_kind::zero},
         conv_attr,
         executor::global_cpu_engine},
        *m_mkldnn_primitives[input_data_index],
        *m_mkldnn_primitives[weights_index],
        *m_mkldnn_primitives[bias_index],
        *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(conv_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[conv_index] = {input_data_index, weights_index, bias_index, result_index};
    m_node_primitive_idx_map[node] = conv_index;
}

void MKLDNNEmitter::build_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                              const mkldnn::memory::desc& weights_desc,
                                              const mkldnn::memory::desc& bias_desc,
                                              const mkldnn::memory::desc& result_desc,
                                              const ngraph::Strides& strides,
                                              const ngraph::Strides& dilation_strides,
                                              const ngraph::CoordinateDiff& padding_below,
                                              const ngraph::CoordinateDiff& padding_above,
                                              const Node* node,
                                              const mkldnn::post_ops& pops)
{
    const size_t input_data_index = build_memory_primitive(input_data_desc);
    const size_t weights_index = build_memory_primitive(weights_desc);
    const size_t bias_index = build_memory_primitive(bias_desc);
    const size_t result_index = build_memory_primitive(result_desc);

    mkldnn::primitive_attr conv_attr;
    conv_attr.set_post_ops(pops);

    mkldnn::algorithm convolution_algo = mkldnn_utils::get_conv_algo();
    size_t conv_index = -1;
    try
    {
        conv_index = insert_primitive(new mkldnn::convolution_forward(
            {{mkldnn::prop_kind::forward,
              convolution_algo,
              input_data_desc,
              weights_desc,
              bias_desc,
              result_desc,
              mkldnn::memory::dims(strides.begin(), strides.end()),
              mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
              mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
              mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
              mkldnn::padding_kind::zero},
             conv_attr,
             executor::global_cpu_engine},
            *m_mkldnn_primitives[input_data_index],
            *m_mkldnn_primitives[weights_index],
            *m_mkldnn_primitives[bias_index],
            *m_mkldnn_primitives[result_index]));

        NGRAPH_ASSERT(m_primitive_deps.find(conv_index) == m_primitive_deps.end() &&
                      m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
            << "Primitive already created for node " << node->description();

        m_primitive_deps[conv_index] = {input_data_index, weights_index, bias_index, result_index};
        m_node_primitive_idx_map[node] = conv_index;
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not create convolution " + e.message);
    }
}

void MKLDNNEmitter::build_convolution_backward_weights_bias(
    const mkldnn::memory::desc& in_data_desc,
    const mkldnn::memory::desc& in_delta_desc,
    const mkldnn::memory::desc& out_weights_delta_desc,
    const mkldnn::memory::desc& out_bias_delta_desc,
    const ngraph::Strides& ng_strides,
    const ngraph::Strides& ng_dilation_strides,
    const ngraph::CoordinateDiff& ng_padding_below,
    const ngraph::CoordinateDiff& ng_padding_above,
    const Node* node)
{
    const size_t in_data_index = build_memory_primitive(in_data_desc);
    const size_t in_delta_index = build_memory_primitive(in_delta_desc);
    const size_t out_weights_delta_index = build_memory_primitive(out_weights_delta_desc);
    const size_t out_bias_delta_index = build_memory_primitive(out_bias_delta_desc);

    mkldnn::memory::dims strides(ng_strides.begin(), ng_strides.end());
    mkldnn::memory::dims dilation(ng_dilation_strides.begin(), ng_dilation_strides.end());
    mkldnn::memory::dims padding_l(ng_padding_below.begin(), ng_padding_below.end());
    mkldnn::memory::dims padding_r(ng_padding_above.begin(), ng_padding_above.end());
    mkldnn::algorithm convolution_algo = mkldnn_utils::get_conv_algo();
    mkldnn::convolution_forward::primitive_desc fwd_pd{{mkldnn::prop_kind::forward,
                                                        convolution_algo,
                                                        in_data_desc,
                                                        out_weights_delta_desc,
                                                        out_bias_delta_desc,
                                                        in_delta_desc,
                                                        strides,
                                                        dilation,
                                                        padding_l,
                                                        padding_r,
                                                        mkldnn::padding_kind::zero},
                                                       executor::global_cpu_engine};

    mkldnn::convolution_backward_weights::primitive_desc bwd_pd{{convolution_algo,
                                                                 in_data_desc,
                                                                 out_weights_delta_desc,
                                                                 out_bias_delta_desc,
                                                                 in_delta_desc,
                                                                 strides,
                                                                 dilation,
                                                                 padding_l,
                                                                 padding_r,
                                                                 mkldnn::padding_kind::zero},
                                                                executor::global_cpu_engine,
                                                                fwd_pd};

    const size_t conv_index = insert_primitive(
        new mkldnn::convolution_backward_weights(bwd_pd,
                                                 *m_mkldnn_primitives[in_data_index],
                                                 *m_mkldnn_primitives[in_delta_index],
                                                 *m_mkldnn_primitives[out_weights_delta_index],
                                                 *m_mkldnn_primitives[out_bias_delta_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(conv_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[conv_index] = {
        in_data_index, in_delta_index, out_weights_delta_index, out_bias_delta_index};
    m_node_primitive_idx_map[node] = conv_index;
}

void MKLDNNEmitter::build_convolution_backward_weights_bias(
    const mkldnn::convolution_backward_weights::desc& bwd_desc,
    const mkldnn::convolution_forward::desc& fwd_desc,
    size_t conv_index)
{
    size_t in_data_index = m_primitive_deps[conv_index][0];
    build_memory_primitive(bwd_desc.data.src_desc, in_data_index);
    size_t in_delta_index = m_primitive_deps[conv_index][1];
    build_memory_primitive(bwd_desc.data.diff_dst_desc, in_delta_index);
    size_t out_weights_delta_index = m_primitive_deps[conv_index][2];
    build_memory_primitive(bwd_desc.data.diff_weights_desc, out_weights_delta_index);
    size_t out_bias_delta_index = m_primitive_deps[conv_index][3];
    build_memory_primitive(bwd_desc.data.diff_bias_desc, out_bias_delta_index);

    mkldnn::convolution_forward::primitive_desc fwd_pd{fwd_desc, executor::global_cpu_engine};

    mkldnn::convolution_backward_weights::primitive_desc bwd_pd{
        bwd_desc, executor::global_cpu_engine, fwd_pd};

    m_mkldnn_primitives[conv_index] =
        new mkldnn::convolution_backward_weights(bwd_pd,
                                                 *m_mkldnn_primitives[in_data_index],
                                                 *m_mkldnn_primitives[in_delta_index],
                                                 *m_mkldnn_primitives[out_weights_delta_index],
                                                 *m_mkldnn_primitives[out_bias_delta_index]);
}

void MKLDNNEmitter::build_convolution_backward_weights(const mkldnn::memory::desc& input_desc,
                                                       const mkldnn::memory::desc& delta_desc,
                                                       const mkldnn::memory::desc& result_desc,
                                                       const ngraph::Strides& strides,
                                                       const ngraph::Strides& dilation_strides,
                                                       const ngraph::CoordinateDiff& padding_below,
                                                       const ngraph::CoordinateDiff& padding_above,
                                                       const Node* node)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t delta_index = build_memory_primitive(delta_desc);
    size_t result_index = build_memory_primitive(result_desc);

    mkldnn::algorithm convolution_algo = mkldnn_utils::get_conv_algo();
    size_t primitive_index = insert_primitive(new mkldnn::convolution_backward_weights(
        {{convolution_algo,
          input_desc,
          result_desc,
          delta_desc,
          mkldnn::memory::dims(strides.begin(), strides.end()),
          mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
          mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
          mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
          mkldnn::padding_kind::zero},
         executor::global_cpu_engine,
         // Forward primitive descriptor corresponding to this backward weights descriptor
         {{mkldnn::prop_kind::forward,
           convolution_algo,
           input_desc,
           result_desc,
           delta_desc,
           mkldnn::memory::dims(strides.begin(), strides.end()),
           mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
           mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
           mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
           mkldnn::padding_kind::zero},
          executor::global_cpu_engine}},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[delta_index],
        *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {input_index, delta_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
}

void MKLDNNEmitter::build_convolution_backward_weights(
    const mkldnn::convolution_backward_weights::desc& bwd_desc,
    const mkldnn::convolution_forward::desc& fwd_desc,
    size_t conv_index)
{
    size_t in_data_index = m_primitive_deps[conv_index][0];
    build_memory_primitive(bwd_desc.data.src_desc, in_data_index);
    size_t in_delta_index = m_primitive_deps[conv_index][1];
    build_memory_primitive(bwd_desc.data.diff_dst_desc, in_delta_index);
    size_t out_weights_delta_index = m_primitive_deps[conv_index][2];
    build_memory_primitive(bwd_desc.data.diff_weights_desc, out_weights_delta_index);

    m_mkldnn_primitives[conv_index] = new mkldnn::convolution_backward_weights(
        {bwd_desc,
         executor::global_cpu_engine,
         // Forward primitive descriptor corresponding to this backward weights descriptor
         {fwd_desc, executor::global_cpu_engine}},
        *m_mkldnn_primitives[in_data_index],
        *m_mkldnn_primitives[in_delta_index],
        *m_mkldnn_primitives[out_weights_delta_index]);
}

void MKLDNNEmitter::build_convolution_backward_data(const mkldnn::memory::desc& weights_desc,
                                                    const mkldnn::memory::desc& delta_desc,
                                                    const mkldnn::memory::desc& result_desc,
                                                    const ngraph::Strides& strides,
                                                    const ngraph::Strides& dilation_strides,
                                                    const ngraph::CoordinateDiff& padding_below,
                                                    const ngraph::CoordinateDiff& padding_above,
                                                    const Node* node)
{
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t delta_index = build_memory_primitive(delta_desc);
    size_t result_index = build_memory_primitive(result_desc);

    mkldnn::algorithm convolution_algo = mkldnn_utils::get_conv_algo();
    size_t primitive_index = insert_primitive(new mkldnn::convolution_backward_data(
        {{convolution_algo,
          result_desc,
          weights_desc,
          delta_desc,
          mkldnn::memory::dims(strides.begin(), strides.end()),
          mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
          mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
          mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
          mkldnn::padding_kind::zero},
         executor::global_cpu_engine,
         // Forward primitive descriptor corresponding to this backward data descriptor
         {{mkldnn::prop_kind::forward,
           convolution_algo,
           result_desc,
           weights_desc,
           delta_desc,
           mkldnn::memory::dims(strides.begin(), strides.end()),
           mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
           mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
           mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
           mkldnn::padding_kind::zero},
          executor::global_cpu_engine}},
        *m_mkldnn_primitives[delta_index],
        *m_mkldnn_primitives[weights_index],
        *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {weights_index, delta_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
}

void MKLDNNEmitter::build_convolution_backward_data(
    const mkldnn::convolution_backward_data::desc& bwd_desc,
    const mkldnn::convolution_forward::desc& fwd_desc,
    size_t conv_index)
{
    size_t weights_index = m_primitive_deps[conv_index][0];
    build_memory_primitive(bwd_desc.data.weights_desc, weights_index);
    size_t delta_index = m_primitive_deps[conv_index][1];
    build_memory_primitive(bwd_desc.data.diff_dst_desc, delta_index);
    size_t result_index = m_primitive_deps[conv_index][2];
    build_memory_primitive(bwd_desc.data.diff_src_desc, result_index);

    m_mkldnn_primitives[conv_index] = new mkldnn::convolution_backward_data(
        {bwd_desc,
         executor::global_cpu_engine,
         // Forward primitive descriptor corresponding to this backward data descriptor
         {fwd_desc, executor::global_cpu_engine}},
        *m_mkldnn_primitives[delta_index],
        *m_mkldnn_primitives[weights_index],
        *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_pooling_forward(mkldnn::algorithm pooling_algorithm,
                                          const mkldnn::memory::desc& input_desc,
                                          const mkldnn::memory::desc& result_desc,
                                          const ngraph::Strides& window_strides,
                                          const ngraph::Shape& window_shape,
                                          const ngraph::Shape& padding_below,
                                          const ngraph::Shape& padding_above,
                                          const Node* node)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index = insert_primitive(new mkldnn::pooling_forward(
        {{mkldnn::prop_kind::forward_inference,
          pooling_algorithm,
          input_desc,
          result_desc,
          mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
          mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
          mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
          mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
          mkldnn::padding_kind::zero},
         executor::global_cpu_engine},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {input_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
}

void MKLDNNEmitter::build_pooling_forward(const mkldnn::pooling_forward::desc& pool_desc,
                                          size_t pool_index)
{
    size_t input_index = m_primitive_deps[pool_index][0];
    build_memory_primitive(pool_desc.data.src_desc, input_index);
    size_t result_index = m_primitive_deps[pool_index][1];
    build_memory_primitive(pool_desc.data.dst_desc, result_index);

    m_mkldnn_primitives[pool_index] =
        new mkldnn::pooling_forward({pool_desc, executor::global_cpu_engine},
                                    *m_mkldnn_primitives[input_index],
                                    *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_pooling_backward(mkldnn::algorithm pooling_algorithm,
                                           const mkldnn::memory::desc& diff_dst_desc,
                                           const mkldnn::memory::desc& diff_src_desc,
                                           const ngraph::Strides& window_strides,
                                           const ngraph::Shape& window_shape,
                                           const ngraph::Shape& padding_below,
                                           const ngraph::Shape& padding_above,
                                           const Node* node)
{
    size_t input_index = build_memory_primitive(diff_dst_desc);
    size_t result_index = build_memory_primitive(diff_src_desc);

    size_t primitive_index = insert_primitive(new mkldnn::pooling_backward(
        {{pooling_algorithm,
          diff_src_desc,
          diff_dst_desc,
          mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
          mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
          mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
          mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
          mkldnn::padding_kind::zero},
         executor::global_cpu_engine,
         {{mkldnn::prop_kind::forward_training,
           pooling_algorithm,
           diff_src_desc,
           diff_dst_desc,
           mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
           mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
           mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
           mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
           mkldnn::padding_kind::zero},
          executor::global_cpu_engine}},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {input_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
}

void MKLDNNEmitter::build_pooling_backward(const mkldnn::pooling_backward::desc& pool_desc,
                                           const mkldnn::pooling_forward::desc& pool_fwd_desc,
                                           size_t pool_index)
{
    size_t input_index = m_primitive_deps[pool_index][0];
    build_memory_primitive(pool_desc.data.diff_dst_desc, input_index);
    size_t result_index = m_primitive_deps[pool_index][1];
    build_memory_primitive(pool_desc.data.diff_src_desc, result_index);

    auto pool_fwd_pd =
        mkldnn::pooling_forward::primitive_desc(pool_fwd_desc, executor::global_cpu_engine);
    auto pool_pd = mkldnn::pooling_backward::primitive_desc(
        pool_desc, executor::global_cpu_engine, pool_fwd_pd);

    m_mkldnn_primitives[pool_index] = new mkldnn::pooling_backward(
        pool_pd, *m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_max_pooling_backward(mkldnn::algorithm pooling_algorithm,
                                               const mkldnn::memory::desc& fprop_src_desc,
                                               const mkldnn::memory::desc& diff_dst_desc,
                                               const mkldnn::memory::desc& diff_src_desc,
                                               const ngraph::Strides& window_strides,
                                               const ngraph::Shape& window_shape,
                                               const ngraph::Shape& padding_below,
                                               const ngraph::Shape& padding_above,
                                               const Node* node)
{
    size_t fprop_src_index = build_memory_primitive(fprop_src_desc);
    size_t diff_dst_index = build_memory_primitive(diff_dst_desc);
    size_t diff_src_index = build_memory_primitive(diff_src_desc);

    mkldnn::pooling_forward::primitive_desc fwd_pd{
        {mkldnn::prop_kind::forward_training,
         pooling_algorithm,
         diff_src_desc,
         diff_dst_desc,
         mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
         mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
         mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
         mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
         mkldnn::padding_kind::zero},
        executor::global_cpu_engine};

    auto ws_index = build_memory_primitive(fwd_pd.workspace_primitive_desc().desc());
    // Allocate workspace
    // TODO (jbobba): Might need to align memory
    auto ws = std::unique_ptr<MKLDNNWorkspace>(
        new MKLDNNWorkspace(fwd_pd.workspace_primitive_desc().get_size()));
    auto ws_buf_index = insert_workspace(ws);

    size_t fwd_primitive_index = insert_primitive(new mkldnn::pooling_forward(
        fwd_pd,
        *m_mkldnn_primitives[fprop_src_index],
        *m_mkldnn_primitives
            [diff_src_index], // HACK - Uses diff_src buffer. Safe since diff_src > fprop_dst
        *m_mkldnn_primitives[ws_index]));

    size_t bwd_primitive_index = insert_primitive(new mkldnn::pooling_backward(
        {{pooling_algorithm,
          diff_src_desc,
          diff_dst_desc,
          mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
          mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
          mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
          mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
          mkldnn::padding_kind::zero},
         executor::global_cpu_engine,
         fwd_pd},
        *m_mkldnn_primitives[diff_dst_index],
        *m_mkldnn_primitives[ws_index],
        *m_mkldnn_primitives[diff_src_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(fwd_primitive_index) == m_primitive_deps.end() &&
                  m_primitive_deps.find(bwd_primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[fwd_primitive_index] = {
        fprop_src_index, diff_src_index, ws_index, ws_buf_index};
    m_primitive_deps[bwd_primitive_index] = {
        diff_dst_index, ws_index, diff_src_index, ws_buf_index};

    m_node_primitive_idx_map[node] = bwd_primitive_index;
}

void MKLDNNEmitter::build_max_pooling_backward(const mkldnn::pooling_backward::desc& bwd_pool_desc,
                                               const mkldnn::pooling_forward::desc& fwd_pool_desc,
                                               const mkldnn::memory::desc& fprop_src_desc,
                                               size_t fwd_pool_index,
                                               size_t bwd_pool_index)
{
    size_t fprop_src_index = m_primitive_deps[fwd_pool_index][0];
    build_memory_primitive(fprop_src_desc, fprop_src_index);
    size_t diff_dst_index = m_primitive_deps[bwd_pool_index][0];
    build_memory_primitive(bwd_pool_desc.data.diff_dst_desc, diff_dst_index);
    size_t diff_src_index = m_primitive_deps[fwd_pool_index][1];
    build_memory_primitive(bwd_pool_desc.data.diff_src_desc, diff_src_index);
    m_primitive_deps[bwd_pool_index][2] = diff_src_index;

    mkldnn::pooling_forward::primitive_desc fwd_pd{fwd_pool_desc, executor::global_cpu_engine};

    size_t ws_index = m_primitive_deps[fwd_pool_index][2];
    build_memory_primitive(fwd_pd.workspace_primitive_desc().desc(), ws_index);
    m_primitive_deps[bwd_pool_index][1] = ws_index;

    // Allocate workspace
    // TODO (jbobba): Might need to align memory
    auto ws = std::unique_ptr<MKLDNNWorkspace>(
        new MKLDNNWorkspace(fwd_pd.workspace_primitive_desc().get_size()));
    auto ws_buf_index = insert_workspace(ws);
    m_primitive_deps[fwd_pool_index][3] = ws_buf_index;
    m_primitive_deps[bwd_pool_index][3] = ws_buf_index;

    m_mkldnn_primitives[fwd_pool_index] = new mkldnn::pooling_forward(
        fwd_pd,
        *m_mkldnn_primitives[fprop_src_index],
        *m_mkldnn_primitives
            [diff_src_index], // HACK - Uses diff_src buffer. Safe since diff_src > fprop_dst
        *m_mkldnn_primitives[ws_index]);

    m_mkldnn_primitives[bwd_pool_index] =
        new mkldnn::pooling_backward({bwd_pool_desc, executor::global_cpu_engine, fwd_pd},
                                     *m_mkldnn_primitives[diff_dst_index],
                                     *m_mkldnn_primitives[ws_index],
                                     *m_mkldnn_primitives[diff_src_index]);
}

void MKLDNNEmitter::build_max_pooling_with_indices_forward(mkldnn::algorithm pooling_algorithm,
                                                           const mkldnn::memory::desc& src_desc,
                                                           const mkldnn::memory::desc& dst_desc,
                                                           const ngraph::Strides& window_strides,
                                                           const ngraph::Shape& window_shape,
                                                           const ngraph::Shape& padding_below,
                                                           const ngraph::Shape& padding_above,
                                                           const Node* node)
{
    size_t src_index = build_memory_primitive(src_desc);
    size_t dst_index = build_memory_primitive(dst_desc);

    mkldnn::pooling_forward::primitive_desc fwd_pd{
        {mkldnn::prop_kind::forward_training,
         pooling_algorithm,
         src_desc,
         dst_desc,
         mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
         mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
         mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
         mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
         mkldnn::padding_kind::zero},
        executor::global_cpu_engine};

    auto ws_index = build_memory_primitive(fwd_pd.workspace_primitive_desc().desc());

    size_t fwd_primitive_index =
        insert_primitive(new mkldnn::pooling_forward(fwd_pd,
                                                     *m_mkldnn_primitives[src_index],
                                                     *m_mkldnn_primitives[dst_index],
                                                     *m_mkldnn_primitives[ws_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(fwd_primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[fwd_primitive_index] = {src_index, dst_index, ws_index};
    m_node_primitive_idx_map[node] = fwd_primitive_index;
}

void MKLDNNEmitter::build_max_pooling_with_indices_forward(
    const mkldnn::pooling_forward::desc& max_pool_desc, size_t max_pool_index)
{
    size_t src_index = m_primitive_deps[max_pool_index][0];
    build_memory_primitive(max_pool_desc.data.src_desc, src_index);
    size_t dst_index = m_primitive_deps[max_pool_index][1];
    build_memory_primitive(max_pool_desc.data.dst_desc, dst_index);

    mkldnn::pooling_forward::primitive_desc fwd_pd{max_pool_desc, executor::global_cpu_engine};

    size_t ws_index = m_primitive_deps[max_pool_index][2];
    build_memory_primitive(fwd_pd.workspace_primitive_desc().desc(), ws_index);

    m_mkldnn_primitives[max_pool_index] =
        new mkldnn::pooling_forward(fwd_pd,
                                    *m_mkldnn_primitives[src_index],
                                    *m_mkldnn_primitives[dst_index],
                                    *m_mkldnn_primitives[ws_index]);
}

void MKLDNNEmitter::build_max_pooling_with_indices_backward(
    mkldnn::algorithm pooling_algorithm,
    const mkldnn::memory::desc& diff_dst_desc,
    const mkldnn::memory::desc& diff_src_desc,
    const ngraph::Strides& window_strides,
    const ngraph::Shape& window_shape,
    const ngraph::Shape& padding_below,
    const ngraph::Shape& padding_above,
    const Node* node)
{
    size_t diff_dst_index = build_memory_primitive(diff_dst_desc);
    size_t diff_src_index = build_memory_primitive(diff_src_desc);

    mkldnn::pooling_forward::primitive_desc fwd_pd{
        {mkldnn::prop_kind::forward_training,
         pooling_algorithm,
         diff_src_desc,
         diff_dst_desc,
         mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
         mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
         mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
         mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
         mkldnn::padding_kind::zero},
        executor::global_cpu_engine};

    auto fprop_ws_index = build_memory_primitive(fwd_pd.workspace_primitive_desc().desc());

    size_t bwd_primitive_index = insert_primitive(new mkldnn::pooling_backward(
        {{pooling_algorithm,
          diff_src_desc,
          diff_dst_desc,
          mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
          mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
          mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
          mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
          mkldnn::padding_kind::zero},
         executor::global_cpu_engine,
         fwd_pd},
        *m_mkldnn_primitives[diff_dst_index],
        *m_mkldnn_primitives[fprop_ws_index],
        *m_mkldnn_primitives[diff_src_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(bwd_primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[bwd_primitive_index] = {diff_dst_index, fprop_ws_index, diff_src_index};
    m_node_primitive_idx_map[node] = bwd_primitive_index;
}

void MKLDNNEmitter::build_max_pooling_with_indices_backward(
    const mkldnn::pooling_backward::desc& bwd_pool_desc,
    const mkldnn::pooling_forward::desc& fwd_pool_desc,
    size_t max_pool_index)
{
    size_t diff_dst_index = m_primitive_deps[max_pool_index][0];
    build_memory_primitive(bwd_pool_desc.data.diff_dst_desc, diff_dst_index);
    size_t diff_src_index = m_primitive_deps[max_pool_index][2];
    build_memory_primitive(bwd_pool_desc.data.diff_src_desc, diff_src_index);

    mkldnn::pooling_forward::primitive_desc fwd_pd{fwd_pool_desc, executor::global_cpu_engine};

    size_t fprop_ws_index = m_primitive_deps[max_pool_index][1];
    build_memory_primitive(fwd_pd.workspace_primitive_desc().desc(), fprop_ws_index);

    m_mkldnn_primitives[max_pool_index] =
        new mkldnn::pooling_backward({bwd_pool_desc, executor::global_cpu_engine, fwd_pd},
                                     *m_mkldnn_primitives[diff_dst_index],
                                     *m_mkldnn_primitives[fprop_ws_index],
                                     *m_mkldnn_primitives[diff_src_index]);
}

void MKLDNNEmitter::build_reorder(const mkldnn::memory::desc& input_desc,
                                  const mkldnn::memory::desc& result_desc,
                                  const Node* node)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index = 0;
    try
    {
        primitive_index = insert_primitive(new mkldnn::reorder(*m_mkldnn_primitives[input_index],
                                                               *m_mkldnn_primitives[result_index]));

        NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                      m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
            << "Primitive already created for node " << node->description();

        m_primitive_deps[primitive_index] = {input_index, result_index};
        m_node_primitive_idx_map[node] = primitive_index;
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not create mkldnn primitive " + e.message);
    }
}

void MKLDNNEmitter::build_reorder(const mkldnn::memory::desc& input_desc,
                                  const mkldnn::memory::desc& result_desc,
                                  size_t reorder_index)
{
    size_t input_index = m_primitive_deps[reorder_index][0];
    build_memory_primitive(input_desc, input_index);
    size_t result_index = m_primitive_deps[reorder_index][1];
    build_memory_primitive(result_desc, result_index);

    m_mkldnn_primitives[reorder_index] =
        new mkldnn::reorder(*m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_lrn_forward(const mkldnn::memory::desc& input_desc,
                                      const mkldnn::memory::desc& result_desc,
                                      float alpha,
                                      float beta,
                                      float bias,
                                      int nsize,
                                      const Node* node)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    auto lrn_desc = mkldnn::lrn_forward::desc(mkldnn::prop_kind::forward_scoring,
                                              mkldnn::algorithm::lrn_across_channels,
                                              input_desc,
                                              nsize,
                                              alpha,
                                              beta,
                                              bias);
    auto lrn_prim_desc = mkldnn::lrn_forward::primitive_desc(lrn_desc, executor::global_cpu_engine);
    size_t primitive_index = insert_primitive(new mkldnn::lrn_forward(
        lrn_prim_desc, *m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {input_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
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

void MKLDNNEmitter::build_lrn_forward(const mkldnn::lrn_forward::desc& lrn_desc, size_t lrn_index)
{
    size_t input_index = m_primitive_deps[lrn_index][0];
    build_memory_primitive(lrn_desc.data.data_desc, input_index);
    size_t result_index = m_primitive_deps[lrn_index][1];
    build_memory_primitive(lrn_desc.data.data_desc, result_index);

    auto lrn_prim_desc = mkldnn::lrn_forward::primitive_desc(lrn_desc, executor::global_cpu_engine);

    m_mkldnn_primitives[lrn_index] = new mkldnn::lrn_forward(
        lrn_prim_desc, *m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_relu_forward(const mkldnn::memory::desc& input_desc,
                                       const mkldnn::memory::desc& result_desc,
                                       const Node* node)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    const float negative_slope = 0.0f;
    auto relu_desc = mkldnn::eltwise_forward::desc(
        mkldnn::prop_kind::forward, mkldnn::algorithm::eltwise_relu, input_desc, negative_slope);
    auto relu_pd = mkldnn::eltwise_forward::primitive_desc(relu_desc, executor::global_cpu_engine);

    size_t primitive_index = insert_primitive(new mkldnn::eltwise_forward(
        relu_pd, *m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {input_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
}

mkldnn::eltwise_forward::desc MKLDNNEmitter::get_relu_forward_desc(const ngraph::Node* node)
{
    const float negative_slope = 0.0f;

    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);

    return mkldnn::eltwise_forward::desc(
        mkldnn::prop_kind::forward, mkldnn::algorithm::eltwise_relu, input_desc, negative_slope);
}

void MKLDNNEmitter::build_relu_forward(const mkldnn::eltwise_forward::desc& relu_desc,
                                       size_t relu_index)
{
    size_t input_index = m_primitive_deps[relu_index][0];
    build_memory_primitive(relu_desc.data.data_desc, input_index);
    size_t result_index = m_primitive_deps[relu_index][1];
    build_memory_primitive(relu_desc.data.data_desc, result_index);

    m_mkldnn_primitives[relu_index] =
        new mkldnn::eltwise_forward({relu_desc, executor::global_cpu_engine},
                                    *m_mkldnn_primitives[input_index],
                                    *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_relu_backward(const mkldnn::memory::desc& input_desc,
                                        const mkldnn::memory::desc& delta_desc,
                                        const mkldnn::memory::desc& result_desc,
                                        const Node* node)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t delta_index = build_memory_primitive(delta_desc);
    size_t result_index = build_memory_primitive(result_desc);

    /* Backward relu */
    const float negative_slope = 0.0f;
    auto relu_desc = mkldnn::eltwise_forward::desc(
        mkldnn::prop_kind::forward, mkldnn::algorithm::eltwise_relu, input_desc, negative_slope);
    auto relu_pd = mkldnn::eltwise_forward::primitive_desc(relu_desc, executor::global_cpu_engine);

    /* create backward relu primitive_descriptor */
    auto relu_bwd_desc = mkldnn::eltwise_backward::desc(
        mkldnn::algorithm::eltwise_relu, result_desc, input_desc, negative_slope);
    auto relu_bwd_pd = mkldnn::eltwise_backward::primitive_desc(
        relu_bwd_desc, executor::global_cpu_engine, relu_pd);

    size_t primitive_index =
        insert_primitive(new mkldnn::eltwise_backward(relu_bwd_pd,
                                                      *m_mkldnn_primitives[input_index],
                                                      *m_mkldnn_primitives[delta_index],
                                                      *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {input_index, delta_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
}

mkldnn::eltwise_backward::desc MKLDNNEmitter::get_relu_backward_desc(const ngraph::Node* node)
{
    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

    const float negative_slope = 0.0f;
    return mkldnn::eltwise_backward::desc(
        mkldnn::algorithm::eltwise_relu, result_desc, input_desc, negative_slope);
}

void MKLDNNEmitter::build_relu_backward(const mkldnn::eltwise_backward::desc& bwd_desc,
                                        const mkldnn::eltwise_forward::desc& fwd_desc,
                                        size_t relu_index)
{
    size_t input_index = m_primitive_deps[relu_index][0];
    build_memory_primitive(bwd_desc.data.data_desc, input_index);
    size_t delta_index = m_primitive_deps[relu_index][1];
    build_memory_primitive(bwd_desc.data.diff_data_desc, delta_index);
    size_t result_index = m_primitive_deps[relu_index][2];
    build_memory_primitive(bwd_desc.data.data_desc, result_index);

    /* create forward relu primitive descriptor*/
    auto relu_pd = mkldnn::eltwise_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    /* create backward relu primitive_descriptor */
    auto relu_bwd_pd =
        mkldnn::eltwise_backward::primitive_desc(bwd_desc, executor::global_cpu_engine, relu_pd);

    m_mkldnn_primitives[relu_index] =
        new mkldnn::eltwise_backward(relu_bwd_pd,
                                     *m_mkldnn_primitives[input_index],
                                     *m_mkldnn_primitives[delta_index],
                                     *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_sigmoid_forward(const mkldnn::memory::desc& input_desc,
                                          const mkldnn::memory::desc& result_desc,
                                          const Node* node)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index =
        insert_primitive(new mkldnn::eltwise_forward({{mkldnn::prop_kind::forward_training,
                                                       mkldnn::algorithm::eltwise_logistic,
                                                       input_desc,
                                                       0,
                                                       0},
                                                      executor::global_cpu_engine},
                                                     *m_mkldnn_primitives[input_index],
                                                     *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {input_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
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

void MKLDNNEmitter::build_sigmoid_forward(const mkldnn::eltwise_forward::desc& sigmoid_desc,
                                          size_t sigmoid_index)
{
    size_t input_index = m_primitive_deps[sigmoid_index][0];
    build_memory_primitive(sigmoid_desc.data.data_desc, input_index);
    size_t result_index = m_primitive_deps[sigmoid_index][1];
    build_memory_primitive(sigmoid_desc.data.data_desc, result_index);

    m_mkldnn_primitives[sigmoid_index] =
        new mkldnn::eltwise_forward({sigmoid_desc, executor::global_cpu_engine},
                                    *m_mkldnn_primitives[input_index],
                                    *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_sigmoid_backward(const mkldnn::memory::desc& input_desc,
                                           const mkldnn::memory::desc& delta_desc,
                                           const mkldnn::memory::desc& result_desc,
                                           const Node* node)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t delta_index = build_memory_primitive(delta_desc);
    size_t result_index = build_memory_primitive(result_desc);

    // sigmoid forward primitive desc
    mkldnn::eltwise_forward::primitive_desc sigmoid_fwd_pd =
        mkldnn::eltwise_forward::primitive_desc(
            {mkldnn::prop_kind::forward, mkldnn::algorithm::eltwise_logistic, input_desc, 0, 0},
            executor::global_cpu_engine);

    size_t primitive_index = insert_primitive(new mkldnn::eltwise_backward(
        {{mkldnn::algorithm::eltwise_logistic, delta_desc, input_desc, 0, 0},
         executor::global_cpu_engine,
         sigmoid_fwd_pd},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[delta_index],
        *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {input_index, delta_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
}

mkldnn::eltwise_backward::desc MKLDNNEmitter::get_sigmoid_backward_desc(const ngraph::Node* node)
{
    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);

    return mkldnn::eltwise_backward::desc(
        mkldnn::algorithm::eltwise_logistic, delta_desc, input_desc, 0, 0);
}

void MKLDNNEmitter::build_sigmoid_backward(const mkldnn::eltwise_backward::desc& bwd_desc,
                                           const mkldnn::eltwise_forward::desc& fwd_desc,
                                           size_t sigmoid_index)
{
    size_t input_index = m_primitive_deps[sigmoid_index][0];
    build_memory_primitive(bwd_desc.data.data_desc, input_index);
    size_t delta_index = m_primitive_deps[sigmoid_index][1];
    build_memory_primitive(bwd_desc.data.diff_data_desc, delta_index);
    size_t result_index = m_primitive_deps[sigmoid_index][2];
    build_memory_primitive(bwd_desc.data.data_desc, result_index);

    // sigmoid forward primitive desc
    mkldnn::eltwise_forward::primitive_desc sigmoid_fwd_pd =
        mkldnn::eltwise_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    m_mkldnn_primitives[sigmoid_index] =
        new mkldnn::eltwise_backward({bwd_desc, executor::global_cpu_engine, sigmoid_fwd_pd},
                                     *m_mkldnn_primitives[input_index],
                                     *m_mkldnn_primitives[delta_index],
                                     *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_elementwise_add(
    const mkldnn::memory::desc& input0_data_desc,
    const mkldnn::memory::desc& input1_data_desc,
    const mkldnn::memory::desc& result_desc,
    const std::vector<float>& scale_vector,
    const std::vector<mkldnn::memory::primitive_desc>& inputs_pd,
    const Node* node)
{
    std::vector<mkldnn::memory::primitive::at> inputs_primitive;

    size_t input0_data_index = build_memory_primitive(input0_data_desc);
    size_t input1_data_index = build_memory_primitive(input1_data_desc);
    size_t result_index = build_memory_primitive(result_desc);

    inputs_primitive.push_back(*m_mkldnn_primitives[input0_data_index]);
    inputs_primitive.push_back(*m_mkldnn_primitives[input1_data_index]);

    // elementwise sum primtive descriptor
    mkldnn::sum::primitive_desc sum_pd =
        mkldnn::sum::primitive_desc(result_desc, scale_vector, inputs_pd);
    // sum primitive
    size_t add_index = insert_primitive(
        new mkldnn::sum(sum_pd, inputs_primitive, *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(add_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[add_index] = {input0_data_index, input1_data_index, result_index};
    m_node_primitive_idx_map[node] = add_index;
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

void MKLDNNEmitter::build_elementwise_add(const mkldnn::sum::primitive_desc& sum_pd,
                                          size_t add_index)
{
    std::vector<mkldnn::memory::primitive::at> inputs_primitive;

    size_t input0_data_index = m_primitive_deps[add_index][0];
    build_memory_primitive(sum_pd.dst_primitive_desc().desc(), input0_data_index);
    size_t input1_data_index = m_primitive_deps[add_index][1];
    build_memory_primitive(sum_pd.dst_primitive_desc().desc(), input1_data_index);
    size_t result_index = m_primitive_deps[add_index][2];
    build_memory_primitive(sum_pd.dst_primitive_desc().desc(), result_index);

    inputs_primitive.push_back(*m_mkldnn_primitives[input0_data_index]);
    inputs_primitive.push_back(*m_mkldnn_primitives[input1_data_index]);

    // sum primitive
    m_mkldnn_primitives[add_index] =
        new mkldnn::sum(sum_pd, inputs_primitive, *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_batchnorm_forward(const mkldnn::memory::desc& input_desc,
                                            const mkldnn::memory::desc& weights_desc,
                                            const mkldnn::memory::desc& result_desc,
                                            const mkldnn::memory::desc& mean_desc,
                                            const mkldnn::memory::desc& variance_desc,
                                            const double eps,
                                            bool use_global_stats,
                                            bool bn_training_flag,
                                            const Node* node,
                                            const mkldnn::post_ops& pops)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t result_index = build_memory_primitive(result_desc);
    size_t mean_index = build_memory_primitive(mean_desc);
    size_t variance_index = build_memory_primitive(variance_desc);

    mkldnn::primitive_attr bn_attr;
    bn_attr.set_post_ops(pops);

    size_t batchnorm_index;
    if (bn_training_flag && !use_global_stats)
    {
        batchnorm_index = insert_primitive(new mkldnn::batch_normalization_forward(
            {{mkldnn::prop_kind::forward_training,
              input_desc,
              eps,
              mkldnn::batch_normalization_flag::use_scale_shift},
             bn_attr,
             executor::global_cpu_engine},
            mkldnn::primitive::at(*m_mkldnn_primitives[input_index]),
            mkldnn::primitive::at(*m_mkldnn_primitives[weights_index]),
            static_cast<mkldnn::memory>(*m_mkldnn_primitives[result_index]),
            *m_mkldnn_primitives[mean_index],
            *m_mkldnn_primitives[variance_index]));

        m_primitive_deps[batchnorm_index] = {
            input_index, weights_index, result_index, mean_index, variance_index};
    }
    else
    {
        batchnorm_index = insert_primitive(new mkldnn::batch_normalization_forward(
            {{mkldnn::prop_kind::forward_training,
              input_desc,
              eps,
              mkldnn::batch_normalization_flag::use_scale_shift |
                  mkldnn::batch_normalization_flag::use_global_stats},
             bn_attr,
             executor::global_cpu_engine},
            mkldnn::primitive::at(*m_mkldnn_primitives[input_index]),
            mkldnn::primitive::at(*m_mkldnn_primitives[mean_index]),
            mkldnn::primitive::at(*m_mkldnn_primitives[variance_index]),
            mkldnn::primitive::at(*m_mkldnn_primitives[weights_index]),
            static_cast<mkldnn::memory>(*m_mkldnn_primitives[result_index])));

        m_primitive_deps[batchnorm_index] = {
            input_index, mean_index, variance_index, weights_index, result_index};
    }

    NGRAPH_ASSERT(m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_node_primitive_idx_map[node] = batchnorm_index;
}

void MKLDNNEmitter::build_batchnorm_forward(
    const mkldnn::batch_normalization_forward::desc& batchnorm_desc,
    const mkldnn::memory::desc& weights_desc,
    bool bn_training_flag,
    size_t batchnorm_index,
    const mkldnn::post_ops& pops)
{
    size_t input_index = m_primitive_deps[batchnorm_index][0];
    build_memory_primitive(batchnorm_desc.data.data_desc, input_index);

    mkldnn::primitive_attr bn_attr;
    bn_attr.set_post_ops(pops);

    auto use_global_stats = batchnorm_desc.data.flags & 0x1U;
    if (bn_training_flag && !use_global_stats)
    {
        size_t weights_index = m_primitive_deps[batchnorm_index][1];
        build_memory_primitive(weights_desc, weights_index);
        size_t result_index = m_primitive_deps[batchnorm_index][2];
        build_memory_primitive(batchnorm_desc.data.data_desc, result_index);
        size_t mean_index = m_primitive_deps[batchnorm_index][3];
        build_memory_primitive(batchnorm_desc.data.mean_desc, mean_index);
        size_t variance_index = m_primitive_deps[batchnorm_index][4];
        build_memory_primitive(batchnorm_desc.data.variance_desc, variance_index);

        m_mkldnn_primitives[batchnorm_index] = new mkldnn::batch_normalization_forward(
            {batchnorm_desc, bn_attr, executor::global_cpu_engine},
            mkldnn::primitive::at(*m_mkldnn_primitives[input_index]),
            mkldnn::primitive::at(*m_mkldnn_primitives[weights_index]),
            static_cast<mkldnn::memory>(*m_mkldnn_primitives[result_index]),
            *m_mkldnn_primitives[mean_index],
            *m_mkldnn_primitives[variance_index]);
    }
    else
    {
        size_t weights_index = m_primitive_deps[batchnorm_index][3];
        build_memory_primitive(weights_desc, weights_index);
        size_t result_index = m_primitive_deps[batchnorm_index][4];
        build_memory_primitive(batchnorm_desc.data.data_desc, result_index);
        size_t mean_index = m_primitive_deps[batchnorm_index][1];
        build_memory_primitive(batchnorm_desc.data.mean_desc, mean_index);
        size_t variance_index = m_primitive_deps[batchnorm_index][2];
        build_memory_primitive(batchnorm_desc.data.variance_desc, variance_index);

        m_mkldnn_primitives[batchnorm_index] = new mkldnn::batch_normalization_forward(
            {batchnorm_desc, bn_attr, executor::global_cpu_engine},
            mkldnn::primitive::at(*m_mkldnn_primitives[input_index]),
            mkldnn::primitive::at(*m_mkldnn_primitives[mean_index]),
            mkldnn::primitive::at(*m_mkldnn_primitives[variance_index]),
            mkldnn::primitive::at(*m_mkldnn_primitives[weights_index]),
            static_cast<mkldnn::memory>(*m_mkldnn_primitives[result_index]));
    }
}

void MKLDNNEmitter::build_batchnorm_backward(const mkldnn::memory::desc& weights_desc,
                                             const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& mean_desc,
                                             const mkldnn::memory::desc& variance_desc,
                                             const mkldnn::memory::desc& delta_desc,
                                             const mkldnn::memory::desc& dinput_desc,
                                             const mkldnn::memory::desc& dweights_desc,
                                             const double eps,
                                             const Node* node)
{
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t input_index = build_memory_primitive(input_desc);
    size_t mean_index = build_memory_primitive(mean_desc);
    size_t variance_index = build_memory_primitive(variance_desc);
    size_t delta_index = build_memory_primitive(delta_desc);
    size_t dinput_index = build_memory_primitive(dinput_desc);
    size_t dweights_index = build_memory_primitive(dweights_desc);

    size_t batchnorm_index = insert_primitive(new mkldnn::batch_normalization_backward(
        {{mkldnn::prop_kind::backward,
          delta_desc,
          input_desc,
          eps,
          mkldnn::batch_normalization_flag::use_scale_shift},
         executor::global_cpu_engine,
         {{mkldnn::prop_kind::forward_training,
           input_desc,
           eps,
           mkldnn::batch_normalization_flag::use_scale_shift},
          executor::global_cpu_engine}},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[mean_index],
        *m_mkldnn_primitives[variance_index],
        *m_mkldnn_primitives[delta_index],
        *m_mkldnn_primitives[weights_index],
        *m_mkldnn_primitives[dinput_index],
        *m_mkldnn_primitives[dweights_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(batchnorm_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[batchnorm_index] = {weights_index,
                                         input_index,
                                         mean_index,
                                         variance_index,
                                         delta_index,
                                         dinput_index,
                                         dweights_index};
    m_node_primitive_idx_map[node] = batchnorm_index;
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
    const mkldnn::batch_normalization_backward::desc& batchnorm_desc,
    const mkldnn::memory::desc& weights_desc,
    const mkldnn::memory::desc& dweights_desc,
    size_t batchnorm_index)
{
    size_t weights_index = m_primitive_deps[batchnorm_index][0];
    build_memory_primitive(weights_desc, weights_index);
    size_t input_index = m_primitive_deps[batchnorm_index][1];
    build_memory_primitive(batchnorm_desc.data.data_desc, input_index);
    size_t mean_index = m_primitive_deps[batchnorm_index][2];
    build_memory_primitive(batchnorm_desc.data.mean_desc, mean_index);
    size_t variance_index = m_primitive_deps[batchnorm_index][3];
    build_memory_primitive(batchnorm_desc.data.variance_desc, variance_index);
    size_t delta_index = m_primitive_deps[batchnorm_index][4];
    build_memory_primitive(batchnorm_desc.data.diff_data_desc, delta_index);
    size_t dinput_index = m_primitive_deps[batchnorm_index][5];
    build_memory_primitive(batchnorm_desc.data.data_desc, dinput_index);
    size_t dweights_index = m_primitive_deps[batchnorm_index][6];
    build_memory_primitive(dweights_desc, dweights_index);

    m_mkldnn_primitives[batchnorm_index] = new mkldnn::batch_normalization_backward(
        {batchnorm_desc,
         executor::global_cpu_engine,
         {{mkldnn::prop_kind::forward_training,
           batchnorm_desc.data.data_desc,
           static_cast<double>(batchnorm_desc.data.batch_norm_epsilon),
           mkldnn::batch_normalization_flag::use_scale_shift},
          executor::global_cpu_engine}},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[mean_index],
        *m_mkldnn_primitives[variance_index],
        *m_mkldnn_primitives[delta_index],
        *m_mkldnn_primitives[weights_index],
        *m_mkldnn_primitives[dinput_index],
        *m_mkldnn_primitives[dweights_index]);
}

void MKLDNNEmitter::build_rnn_forward(const mkldnn::memory::desc& src_layer_desc,
                                      const mkldnn::memory::desc& src_iter_desc,
                                      const mkldnn::memory::desc& weights_layer_desc,
                                      const mkldnn::memory::desc& weights_iter_desc,
                                      const mkldnn::memory::desc& bias_desc,
                                      const mkldnn::memory::desc& dst_layer_desc,
                                      const mkldnn::memory::desc& dst_iter_desc,
                                      const mkldnn::rnn_direction& rnn_direction,
                                      const mkldnn::algorithm& rnn_algorithm,
                                      const Node* node)
{
    size_t src_layer_index = build_memory_primitive(src_layer_desc);
    size_t src_iter_index = build_memory_primitive(src_iter_desc);
    size_t weights_layer_index = build_memory_primitive(weights_layer_desc);
    size_t weights_iter_index = build_memory_primitive(weights_iter_desc);
    size_t bias_index = build_memory_primitive(bias_desc);
    size_t dst_layer_index = build_memory_primitive(dst_layer_desc);
    size_t dst_iter_index = build_memory_primitive(dst_iter_desc);

    mkldnn::rnn_cell::desc rnn_cell(rnn_algorithm);
    mkldnn::rnn_forward::desc rnn_layer_desc(mkldnn::prop_kind::forward_training,
                                             rnn_cell,
                                             rnn_direction,
                                             src_layer_desc,
                                             src_iter_desc,
                                             weights_layer_desc,
                                             weights_iter_desc,
                                             bias_desc,
                                             dst_layer_desc,
                                             dst_iter_desc);

    auto rnn_layer_prim_desc =
        mkldnn::rnn_forward::primitive_desc(rnn_layer_desc, executor::global_cpu_engine);
    auto workspace_index =
        build_memory_primitive(rnn_layer_prim_desc.workspace_primitive_desc().desc());
    auto workspace = std::unique_ptr<MKLDNNWorkspace>(
        new MKLDNNWorkspace(rnn_layer_prim_desc.workspace_primitive_desc().get_size()));

    auto workspace_buf_index = insert_workspace(workspace);

    size_t rnn_index = insert_primitive(new mkldnn::rnn_forward(
        rnn_layer_prim_desc,
        mkldnn::primitive::at(*m_mkldnn_primitives[src_layer_index]),
        mkldnn::primitive::at(*m_mkldnn_primitives[src_iter_index]),
        mkldnn::primitive::at(*m_mkldnn_primitives[weights_layer_index]),
        mkldnn::primitive::at(*m_mkldnn_primitives[weights_iter_index]),
        mkldnn::primitive::at(*m_mkldnn_primitives[bias_index]),
        static_cast<mkldnn::memory>(*m_mkldnn_primitives[dst_layer_index]),
        static_cast<mkldnn::memory>(*m_mkldnn_primitives[dst_iter_index]),
        static_cast<mkldnn::memory>(*m_mkldnn_primitives[workspace_index])));

    NGRAPH_ASSERT(m_primitive_deps.find(rnn_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[rnn_index] = {src_layer_index,
                                   src_iter_index,
                                   weights_layer_index,
                                   weights_iter_index,
                                   bias_index,
                                   dst_layer_index,
                                   dst_iter_index,
                                   workspace_index,
                                   workspace_buf_index};
    m_node_primitive_idx_map[node] = rnn_index;
}

void MKLDNNEmitter::build_rnn_forward(const mkldnn::rnn_forward::desc& rnn_desc, size_t rnn_index)
{
    size_t src_layer_index = m_primitive_deps[rnn_index][0];
    build_memory_primitive(rnn_desc.data.src_layer_desc, src_layer_index);
    size_t src_iter_index = m_primitive_deps[rnn_index][1];
    build_memory_primitive(rnn_desc.data.src_iter_desc, src_iter_index);
    size_t weights_layer_index = m_primitive_deps[rnn_index][2];
    build_memory_primitive(rnn_desc.data.weights_layer_desc, weights_layer_index);
    size_t weights_iter_index = m_primitive_deps[rnn_index][3];
    build_memory_primitive(rnn_desc.data.weights_iter_desc, weights_iter_index);
    size_t bias_index = m_primitive_deps[rnn_index][4];
    build_memory_primitive(rnn_desc.data.bias_desc, bias_index);
    size_t dst_layer_index = m_primitive_deps[rnn_index][5];
    build_memory_primitive(rnn_desc.data.dst_layer_desc, dst_layer_index);
    size_t dst_iter_index = m_primitive_deps[rnn_index][6];
    build_memory_primitive(rnn_desc.data.dst_iter_desc, dst_iter_index);

    auto rnn_layer_prim_desc =
        mkldnn::rnn_forward::primitive_desc(rnn_desc, executor::global_cpu_engine);
    size_t workspace_index = m_primitive_deps[rnn_index][7];
    build_memory_primitive(rnn_layer_prim_desc.workspace_primitive_desc().desc(), workspace_index);
    auto workspace = std::unique_ptr<MKLDNNWorkspace>(
        new MKLDNNWorkspace(rnn_layer_prim_desc.workspace_primitive_desc().get_size()));
    auto workspace_buf_index = insert_workspace(workspace);
    m_primitive_deps[rnn_index][8] = workspace_buf_index;

    m_mkldnn_primitives[rnn_index] =
        new mkldnn::rnn_forward(rnn_layer_prim_desc,
                                mkldnn::primitive::at(*m_mkldnn_primitives[src_layer_index]),
                                mkldnn::primitive::at(*m_mkldnn_primitives[src_iter_index]),
                                mkldnn::primitive::at(*m_mkldnn_primitives[weights_layer_index]),
                                mkldnn::primitive::at(*m_mkldnn_primitives[weights_iter_index]),
                                mkldnn::primitive::at(*m_mkldnn_primitives[bias_index]),
                                static_cast<mkldnn::memory>(*m_mkldnn_primitives[dst_layer_index]),
                                static_cast<mkldnn::memory>(*m_mkldnn_primitives[dst_iter_index]),
                                static_cast<mkldnn::memory>(*m_mkldnn_primitives[workspace_index]));
}

size_t MKLDNNEmitter::build_concat(const std::vector<mkldnn::memory::desc>& inputs_data_desc,
                                   const mkldnn::memory::desc& result_desc,
                                   const size_t concat_dim,
                                   const Node* node)
{
    std::vector<mkldnn::memory::primitive::at> inputs_primitive;
    std::vector<size_t> inputs_data_index;
    std::vector<size_t> in_out_index;
    std::vector<mkldnn::memory::primitive_desc> inputs_pd;

    for (size_t i = 0; i < inputs_data_desc.size(); i++)
    {
        inputs_pd.push_back(mkldnn::memory::primitive_desc(
            inputs_data_desc[i], runtime::cpu::executor::global_cpu_engine));
    }

    for (size_t i = 0; i < inputs_data_desc.size(); i++)
    {
        inputs_data_index.push_back(build_memory_primitive(inputs_data_desc[i]));
        inputs_primitive.push_back(*m_mkldnn_primitives[inputs_data_index[i]]);
    }
    size_t result_index = build_memory_primitive(result_desc);

    // concat primtive descriptor
    mkldnn::concat::primitive_desc concat_pd =
        mkldnn::concat::primitive_desc(result_desc, static_cast<int>(concat_dim), inputs_pd);
    // concat primitive
    size_t concat_index = insert_primitive(
        new mkldnn::concat(concat_pd, inputs_primitive, *m_mkldnn_primitives[result_index]));

    for (size_t i = 0; i < inputs_data_index.size(); i++)
    {
        in_out_index.push_back(inputs_data_index[i]);
    }
    in_out_index.push_back(result_index);

    NGRAPH_ASSERT(m_primitive_deps.find(concat_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[concat_index] = in_out_index;
    m_node_primitive_idx_map[node] = concat_index;

    return concat_index;
}

mkldnn::concat::primitive_desc MKLDNNEmitter::get_concat_desc(const ngraph::Node* node,
                                                              size_t nargs)
{
    auto concat = static_cast<const ngraph::op::Concat*>(node);

    std::vector<mkldnn::memory::primitive_desc> inputs_pd;
    for (size_t i = 0; i < nargs; i++)
    {
        inputs_pd.push_back(mkldnn::memory::primitive_desc(
            mkldnn_utils::get_input_mkldnn_md(node, i), runtime::cpu::executor::global_cpu_engine));
    }

    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

    size_t concat_dim = concat->get_concatenation_axis();

    // concat primtive descriptor
    return mkldnn::concat::primitive_desc(result_desc, static_cast<int>(concat_dim), inputs_pd);
}

void MKLDNNEmitter::build_concat(const mkldnn::concat::primitive_desc& concat_pd,
                                 const std::vector<mkldnn::memory::desc>& inputs_data_desc,
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
        size_t inputs_data_index = m_primitive_deps[concat_index][i];
        build_memory_primitive(inputs_data_desc[i], inputs_data_index);
        inputs_primitive.push_back(*m_mkldnn_primitives[inputs_data_index]);
    }
    size_t result_index = m_primitive_deps[concat_index][inputs_data_desc.size()];
    build_memory_primitive(concat_pd.dst_primitive_desc().desc(), result_index);

    // concat primitive
    m_mkldnn_primitives[concat_index] =
        new mkldnn::concat(concat_pd, inputs_primitive, *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_slice(const mkldnn::memory::desc& input_desc,
                                const mkldnn::memory::desc& result_desc,
                                const ngraph::Coordinate& lower_bounds,
                                const ngraph::Shape& result_shape,
                                const Node* node)
{
    mkldnn::memory::primitive_desc input_pd =
        mkldnn::memory::primitive_desc(input_desc, runtime::cpu::executor::global_cpu_engine);
    size_t input_index = build_memory_primitive(input_desc);

    auto dims = mkldnn::memory::dims(result_shape.begin(), result_shape.end());
    auto offsets = mkldnn::memory::dims(lower_bounds.begin(), lower_bounds.end());
    auto view_pd = mkldnn::view::primitive_desc(input_pd, dims, offsets).dst_primitive_desc();

    mkldnn::memory::primitive_desc result_pd =
        mkldnn::memory::primitive_desc(result_desc, runtime::cpu::executor::global_cpu_engine);
    size_t result_index = build_memory_primitive(result_desc);

    // reorder primitive descriptor
    mkldnn::reorder::primitive_desc reorder_pd =
        mkldnn::reorder::primitive_desc(view_pd, result_pd);
    // reorder primitive
    size_t reorder_index = insert_primitive(new mkldnn::reorder(
        reorder_pd, *m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(reorder_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    std::vector<size_t> in_out_index;
    in_out_index.push_back(input_index);
    in_out_index.push_back(result_index);

    m_primitive_deps[reorder_index] = in_out_index;
    m_node_primitive_idx_map[node] = reorder_index;
}

void MKLDNNEmitter::build_slice(const mkldnn::memory::desc& input_desc,
                                const mkldnn::memory::desc& result_desc,
                                const ngraph::Coordinate& lower_bounds,
                                const ngraph::Shape& result_shape,
                                size_t slice_index)
{
    std::vector<size_t> in_out_index;
    mkldnn::memory::primitive_desc input_pd =
        mkldnn::memory::primitive_desc(input_desc, runtime::cpu::executor::global_cpu_engine);
    size_t input_index = m_primitive_deps[slice_index][0];
    build_memory_primitive(input_desc, input_index);

    auto dims = mkldnn::memory::dims(result_shape.begin(), result_shape.end());
    auto offsets = mkldnn::memory::dims(lower_bounds.begin(), lower_bounds.end());
    auto view_pd = mkldnn::view::primitive_desc(input_pd, dims, offsets).dst_primitive_desc();

    mkldnn::memory::primitive_desc result_pd =
        mkldnn::memory::primitive_desc(result_desc, runtime::cpu::executor::global_cpu_engine);
    size_t result_index = m_primitive_deps[slice_index][1];
    build_memory_primitive(result_desc, result_index);

    // reorder primitive descriptor
    mkldnn::reorder::primitive_desc reorder_pd =
        mkldnn::reorder::primitive_desc(view_pd, result_pd);
    // reorder primitive
    m_mkldnn_primitives[slice_index] = new mkldnn::reorder(
        reorder_pd, *m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_softmax_forward(const mkldnn::memory::desc& input_desc,
                                          const mkldnn::memory::desc& result_desc,
                                          int softmax_axis,
                                          const Node* node)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index = insert_primitive(
        new mkldnn::softmax_forward({{mkldnn::prop_kind::forward_scoring, input_desc, softmax_axis},
                                     executor::global_cpu_engine},
                                    *m_mkldnn_primitives[input_index],
                                    *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {input_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
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

void MKLDNNEmitter::build_softmax_forward(const mkldnn::softmax_forward::desc& softmax_desc,
                                          size_t softmax_index)
{
    size_t input_index = m_primitive_deps[softmax_index][0];
    build_memory_primitive(softmax_desc.data.data_desc, input_index);
    size_t result_index = m_primitive_deps[softmax_index][1];
    build_memory_primitive(softmax_desc.data.data_desc, result_index);

    m_mkldnn_primitives[softmax_index] =
        new mkldnn::softmax_forward({softmax_desc, executor::global_cpu_engine},
                                    *m_mkldnn_primitives[input_index],
                                    *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_leaky_relu(const mkldnn::memory::desc& input_desc,
                                     const mkldnn::memory::desc& result_desc,
                                     float alpha,
                                     const Node* node)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index =
        insert_primitive(new mkldnn::eltwise_forward({{mkldnn::prop_kind::forward_training,
                                                       mkldnn::algorithm::eltwise_relu,
                                                       input_desc,
                                                       alpha,
                                                       0.0f},
                                                      executor::global_cpu_engine},
                                                     *m_mkldnn_primitives[input_index],
                                                     *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {input_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
}

mkldnn::eltwise_forward::desc MKLDNNEmitter::get_leaky_relu_desc(const ngraph::Node* node)
{
    auto alpha = static_cast<const LeakyRelu*>(node)->get_alpha();

    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);

    return mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward_training,
                                         mkldnn::algorithm::eltwise_relu,
                                         input_desc,
                                         alpha,
                                         0.0f);
}

void MKLDNNEmitter::build_leaky_relu(const mkldnn::eltwise_forward::desc& leaky_relu_desc,
                                     size_t leaky_relu_index)
{
    size_t input_index = m_primitive_deps[leaky_relu_index][0];
    build_memory_primitive(leaky_relu_desc.data.data_desc, input_index);
    size_t result_index = m_primitive_deps[leaky_relu_index][1];
    build_memory_primitive(leaky_relu_desc.data.data_desc, result_index);

    m_mkldnn_primitives[leaky_relu_index] =
        new mkldnn::eltwise_forward({leaky_relu_desc, executor::global_cpu_engine},
                                    *m_mkldnn_primitives[input_index],
                                    *m_mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_bounded_relu(const mkldnn::memory::desc& input_desc,
                                       const mkldnn::memory::desc& result_desc,
                                       float alpha,
                                       const Node* node)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index =
        insert_primitive(new mkldnn::eltwise_forward({{mkldnn::prop_kind::forward_training,
                                                       mkldnn::algorithm::eltwise_bounded_relu,
                                                       input_desc,
                                                       alpha,
                                                       0.0f},
                                                      executor::global_cpu_engine},
                                                     *m_mkldnn_primitives[input_index],
                                                     *m_mkldnn_primitives[result_index]));

    NGRAPH_ASSERT(m_primitive_deps.find(primitive_index) == m_primitive_deps.end() &&
                  m_node_primitive_idx_map.find(node) == m_node_primitive_idx_map.end())
        << "Primitive already created for node " << node->description();

    m_primitive_deps[primitive_index] = {input_index, result_index};
    m_node_primitive_idx_map[node] = primitive_index;
}

mkldnn::eltwise_forward::desc MKLDNNEmitter::get_bounded_relu_desc(const ngraph::Node* node)
{
    auto alpha = static_cast<const BoundedRelu*>(node)->get_alpha();

    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);

    return mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward_training,
                                         mkldnn::algorithm::eltwise_bounded_relu,
                                         input_desc,
                                         alpha,
                                         0.0f);
}

void MKLDNNEmitter::build_bounded_relu(const mkldnn::eltwise_forward::desc& bounded_relu_desc,
                                       size_t bounded_relu_index)
{
    size_t input_index = m_primitive_deps[bounded_relu_index][0];
    build_memory_primitive(bounded_relu_desc.data.data_desc, input_index);
    size_t result_index = m_primitive_deps[bounded_relu_index][1];
    build_memory_primitive(bounded_relu_desc.data.data_desc, result_index);

    m_mkldnn_primitives[bounded_relu_index] =
        new mkldnn::eltwise_forward({bounded_relu_desc, executor::global_cpu_engine},
                                    *m_mkldnn_primitives[input_index],
                                    *m_mkldnn_primitives[result_index]);
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

// Translate inputs of \p node to TensorViewWrappers.
static void translate_node_inputs(const Node* node, std::vector<TensorViewWrapper>& args)
{
    NGRAPH_ASSERT(args.empty()) << "Expected output vector";

    for (const descriptor::Input& input : node->get_inputs())
    {
        const descriptor::Output& output = input.get_output();
        std::shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
        // TODO: We should use m_variable_name_map[tv->get_name()] but this map hasn't been
        // populated when MKLDNN primitives are built. Neither it was when intrinsics where built
        // for CommonFunctionCollection.
        args.emplace_back(TensorViewWrapper(tv, /*m_variable_name_map[*/ tv->get_name() /*]*/));
    }
}

// Translate outputs of \p node to TensorViewWrappers.
static void translate_node_outputs(const Node* node, std::vector<TensorViewWrapper>& out)
{
    NGRAPH_ASSERT(out.empty()) << "Expected output vector";

    for (const descriptor::Output& output : node->get_outputs())
    {
        std::shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
        // TODO: We should use m_variable_name_map[tv->get_name()] but this map hasn't been
        // populated when MKLDNN primitives are built. Neither it was when intrinsics where built
        // for CommonFunctionCollection.
        out.push_back(TensorViewWrapper(tv, /*m_variable_name_map[*/ tv->get_name() /*]*/));
    }
}

// Translate inputs and outputs of \p node to TensorViewWrappers.
static void translate_node_inputs_outputs(const Node* node,
                                          std::vector<TensorViewWrapper>& args,
                                          std::vector<TensorViewWrapper>& out)
{
    translate_node_inputs(node, args);
    translate_node_outputs(node, out);
}

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                // The following functions build the MKLDNN primitive for each type of nGraph Node.

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Add)
                {
                    std::vector<float> scale_vector(2, 1);
                    std::vector<mkldnn::memory::primitive_desc> inputs_pd;

                    auto input0_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto input1_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    inputs_pd.push_back(mkldnn::memory::primitive_desc(
                        input0_data_desc, runtime::cpu::executor::global_cpu_engine));
                    inputs_pd.push_back(mkldnn::memory::primitive_desc(
                        input1_data_desc, runtime::cpu::executor::global_cpu_engine));

                    mkldnn_emitter.build_elementwise_add(input0_data_desc,
                                                         input1_data_desc,
                                                         result_desc,
                                                         scale_vector,
                                                         inputs_pd,
                                                         node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Lstm)
                {
                    std::vector<TensorViewWrapper> args;
                    std::vector<TensorViewWrapper> out;
                    translate_node_inputs_outputs(node, args, out);

                    mkldnn_emitter.build_rnn<Lstm>(node, args, out);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Rnn)
                {
                    std::vector<TensorViewWrapper> args;
                    std::vector<TensorViewWrapper> out;
                    translate_node_inputs_outputs(node, args, out);

                    mkldnn_emitter.build_rnn<Rnn>(node, args, out);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormTraining)
                {
                    std::vector<TensorViewWrapper> args;
                    std::vector<TensorViewWrapper> out;
                    translate_node_inputs_outputs(node, args, out);

                    mkldnn_emitter.build_batch_norm_primitive<BatchNormInference>(
                        node, args, out, false /*Append relu*/, true /*Training*/);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormInference)
                {
                    std::vector<TensorViewWrapper> args;
                    std::vector<TensorViewWrapper> out;
                    translate_node_inputs_outputs(node, args, out);

                    mkldnn_emitter.build_batch_norm_primitive<BatchNormInference>(
                        node, args, out, false /*Append relu*/, false /*Training*/);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormTrainingRelu)
                {
                    std::vector<TensorViewWrapper> args;
                    std::vector<TensorViewWrapper> out;
                    translate_node_inputs_outputs(node, args, out);

                    mkldnn_emitter.build_batch_norm_primitive<BatchNormTrainingRelu>(
                        node, args, out, true /*Append relu*/, true /*Training*/);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormInferenceRelu)
                {
                    std::vector<TensorViewWrapper> args;
                    std::vector<TensorViewWrapper> out;
                    translate_node_inputs_outputs(node, args, out);

                    mkldnn_emitter.build_batch_norm_primitive<BatchNormInferenceRelu>(
                        node, args, out, true /*Append relu*/, false /*Training*/);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormTrainingBackprop)
                {
                    std::vector<TensorViewWrapper> args;
                    std::vector<TensorViewWrapper> out;
                    translate_node_inputs_outputs(node, args, out);

                    auto weights_shape = Shape{2, args[0].get_size()};
                    auto weights_desc = mkldnn_emitter.build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto mean_desc = mkldnn_utils::get_input_mkldnn_md(node, 3);
                    auto variance_desc = mkldnn_utils::get_input_mkldnn_md(node, 4);
                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 5);
                    auto dinput_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto dweights_desc = mkldnn_emitter.build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);

                    const auto* batchnorm = static_cast<const BatchNormTrainingBackprop*>(node);
                    mkldnn_emitter.build_batchnorm_backward(weights_desc,
                                                            input_desc,
                                                            mean_desc,
                                                            variance_desc,
                                                            delta_desc,
                                                            dinput_desc,
                                                            dweights_desc,
                                                            batchnorm->get_eps_value(),
                                                            node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Concat)
                {
                    std::vector<mkldnn::memory::desc> inputs_data_desc;
                    for (size_t i = 0, end = node->get_inputs().size(); i < end; i++)
                    {
                        inputs_data_desc.push_back(mkldnn_utils::get_input_mkldnn_md(node, i));
                    }

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t concat_dim =
                        (static_cast<const Concat*>(node))->get_concatenation_axis();
                    mkldnn_emitter.build_concat(inputs_data_desc, result_desc, concat_dim, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(LRN)
                {
                    auto input_data_desc = runtime::cpu::mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = runtime::cpu::mkldnn_utils::get_output_mkldnn_md(node, 0);

                    const auto* lrn = static_cast<const LRN*>(node);

                    mkldnn_emitter.build_lrn_forward(input_data_desc,
                                                     result_desc,
                                                     static_cast<float>(lrn->get_alpha()),
                                                     static_cast<float>(lrn->get_beta()),
                                                     static_cast<float>(lrn->get_bias()),
                                                     static_cast<int>(lrn->get_nsize()),
                                                     node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Slice)
                {
                    std::vector<TensorViewWrapper> args;
                    std::vector<TensorViewWrapper> out;
                    translate_node_inputs_outputs(node, args, out);

                    const Slice* slice = static_cast<const Slice*>(node);
                    auto out_shape = out[0].get_shape();
                    auto lower_bounds = slice->get_lower_bounds();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    mkldnn_emitter.build_slice(
                        input_desc, result_desc, lower_bounds, out_shape, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionRelu)
                {
                    mkldnn_emitter.build_convolution<ConvolutionRelu>(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConvolutionRelu)
                {
                    mkldnn_emitter.build_convolution<QuantizedConvolutionRelu>(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConvolution)
                {
                    mkldnn_emitter.build_convolution<QuantizedConvolution>(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(GroupConvolution)
                {
                    Strides window_dilation_strides_adjusted;
                    auto convolution = static_cast<const ngraph::op::GroupConvolution*>(node);
                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto padding_below = convolution->get_padding_below();
                    auto padding_above = convolution->get_padding_above();
                    auto filter_strides = convolution->get_window_movement_strides();

                    mkldnn_emitter.build_convolution_forward(input_data_desc,
                                                             weights_desc,
                                                             result_desc,
                                                             filter_strides,
                                                             window_dilation_strides_adjusted,
                                                             padding_below,
                                                             padding_above,
                                                             node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(GroupConvolutionBias)
                {
                    Strides window_dilation_strides_adjusted;
                    auto convolution = static_cast<const ngraph::op::GroupConvolutionBias*>(node);
                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto bias_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto padding_below = convolution->get_padding_below();
                    auto padding_above = convolution->get_padding_above();
                    auto filter_strides = convolution->get_window_movement_strides();

                    const float ops_scale = 1.f;
                    const float ops_alpha = -0.f; // relu negative slope
                    const float ops_beta = 0.f;

                    mkldnn::post_ops ops;
                    if (convolution->with_relu())
                    {
                        ops.append_eltwise(
                            ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    mkldnn_emitter.build_convolution_forward(input_data_desc,
                                                             weights_desc,
                                                             bias_desc,
                                                             result_desc,
                                                             filter_strides,
                                                             window_dilation_strides_adjusted,
                                                             padding_below,
                                                             padding_above,
                                                             node,
                                                             ops);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Convolution)
                {
                    mkldnn_emitter.build_convolution<Convolution>(node);
                }

                template <typename OpTy>
                void build_convolution_backward(MKLDNNEmitter& mkldnn_emitter,
                                                const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OpTy*>(node);

                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto arg0_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto arg1_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto out0_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    if (std::is_same<OpTy, ngraph::op::ConvolutionBackpropData>())
                    {
                        // MKLDNN relies on named formats for kernel selection
                        if (arg0_desc.data.format == mkldnn_nchw)
                        {
                            arg0_desc.data.format = mkldnn_oihw;
                        }
                        if (arg0_desc.data.format == mkldnn_ncdhw)
                        {
                            arg0_desc.data.format = mkldnn_oidhw;
                        }

                        mkldnn_emitter.build_convolution_backward_data(
                            arg0_desc,
                            arg1_desc,
                            out0_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward(),
                            node);
                        return;
                    }
                    if (std::is_same<OpTy, ngraph::op::ConvolutionBackpropFilters>())
                    {
                        mkldnn_emitter.build_convolution_backward_weights(
                            arg0_desc,
                            arg1_desc,
                            out0_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward(),
                            node);
                        return;
                    }
                    if (std::is_same<OpTy, ngraph::op::ConvolutionBiasBackpropFiltersBias>())
                    {
                        auto out1_desc = mkldnn_utils::get_output_mkldnn_md(node, 1);
                        mkldnn_emitter.build_convolution_backward_weights_bias(
                            arg0_desc,
                            arg1_desc,
                            out0_desc,
                            out1_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward(),
                            node);
                        return;
                    }

                    throw ngraph_error(std::string("Unknown op ") + convolution->get_name());
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionBackpropFilters)
                {
                    build_convolution_backward<ConvolutionBackpropFilters>(mkldnn_emitter, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionBackpropData)
                {
                    build_convolution_backward<ConvolutionBackpropData>(mkldnn_emitter, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConvolutionBias)
                {
                    mkldnn_emitter.build_convolution<QuantizedConvolutionBias>(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConvolutionBiasAdd)
                {
                    mkldnn_emitter.build_convolution<QuantizedConvolutionBiasAdd>(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(
                    QuantizedConvolutionBiasSignedAdd)
                {
                    mkldnn_emitter.build_convolution<QuantizedConvolutionBiasSignedAdd>(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionBias)
                {
                    mkldnn_emitter.build_convolution<ConvolutionBias>(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionBiasAdd)
                {
                    mkldnn_emitter.build_convolution<ConvolutionBiasAdd>(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionAdd)
                {
                    mkldnn_emitter.build_convolution<ConvolutionAdd>(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(
                    ConvolutionBiasBackpropFiltersBias)
                {
                    build_convolution_backward<ConvolutionBiasBackpropFiltersBias>(mkldnn_emitter,
                                                                                   node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(MaxPool)
                {
                    auto max_pool = static_cast<const ngraph::op::MaxPool*>(node);
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    mkldnn_emitter.build_pooling_forward(mkldnn::algorithm::pooling_max,
                                                         input_desc,
                                                         result_desc,
                                                         max_pool->get_window_movement_strides(),
                                                         max_pool->get_window_shape(),
                                                         max_pool->get_padding_below(),
                                                         max_pool->get_padding_above(),
                                                         node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedMaxPool)
                {
                    mkldnn_emitter.build_quantized_max_pool(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedAvgPool)
                {
                    mkldnn_emitter.build_quantized_avg_pool(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(MaxPoolWithIndices)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto max_pool = static_cast<const ngraph::op::MaxPoolWithIndices*>(node);

                    mkldnn_emitter.build_max_pooling_with_indices_forward(
                        mkldnn::algorithm::pooling_max,
                        input_desc,
                        result_desc,
                        max_pool->get_window_movement_strides(),
                        max_pool->get_window_shape(),
                        max_pool->get_padding_below(),
                        max_pool->get_padding_above(),
                        node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(AvgPool)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto avg_pool = static_cast<const ngraph::op::AvgPool*>(node);

                    mkldnn_emitter.build_pooling_forward(
                        (avg_pool->get_include_padding_in_avg_computation()
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding),
                        input_desc,
                        result_desc,
                        avg_pool->get_window_movement_strides(),
                        avg_pool->get_window_shape(),
                        avg_pool->get_padding_below(),
                        avg_pool->get_padding_above(),
                        node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(AvgPoolBackprop)
                {
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto apb = static_cast<const ngraph::op::AvgPoolBackprop*>(node);

                    mkldnn_emitter.build_pooling_backward(
                        (apb->get_include_padding_in_avg_computation()
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding),
                        diff_dst_desc,
                        diff_src_desc,
                        apb->get_window_movement_strides(),
                        apb->get_window_shape(),
                        apb->get_padding_below(),
                        apb->get_padding_above(),
                        node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(MaxPoolBackprop)
                {
                    auto fprop_src_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto mpb = static_cast<const ngraph::op::MaxPoolBackprop*>(node);

                    mkldnn_emitter.build_max_pooling_backward(mkldnn::algorithm::pooling_max,
                                                              fprop_src_desc,
                                                              diff_dst_desc,
                                                              diff_src_desc,
                                                              mpb->get_window_movement_strides(),
                                                              mpb->get_window_shape(),
                                                              mpb->get_padding_below(),
                                                              mpb->get_padding_above(),
                                                              node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(MaxPoolWithIndicesBackprop)
                {
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto mpb = static_cast<const ngraph::op::MaxPoolWithIndicesBackprop*>(node);

                    mkldnn_emitter.build_max_pooling_with_indices_backward(
                        mkldnn::algorithm::pooling_max,
                        diff_dst_desc,
                        diff_src_desc,
                        mpb->get_window_movement_strides(),
                        mpb->get_window_shape(),
                        mpb->get_padding_below(),
                        mpb->get_padding_above(),
                        node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(
                    ngraph::runtime::cpu::op::ConvertLayout)
                {
                    std::vector<TensorViewWrapper> args;
                    translate_node_inputs(node, args);

                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    // This is a special case to handle nchw(oihw) to goihw/Goihw16g/Goihw8g for
                    // GroupConvolution's weights.
                    if (input_desc.data.format == mkldnn_nchw &&
                        result_desc.data.format == mkldnn_goihw)
                    {
                        input_desc = result_desc;
                    }
                    else if (input_desc.data.format == mkldnn_nchw &&
                             input_desc.data.ndims == 4 /*nchw*/ &&
                             result_desc.data.ndims == 5 /*Goihw16g/Goihw8g/etc*/ &&
                             node->get_users().size() == 1)
                    {
                        Shape weights_shape_groups;
                        if (auto gconv = std::dynamic_pointer_cast<ngraph::op::GroupConvolution>(
                                node->get_users()[0]))
                        {
                            weights_shape_groups = gconv->get_weights_dimensions();
                        }
                        else if (auto gconvb =
                                     std::dynamic_pointer_cast<ngraph::op::GroupConvolutionBias>(
                                         node->get_users()[0]))
                        {
                            weights_shape_groups = gconvb->get_weights_dimensions();
                        }
                        else
                        {
                            throw ngraph_error(
                                "Incompatible input/output shape in ConvertLayout op");
                        }
                        input_desc = mkldnn::memory::desc(
                            mkldnn::memory::dims(weights_shape_groups.begin(),
                                                 weights_shape_groups.end()),
                            mkldnn_utils::get_mkldnn_data_type(args[0].get_element_type()),
                            mkldnn::memory::format::goihw);
                    }

                    mkldnn_emitter.build_reorder(input_desc, result_desc, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ReluBackprop)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    mkldnn_emitter.build_relu_backward(input_desc, delta_desc, result_desc, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Relu)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    mkldnn_emitter.build_relu_forward(input_desc, result_desc, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(LeakyRelu)
                {
                    auto leaky_relu_node = static_cast<const ngraph::op::LeakyRelu*>(node);
                    float alpha = leaky_relu_node->get_alpha();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    mkldnn_emitter.build_leaky_relu(input_desc, result_desc, alpha, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BoundedRelu)
                {
                    auto bounded_relu_node = static_cast<const ngraph::op::BoundedRelu*>(node);
                    float alpha = bounded_relu_node->get_alpha();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    mkldnn_emitter.build_bounded_relu(input_desc, result_desc, alpha, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Sigmoid)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    mkldnn_emitter.build_sigmoid_forward(input_desc, result_desc, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(SigmoidBackprop)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    mkldnn_emitter.build_sigmoid_backward(
                        input_desc, delta_desc, result_desc, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Softmax)
                {
                    auto softmax = static_cast<const ngraph::op::Softmax*>(node);

                    if (softmax->get_axes().size() != 1)
                    {
                        throw ngraph_error("MKLDNN supports softmax only across single axis");
                    }

                    int softmax_axis = static_cast<int>(*(softmax->get_axes().begin()));
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    mkldnn_emitter.build_softmax_forward(
                        input_desc, result_desc, softmax_axis, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Dequantize)
                {
                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    mkldnn_emitter.build_dequantization(node, input_data_desc, result_desc);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Quantize)
                {
                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto quantize = static_cast<const ngraph::op::Quantize*>(node);
                    auto scale_const_op =
                        std::dynamic_pointer_cast<Constant>(quantize->get_argument(1));

                    if (scale_const_op == nullptr)
                    {
                        throw ngraph_error("Quantize scale must be a constant");
                    }

                    auto scale = scale_const_op->get_vector<float>();
                    std::vector<float> scales;
                    scales.push_back(1.0 / scale[0]);

                    mkldnn_emitter.build_quantize_reorder(
                        input_data_desc, result_desc, scales, node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConcat)
                {
                    int args_size = node->get_inputs().size();

                    std::vector<mkldnn::memory::desc> inputs_data_desc;
                    for (size_t i = 0; i < args_size; i++)
                    {
                        inputs_data_desc.push_back(mkldnn_utils::get_input_mkldnn_md(node, i));
                    }

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t concat_dim =
                        (static_cast<const QuantizedConcat*>(node))->get_concatenation_axis();
                    mkldnn_emitter.build_concat(inputs_data_desc, result_desc, concat_dim, node);
                }
            }
        }
    }
}

using namespace ngraph::runtime::cpu::pass;

#define TI(x) std::type_index(typeid(x))

static const PrimitiveBuildOpMap prim_build_dispatcher{
    {TI(Add), &MKLDNNPrimitiveBuildPass::build_primitive<Add>},
    {TI(Concat), &MKLDNNPrimitiveBuildPass::build_primitive<Concat>},
    {TI(Convert), &MKLDNNPrimitiveBuildPass::build_primitive<Convert>},
    {TI(runtime::cpu::op::ConvertLayout),
     &MKLDNNPrimitiveBuildPass::build_primitive<runtime::cpu::op::ConvertLayout>},
    {TI(AvgPool), &MKLDNNPrimitiveBuildPass::build_primitive<AvgPool>},
    {TI(AvgPoolBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<AvgPoolBackprop>},
    {TI(BatchNormTraining), &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormTraining>},
    {TI(BatchNormInference), &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormInference>},
    {TI(BoundedRelu), &MKLDNNPrimitiveBuildPass::build_primitive<BoundedRelu>},
    {TI(BatchNormTrainingBackprop),
     &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormTrainingBackprop>},
    {TI(Convolution), &MKLDNNPrimitiveBuildPass::build_primitive<Convolution>},
    {TI(GroupConvolution), &MKLDNNPrimitiveBuildPass::build_primitive<GroupConvolution>},
    {TI(ConvolutionRelu), &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionRelu>},
    {TI(ConvolutionBiasAdd), &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBiasAdd>},
    {TI(BatchNormTrainingRelu), &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormTrainingRelu>},
    {TI(BatchNormInferenceRelu),
     &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormInferenceRelu>},
    {TI(ConvolutionBackpropData),
     &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBackpropData>},
    {TI(ConvolutionBackpropFilters),
     &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBackpropFilters>},
    {TI(MaxPool), &MKLDNNPrimitiveBuildPass::build_primitive<MaxPool>},
    {TI(MaxPoolWithIndices), &MKLDNNPrimitiveBuildPass::build_primitive<MaxPoolWithIndices>},
    {TI(MaxPoolBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<MaxPoolBackprop>},
    {TI(MaxPoolWithIndicesBackprop),
     &MKLDNNPrimitiveBuildPass::build_primitive<MaxPoolWithIndicesBackprop>},
    {TI(ConvolutionBias), &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBias>},
    {TI(QuantizedConvolution), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolution>},
    {TI(ConvolutionBiasBackpropFiltersBias),
     &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBiasBackpropFiltersBias>},
    {TI(LRN), &MKLDNNPrimitiveBuildPass::build_primitive<LRN>},
    {TI(Relu), &MKLDNNPrimitiveBuildPass::build_primitive<Relu>},
    {TI(ReluBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<ReluBackprop>},
    {TI(LeakyRelu), &MKLDNNPrimitiveBuildPass::build_primitive<LeakyRelu>},
    {TI(Sigmoid), &MKLDNNPrimitiveBuildPass::build_primitive<Sigmoid>},
    {TI(SigmoidBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<SigmoidBackprop>},
    {TI(Lstm), &MKLDNNPrimitiveBuildPass::build_primitive<Lstm>},
    {TI(Rnn), &MKLDNNPrimitiveBuildPass::build_primitive<Rnn>},
    {TI(QuantizedMaxPool), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedMaxPool>},
    {TI(QuantizedAvgPool), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedAvgPool>},
    {TI(Softmax), &MKLDNNPrimitiveBuildPass::build_primitive<Softmax>},
    {TI(Slice), &MKLDNNPrimitiveBuildPass::build_primitive<Slice>},
    {TI(ReplaceSlice), &MKLDNNPrimitiveBuildPass::build_primitive<ReplaceSlice>},
    {TI(UpdateSlice), &MKLDNNPrimitiveBuildPass::build_primitive<UpdateSlice>},
    {TI(ConvolutionAdd), &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionAdd>},
    {TI(QuantizedConvolutionRelu),
     &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolutionRelu>},
    {TI(QuantizedConvolutionBias),
     &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolutionBias>},
    {TI(QuantizedConvolutionBiasAdd),
     &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolutionBiasAdd>},
    {TI(QuantizedConvolutionBiasSignedAdd),
     &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolutionBiasSignedAdd>},
    {TI(GroupConvolutionBias), &MKLDNNPrimitiveBuildPass::build_primitive<GroupConvolutionBias>},
    {TI(Quantize), &MKLDNNPrimitiveBuildPass::build_primitive<Quantize>},
    {TI(Dequantize), &MKLDNNPrimitiveBuildPass::build_primitive<Dequantize>},
    {TI(QuantizedConcat), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConcat>},
    {TI(GetOutputElement), &MKLDNNPrimitiveBuildPass::build_primitive<GetOutputElement>},
};

bool runtime::cpu::pass::MKLDNNPrimitiveBuildPass::run_on_call_graph(
    const std::list<std::shared_ptr<Node>>& nodes)
{
    for (const auto& shp_node : nodes)
    {
        Node* node = shp_node.get();

        if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
        {
            auto handler = prim_build_dispatcher.find(TI(*node));
            NGRAPH_ASSERT(handler != prim_build_dispatcher.end())
                << "Unsupported node '" << node->description() << "' in MKLDNNPrimitiveBuildPass";

            handler->second(m_mkldnn_emitter, node);
        }
    }

    return false;
}

#undef TI
