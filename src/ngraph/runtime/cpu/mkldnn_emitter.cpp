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
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/sigmoid.hpp"
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
#include "ngraph/runtime/cpu/op/update_slice.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;
using namespace ngraph::op;
using namespace ngraph::runtime::cpu;

MKLDNNEmitter::~MKLDNNEmitter()
{
    for (auto p : m_mkldnn_primitives)
        delete p;
    for (auto s : m_mkldnn_scratchpad_mds)
        delete s;
#ifndef _WIN32
    // To avoid memory leak in mkldnn, release any buffers that are not free'd yet.
    // https://software.intel.com/en-us/mkl-linux-developer-guide-avoiding-memory-leaks-in-intel-mkl
    // mkl_free_buffers() is not exposed at this point, hence using mkl_serv_free_buffers()
    mkldnn_utils::mkl_serv_free_buffers();
#endif
}

std::vector<mkldnn::primitive*>& MKLDNNEmitter::get_mkldnn_primitives()
{
    return m_mkldnn_primitives;
}

const std::vector<mkldnn::memory*>& MKLDNNEmitter::get_mkldnn_memories() const
{
    return m_mkldnn_memories;
}

const std::vector<char*>& MKLDNNEmitter::get_mkldnn_workspaces()
{
    return m_workspace_bufs;
}

const std::vector<mkldnn::memory::desc*>& MKLDNNEmitter::get_mkldnn_scratchpad_mds() const
{
    return m_mkldnn_scratchpad_mds;
}

size_t MKLDNNEmitter::insert_primitive(mkldnn::primitive* primitive)
{
    m_mkldnn_primitives.emplace_back(primitive);
    return (m_mkldnn_primitives.size() - 1);
}

size_t MKLDNNEmitter::insert_memory(mkldnn::memory* memory)
{
    m_mkldnn_memories.emplace_back(memory);
    return (m_mkldnn_memories.size() - 1);
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

size_t MKLDNNEmitter::insert_scratchpad_md(mkldnn::memory::desc* md)
{
    m_mkldnn_scratchpad_mds.emplace_back(md);
    return (m_mkldnn_scratchpad_mds.size() - 1);
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

std::vector<size_t>& MKLDNNEmitter::get_primitive_deps(size_t index)
{
    return m_primitive_deps.at(index);
}

size_t MKLDNNEmitter::get_max_scratchpad_size() const
{
    return m_max_scratchpad_size;
}

mkldnn::memory::desc
    MKLDNNEmitter::build_blocked_memory_descriptor(const mkldnn::memory::dims& dim,
                                                   const mkldnn::memory::dims& strides,
                                                   mkldnn::memory::data_type dtype) const
{
#if MKLDNN_VERSION_MAJOR >= 1
    return mkldnn::memory::desc(dim, dtype, strides);
#else
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
#endif
}

size_t MKLDNNEmitter::build_quantize_reorder(const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& result_desc,
                                             const std::vector<float>& scales)
{
    mkldnn::primitive_attr attr;
    attr.set_output_scales(0, scales);
    size_t input_index, result_index, primitive_index;

#if MKLDNN_VERSION_MAJOR >= 1
    input_index = build_memory(input_desc);
    result_index = build_memory(result_desc);
    auto reorder_prim_desc = mkldnn::reorder::primitive_desc(
        executor::global_cpu_engine, input_desc, executor::global_cpu_engine, result_desc, attr);
    primitive_index = insert_primitive(new mkldnn::reorder(reorder_prim_desc));
#else
    input_index = build_memory_primitive(input_desc);
    result_index = build_memory_primitive(result_desc);

    attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
    auto reorder_desc = mkldnn::reorder::primitive_desc({input_desc, executor::global_cpu_engine},
                                                        {result_desc, executor::global_cpu_engine},
                                                        attr);
    primitive_index = insert_primitive(new mkldnn::reorder(
        reorder_desc, *m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]));
#endif

    NGRAPH_CHECK(m_primitive_deps.find(primitive_index) == m_primitive_deps.end(),
                 "Dependencies already created for node");

    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
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

size_t MKLDNNEmitter::build_reorder(const mkldnn::memory::desc& input_desc,
                                    const mkldnn::memory::desc& result_desc)
{
    size_t input_index, result_index, primitive_index;

#if MKLDNN_VERSION_MAJOR >= 1
    input_index = build_memory(input_desc);
    result_index = build_memory(result_desc);
    primitive_index = insert_primitive(
        new mkldnn::reorder(*m_mkldnn_memories[input_index], *m_mkldnn_memories[result_index]));

#else
    input_index = build_memory_primitive(input_desc);
    result_index = build_memory_primitive(result_desc);
    primitive_index = insert_primitive(
        new mkldnn::reorder(*m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]));
#endif
    NGRAPH_CHECK(m_primitive_deps.find(primitive_index) == m_primitive_deps.end(),
                 "Dependencies already created for node");

    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
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

mkldnn::eltwise_forward::desc MKLDNNEmitter::get_relu_forward_desc(const ngraph::Node* node)
{
    const float negative_slope = 0.0f;

    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);

    return mkldnn::eltwise_forward::desc(
        mkldnn::prop_kind::forward, mkldnn::algorithm::eltwise_relu, input_desc, negative_slope);
}

mkldnn::eltwise_backward::desc MKLDNNEmitter::get_relu_backward_desc(const ngraph::Node* node)
{
    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

    const float negative_slope = 0.0f;
    return mkldnn::eltwise_backward::desc(
        mkldnn::algorithm::eltwise_relu, result_desc, input_desc, negative_slope);
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

mkldnn::eltwise_backward::desc MKLDNNEmitter::get_sigmoid_backward_desc(const ngraph::Node* node)
{
    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);

    return mkldnn::eltwise_backward::desc(
        mkldnn::algorithm::eltwise_logistic, delta_desc, input_desc, 0, 0);
}

mkldnn::batch_normalization_backward::desc
    MKLDNNEmitter::get_batchnorm_backward_desc(const ngraph::Node* node)
{
    const ngraph::op::BatchNormTrainingBackprop* batchnorm =
        static_cast<const ngraph::op::BatchNormTrainingBackprop*>(node);
    auto eps = batchnorm->get_eps_value();

    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 5);

    return mkldnn::batch_normalization_backward::desc(mkldnn::prop_kind::backward,
                                                      delta_desc,
                                                      input_desc,
                                                      eps,
                                                      mkldnn::BN_FLAG_CLASS::use_scale_shift);
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

size_t MKLDNNEmitter::convolution_forward_init(bool with_bias)
{
    size_t size = m_mkldnn_primitives.size();
    if (with_bias)
    {
// Inputs, Weights, Bias, Results, Conv
#if MKLDNN_VERSION_MAJOR >= 1
        size_t mem_size = m_mkldnn_memories.size();
        m_mkldnn_primitives.resize(size + 1, nullptr);
        m_mkldnn_scratchpad_mds.resize(size + 1, nullptr);
        m_mkldnn_memories.resize(mem_size + 4, nullptr);
        m_primitive_deps[m_mkldnn_primitives.size() - 1] = {
            mem_size, mem_size + 1, mem_size + 2, mem_size + 3};
#else
        m_mkldnn_primitives.resize(size + 5, nullptr);
        m_primitive_deps[m_mkldnn_primitives.size() - 1] = {size, size + 1, size + 2, size + 3};
#endif
    }
    else
    {
// Inputs, Weights, Results, Conv
#if MKLDNN_VERSION_MAJOR >= 1
        size_t mem_size = m_mkldnn_memories.size();
        m_mkldnn_primitives.resize(size + 1, nullptr);
        m_mkldnn_scratchpad_mds.resize(size + 1, nullptr);
        m_mkldnn_memories.resize(mem_size + 3, nullptr);
        m_primitive_deps[m_mkldnn_primitives.size() - 1] = {mem_size, mem_size + 1, mem_size + 2};
#else
        m_mkldnn_primitives.resize(size + 4, nullptr);
        m_primitive_deps[m_mkldnn_primitives.size() - 1] = {size, size + 1, size + 2};
#endif
    }
    return m_mkldnn_primitives.size() - 1;
}

size_t MKLDNNEmitter::inner_product_forward_init(bool with_bias)
{
    size_t size = m_mkldnn_primitives.size();
    if (with_bias)
    {
// Inputs, Weights, Bias, Results, inner_product
#if MKLDNN_VERSION_MAJOR >= 1
        size_t mem_size = m_mkldnn_memories.size();
        m_mkldnn_primitives.resize(size + 1, nullptr);
        m_mkldnn_scratchpad_mds.resize(size + 1, nullptr);
        m_mkldnn_memories.resize(mem_size + 4, nullptr);
        m_primitive_deps[m_mkldnn_primitives.size() - 1] = {
            mem_size, mem_size + 1, mem_size + 2, mem_size + 3};
#else
        m_mkldnn_primitives.resize(size + 5, nullptr);
        m_primitive_deps[m_mkldnn_primitives.size() - 1] = {size, size + 1, size + 2, size + 3};
#endif
    }
    else
    {
// Inputs, Weights, Results, inner_product
#if MKLDNN_VERSION_MAJOR >= 1
        size_t mem_size = m_mkldnn_memories.size();
        m_mkldnn_primitives.resize(size + 1, nullptr);
        m_mkldnn_scratchpad_mds.resize(size + 1, nullptr);
        m_mkldnn_memories.resize(mem_size + 3, nullptr);
        m_primitive_deps[m_mkldnn_primitives.size() - 1] = {mem_size, mem_size + 1, mem_size + 2};
#else
        m_mkldnn_primitives.resize(size + 4, nullptr);
        m_primitive_deps[m_mkldnn_primitives.size() - 1] = {size, size + 1, size + 2};
#endif
    }
    return m_mkldnn_primitives.size() - 1;
}

size_t MKLDNNEmitter::reserve_primitive_space(size_t count, bool new_workspace)
{
    size_t size = m_mkldnn_primitives.size();
#if MKLDNN_VERSION_MAJOR >= 1
    size_t mem_size = m_mkldnn_memories.size();
    m_mkldnn_primitives.resize(size + 1, nullptr);
    m_mkldnn_scratchpad_mds.resize(size + 1, nullptr);
    m_mkldnn_memories.resize(mem_size + count - 1, nullptr);
    for (auto i = 0; i < count - 1; i++)
    {
        m_primitive_deps[m_mkldnn_primitives.size() - 1].push_back(mem_size + i);
    }
#else
    m_mkldnn_primitives.resize(size + count, nullptr);
    for (auto i = 0; i < count - 1; i++)
    {
        m_primitive_deps[m_mkldnn_primitives.size() - 1].push_back(size + i);
    }
#endif

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
    std::vector<float> output_scale;
    output_scale.push_back(scale);
    // mkldnn inner_product attr
    mkldnn::primitive_attr ip_attr;
    ip_attr.set_post_ops(pops);
    /* Specify the scales array and corresponding mask */
    ip_attr.set_output_scales(0, output_scale);

    size_t ip_index;
#if MKLDNN_VERSION_MAJOR >= 1
    size_t input_data_index = build_memory(input_data_desc);
    size_t weights_index = build_memory(weights_desc);
    size_t bias_index = build_memory(bias_desc);
    size_t result_index = build_memory(result_desc);

    // mkldnn inner_product
    ip_index =
        insert_primitive(new mkldnn::inner_product_forward({{
                                                                mkldnn::prop_kind::forward_scoring,
                                                                input_data_desc,
                                                                weights_desc,
                                                                bias_desc,
                                                                result_desc,
                                                            },
                                                            ip_attr,
                                                            executor::global_cpu_engine}));
    m_primitive_deps[ip_index] = {input_data_index, weights_index, bias_index, result_index};
#else
    size_t input_data_index = build_memory_primitive(input_data_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t bias_index = build_memory_primitive(bias_desc);
    size_t result_index = build_memory_primitive(result_desc);

    /* Specify the rounding mode */
    ip_attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);

    // mkldnn inner_product
    ip_index =
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
#endif
    return ip_index;
}

size_t MKLDNNEmitter::build_quantized_inner_product_forward(
    const mkldnn::memory::desc& input_data_desc,
    const mkldnn::memory::desc& weights_desc,
    const mkldnn::memory::desc& result_desc,
    const float scale,
    const mkldnn::post_ops& pops)
{
    std::vector<float> output_scale;
    output_scale.push_back(scale);
    // mkldnn inner_product attr
    mkldnn::primitive_attr ip_attr;
    ip_attr.set_post_ops(pops);
    /* Specify the scales array and corresponding mask */
    ip_attr.set_output_scales(0, output_scale);

    size_t ip_index;
#if MKLDNN_VERSION_MAJOR >= 1
    size_t input_data_index = build_memory(input_data_desc);
    size_t weights_index = build_memory(weights_desc);
    size_t result_index = build_memory(result_desc);

    // mkldnn inner_product
    ip_index = insert_primitive(new mkldnn::inner_product_forward(
        {{
             mkldnn::prop_kind::forward_scoring, input_data_desc, weights_desc, result_desc,
         },
         ip_attr,
         executor::global_cpu_engine}));
    m_primitive_deps[ip_index] = {input_data_index, weights_index, result_index};
#else
    size_t input_data_index = build_memory_primitive(input_data_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t result_index = build_memory_primitive(result_desc);

    /* Specify the rounding mode */
    ip_attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
    // mkldnn inner_product
    ip_index = insert_primitive(new mkldnn::inner_product_forward(
        {{
             mkldnn::prop_kind::forward_scoring, input_data_desc, weights_desc, result_desc,
         },
         ip_attr,
         executor::global_cpu_engine},
        *m_mkldnn_primitives[input_data_index],
        *m_mkldnn_primitives[weights_index],
        *m_mkldnn_primitives[result_index]));
    m_primitive_deps[ip_index] = {input_data_index, weights_index, result_index};
#endif
    return ip_index;
}

#if MKLDNN_VERSION_MAJOR >= 1
mkldnn::memory::desc
    MKLDNNEmitter::build_memory_descriptor(const TensorViewWrapper& tvw,
                                           mkldnn::memory::format_tag fmt_tag) const
{
    return mkldnn::memory::desc(
        mkldnn::memory::dims(tvw.get_shape().begin(), tvw.get_shape().end()),
        mkldnn_utils::get_mkldnn_data_type(tvw.get_element_type()),
        fmt_tag);
}

mkldnn::memory::desc MKLDNNEmitter::build_memory_descriptor(const ngraph::Shape& shape,
                                                            const ngraph::element::Type& et,
                                                            mkldnn::memory::format_tag fmt) const
{
    return mkldnn::memory::desc(mkldnn::memory::dims(shape.begin(), shape.end()),
                                mkldnn_utils::get_mkldnn_data_type(et),
                                fmt);
}

size_t MKLDNNEmitter::build_memory(const mkldnn::memory::desc& desc)
{
    size_t index = insert_memory(new mkldnn::memory(desc, executor::global_cpu_engine, nullptr));
    return index;
}

void MKLDNNEmitter::build_memory(const mkldnn::memory::desc& desc, size_t index)
{
    m_mkldnn_memories[index] = new mkldnn::memory(desc, executor::global_cpu_engine, nullptr);
}

void MKLDNNEmitter::build_memory(std::vector<mkldnn::memory*>& mkldnn_memories,
                                 const mkldnn::memory::desc& desc,
                                 size_t index)
{
    mkldnn_memories[index] = new mkldnn::memory(desc, executor::global_cpu_engine, nullptr);
}

mkldnn::sum::primitive_desc MKLDNNEmitter::get_elementwise_add_desc(const ngraph::Node* node)
{
    std::vector<float> scale_vector(2, 1);
    auto input0_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
    auto input1_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
    std::vector<mkldnn::memory::desc> inputs_desc{input0_data_desc, input1_data_desc};

    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);

    // elementwise sum primtive descriptor
    mkldnn::sum::primitive_desc sum_pd = mkldnn::sum::primitive_desc(
        result_desc, scale_vector, inputs_desc, executor::global_cpu_engine, attr);

    return sum_pd;
}

void MKLDNNEmitter::build_quantize_reorder(
    std::vector<mkldnn::memory*>& mkldnn_memories,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
    const mkldnn::memory::desc& input_desc,
    const mkldnn::memory::desc& result_desc,
    const std::vector<float>& scales,
    const std::vector<size_t>& deps,
    size_t quantize_index,
    const int mask)
{
    size_t input_index = deps[0];
    build_memory(mkldnn_memories, input_desc, input_index);
    size_t result_index = deps[1];
    build_memory(mkldnn_memories, result_desc, result_index);

    mkldnn::primitive_attr attr;
    attr.set_output_scales(mask, scales);
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto reorder_prim_desc = mkldnn::reorder::primitive_desc(
        executor::global_cpu_engine, input_desc, executor::global_cpu_engine, result_desc, attr);

    mkldnn_scratchpad_mds[quantize_index] =
        new mkldnn::memory::desc(reorder_prim_desc.scratchpad_desc());
    mkldnn_primitives[quantize_index] = new mkldnn::reorder(reorder_prim_desc);
}

#if 0
mkldnn::memory::format_tag MKLDNNEmitter::query_convolution_forward_weight_format_tag(
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
    mkldnn::convolution_forward::desc conv_desc_layout(
        mkldnn::prop_kind::forward_inference,
        convolution_algo,
        input_data_desc,
        weights_desc_any, // this needs to be in default format
        result_desc,
        mkldnn_filter_strides,
        mkldnn_dilated_strides,
        mkldnn_padding_below,
        mkldnn_padding_above);

    mkldnn::convolution_forward::primitive_desc prim_desc(conv_desc_layout, executor::global_cpu_engine);
    return static_cast<mkldnn::memory::format_tag>(
        prim_desc.weights_primitive_desc().desc().data.format_tag);
}
#endif

void MKLDNNEmitter::build_deconvolutionbias_forward(
    std::vector<mkldnn::memory*>& mkldnn_memories,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
    const mkldnn::deconvolution_forward::desc& deconv_desc,
    const std::vector<size_t>& deps,
    size_t deconv_index,
    const mkldnn::memory::desc& weights_desc)
{
    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto deconv_pd = mkldnn::deconvolution_forward::primitive_desc(
        deconv_desc, attr, executor::global_cpu_engine);
    mkldnn_scratchpad_mds[deconv_index] = new mkldnn::memory::desc(deconv_pd.scratchpad_desc());

    size_t weights_index = deps[0];
    build_memory(mkldnn_memories, weights_desc, weights_index);
    size_t delta_index = deps[1];
    build_memory(mkldnn_memories, deconv_pd.src_desc(), delta_index);
    size_t bias_index = deps[2];
    build_memory(mkldnn_memories, deconv_pd.bias_desc(), bias_index);
    size_t result_index = deps[3];
    build_memory(mkldnn_memories, deconv_pd.dst_desc(), result_index);

    mkldnn_primitives[deconv_index] = new mkldnn::deconvolution_forward(deconv_pd);
}

void MKLDNNEmitter::build_convolution_backward_weights_bias(
    std::vector<mkldnn::memory*>& mkldnn_memories,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
    const mkldnn::convolution_backward_weights::desc& bwd_desc,
    const mkldnn::convolution_forward::desc& fwd_desc,
    const std::vector<size_t>& deps,
    size_t conv_index)
{
    auto conv_fwd_pd =
        mkldnn::convolution_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto conv_bwd_pd = mkldnn::convolution_backward_weights::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, conv_fwd_pd);
    mkldnn_scratchpad_mds[conv_index] = new mkldnn::memory::desc(conv_bwd_pd.scratchpad_desc());

    size_t src_index = deps[0];
    build_memory(mkldnn_memories, conv_bwd_pd.src_desc(), src_index);
    size_t diff_dst_index = deps[1];
    build_memory(mkldnn_memories, conv_bwd_pd.diff_dst_desc(), diff_dst_index);
    size_t diff_weights_index = deps[2];
    build_memory(mkldnn_memories, conv_bwd_pd.diff_weights_desc(), diff_weights_index);
    size_t diff_bias_index = deps[3];
    build_memory(mkldnn_memories, conv_bwd_pd.diff_bias_desc(), diff_bias_index);

    mkldnn_primitives[conv_index] = new mkldnn::convolution_backward_weights(conv_bwd_pd);
}

void MKLDNNEmitter::build_convolution_backward_weights(
    std::vector<mkldnn::memory*>& mkldnn_memories,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
    const mkldnn::convolution_backward_weights::desc& bwd_desc,
    const mkldnn::convolution_forward::desc& fwd_desc,
    const std::vector<size_t>& deps,
    size_t conv_index)
{
    // Forward primitive descriptor corresponding to this backward weights descriptor
    auto conv_fwd_pd =
        mkldnn::convolution_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto conv_bwd_pd = mkldnn::convolution_backward_weights::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, conv_fwd_pd);
    mkldnn_scratchpad_mds[conv_index] = new mkldnn::memory::desc(conv_bwd_pd.scratchpad_desc());

    size_t src_index = deps[0];
    build_memory(mkldnn_memories, conv_bwd_pd.src_desc(), src_index);
    size_t diff_dst_index = deps[1];
    build_memory(mkldnn_memories, conv_bwd_pd.diff_dst_desc(), diff_dst_index);
    size_t diff_weights_index = deps[2];
    build_memory(mkldnn_memories, conv_bwd_pd.diff_weights_desc(), diff_weights_index);

    mkldnn_primitives[conv_index] = new mkldnn::convolution_backward_weights(conv_bwd_pd);
}

void MKLDNNEmitter::build_convolution_backward_data(
    std::vector<mkldnn::memory*>& mkldnn_memories,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
    const mkldnn::convolution_backward_data::desc& bwd_desc,
    const mkldnn::convolution_forward::desc& fwd_desc,
    const std::vector<size_t>& deps,
    size_t conv_index)
{
    // Forward primitive descriptor corresponding to this backward weights descriptor
    auto conv_fwd_pd =
        mkldnn::convolution_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto conv_bwd_pd = mkldnn::convolution_backward_data::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, conv_fwd_pd);
    mkldnn_scratchpad_mds[conv_index] = new mkldnn::memory::desc(conv_bwd_pd.scratchpad_desc());

    size_t weights_index = deps[0];
    build_memory(mkldnn_memories, conv_bwd_pd.weights_desc(), weights_index);
    size_t diff_dst_index = deps[1];
    build_memory(mkldnn_memories, conv_bwd_pd.diff_dst_desc(), diff_dst_index);
    size_t diff_src_index = deps[2];
    build_memory(mkldnn_memories, conv_bwd_pd.diff_src_desc(), diff_src_index);

    mkldnn_primitives[conv_index] = new mkldnn::convolution_backward_data(conv_bwd_pd);
}

void MKLDNNEmitter::build_pooling_forward(std::vector<mkldnn::memory*>& mkldnn_memories,
                                          std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                          std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                          const mkldnn::pooling_forward::desc& pool_desc,
                                          const std::vector<size_t>& deps,
                                          size_t pool_index)
{
    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto pool_pd =
        mkldnn::pooling_forward::primitive_desc(pool_desc, attr, executor::global_cpu_engine);
    mkldnn_scratchpad_mds[pool_index] = new mkldnn::memory::desc(pool_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(mkldnn_memories, pool_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(mkldnn_memories, pool_pd.dst_desc(), result_index);

    mkldnn_primitives[pool_index] = new mkldnn::pooling_forward(pool_pd);
}

void MKLDNNEmitter::build_pooling_backward(
    std::vector<mkldnn::memory*>& mkldnn_memories,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
    const mkldnn::pooling_backward::desc& pool_desc,
    const mkldnn::pooling_forward::desc& pool_fwd_desc,
    const std::vector<size_t>& deps,
    size_t pool_index)
{
    auto pool_fwd_pd =
        mkldnn::pooling_forward::primitive_desc(pool_fwd_desc, executor::global_cpu_engine);

    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto pool_bwd_pd = mkldnn::pooling_backward::primitive_desc(
        pool_desc, attr, executor::global_cpu_engine, pool_fwd_pd);
    mkldnn_scratchpad_mds[pool_index] = new mkldnn::memory::desc(pool_bwd_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(mkldnn_memories, pool_bwd_pd.diff_dst_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(mkldnn_memories, pool_bwd_pd.diff_src_desc(), result_index);

    mkldnn_primitives[pool_index] = new mkldnn::pooling_backward(pool_bwd_pd);
}

void MKLDNNEmitter::build_max_pooling_backward(
    std::vector<mkldnn::memory*>& mkldnn_memories,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
    std::vector<char*>& mkldnn_workspaces,
    const mkldnn::pooling_backward::desc& bwd_pool_desc,
    const mkldnn::pooling_forward::desc& fwd_pool_desc,
    const mkldnn::memory::desc& fprop_src_desc,
    std::vector<size_t>& fdeps,
    std::vector<size_t>& bdeps,
    size_t fwd_pool_index,
    size_t bwd_pool_index)
{
    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto pool_fwd_pd =
        mkldnn::pooling_forward::primitive_desc(fwd_pool_desc, attr, executor::global_cpu_engine);
    auto pool_bwd_pd = mkldnn::pooling_backward::primitive_desc(
        bwd_pool_desc, attr, executor::global_cpu_engine, pool_fwd_pd);
    mkldnn_scratchpad_mds[fwd_pool_index] = new mkldnn::memory::desc(pool_fwd_pd.scratchpad_desc());
    mkldnn_scratchpad_mds[bwd_pool_index] = new mkldnn::memory::desc(pool_bwd_pd.scratchpad_desc());

    size_t fprop_src_index = fdeps[0];
    build_memory(mkldnn_memories, fprop_src_desc, fprop_src_index);
    size_t diff_dst_index = bdeps[0];
    build_memory(mkldnn_memories, pool_bwd_pd.diff_dst_desc(), diff_dst_index);
    size_t diff_src_index = fdeps[1];
    build_memory(mkldnn_memories, pool_bwd_pd.diff_src_desc(), diff_src_index);
    bdeps[2] = diff_src_index;

    size_t ws_index = fdeps[2];
    build_memory(mkldnn_memories, pool_fwd_pd.workspace_desc(), ws_index);
    bdeps[1] = ws_index;

    // Allocate workspace
    // TODO (jbobba): Might need to align memory
    auto ws = std::unique_ptr<MKLDNNWorkspace>(
        new MKLDNNWorkspace(pool_fwd_pd.workspace_desc().get_size()));
    auto ws_buf_index = insert_workspace(mkldnn_workspaces, ws);
    fdeps[3] = ws_buf_index;
    bdeps[3] = ws_buf_index;

    mkldnn_primitives[fwd_pool_index] = new mkldnn::pooling_forward(pool_fwd_pd);

    mkldnn_primitives[bwd_pool_index] = new mkldnn::pooling_backward(pool_bwd_pd);
}

void MKLDNNEmitter::build_max_pooling_with_indices_forward(
    std::vector<mkldnn::memory*>& mkldnn_memories,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
    const mkldnn::pooling_forward::desc& max_pool_desc,
    const std::vector<size_t>& deps,
    size_t max_pool_index)
{
    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto pool_pd =
        mkldnn::pooling_forward::primitive_desc(max_pool_desc, attr, executor::global_cpu_engine);
    mkldnn_scratchpad_mds[max_pool_index] = new mkldnn::memory::desc(pool_pd.scratchpad_desc());

    size_t src_index = deps[0];
    build_memory(mkldnn_memories, pool_pd.src_desc(), src_index);
    size_t dst_index = deps[1];
    build_memory(mkldnn_memories, pool_pd.dst_desc(), dst_index);

    size_t ws_index = deps[2];
    build_memory(mkldnn_memories, pool_pd.workspace_desc(), ws_index);

    mkldnn_primitives[max_pool_index] = new mkldnn::pooling_forward(pool_pd);
}

void MKLDNNEmitter::build_max_pooling_with_indices_backward(
    std::vector<mkldnn::memory*>& mkldnn_memories,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
    const mkldnn::pooling_backward::desc& bwd_pool_desc,
    const mkldnn::pooling_forward::desc& fwd_pool_desc,
    const std::vector<size_t>& deps,
    size_t max_pool_index)
{
    auto pool_fwd_pd =
        mkldnn::pooling_forward::primitive_desc(fwd_pool_desc, executor::global_cpu_engine);

    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto pool_bwd_pd = mkldnn::pooling_backward::primitive_desc(
        bwd_pool_desc, attr, executor::global_cpu_engine, pool_fwd_pd);
    mkldnn_scratchpad_mds[max_pool_index] = new mkldnn::memory::desc(pool_bwd_pd.scratchpad_desc());

    size_t diff_dst_index = deps[0];
    build_memory(mkldnn_memories, pool_bwd_pd.diff_dst_desc(), diff_dst_index);
    size_t diff_src_index = deps[2];
    build_memory(mkldnn_memories, pool_bwd_pd.diff_src_desc(), diff_src_index);

    size_t fprop_ws_index = deps[1];
    build_memory(mkldnn_memories, pool_fwd_pd.workspace_desc(), fprop_ws_index);

    mkldnn_primitives[max_pool_index] = new mkldnn::pooling_backward(pool_bwd_pd);
}

void MKLDNNEmitter::build_reorder(std::vector<mkldnn::memory*>& mkldnn_memories,
                                  std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                  std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                  const mkldnn::memory::desc& input_desc,
                                  const mkldnn::memory::desc& result_desc,
                                  const std::vector<size_t>& deps,
                                  size_t reorder_index)
{
    size_t input_index = deps[0];
    build_memory(mkldnn_memories, input_desc, input_index);
    size_t result_index = deps[1];
    build_memory(mkldnn_memories, result_desc, result_index);

    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto reorder_pd = mkldnn::reorder::primitive_desc(
        *mkldnn_memories[input_index], *mkldnn_memories[result_index], attr);
    mkldnn_scratchpad_mds[reorder_index] = new mkldnn::memory::desc(reorder_pd.scratchpad_desc());
    mkldnn_primitives[reorder_index] = new mkldnn::reorder(reorder_pd);
}

void MKLDNNEmitter::build_lrn_forward(std::vector<mkldnn::memory*>& mkldnn_memories,
                                      std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                      std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                      const mkldnn::lrn_forward::desc& lrn_desc,
                                      const std::vector<size_t>& deps,
                                      size_t lrn_index)
{
    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto lrn_pd = mkldnn::lrn_forward::primitive_desc(lrn_desc, attr, executor::global_cpu_engine);
    mkldnn_scratchpad_mds[lrn_index] = new mkldnn::memory::desc(lrn_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(mkldnn_memories, lrn_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(mkldnn_memories, lrn_pd.dst_desc(), result_index);

    mkldnn_primitives[lrn_index] = new mkldnn::lrn_forward(lrn_pd);
}

void MKLDNNEmitter::build_relu_forward(std::vector<mkldnn::memory*>& mkldnn_memories,
                                       std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                       std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                       const mkldnn::eltwise_forward::desc& relu_desc,
                                       const std::vector<size_t>& deps,
                                       size_t relu_index)
{
    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto relu_pd =
        mkldnn::eltwise_forward::primitive_desc(relu_desc, attr, executor::global_cpu_engine);
    mkldnn_scratchpad_mds[relu_index] = new mkldnn::memory::desc(relu_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(mkldnn_memories, relu_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(mkldnn_memories, relu_pd.dst_desc(), result_index);

    mkldnn_primitives[relu_index] = new mkldnn::eltwise_forward(relu_pd);
}

void MKLDNNEmitter::build_relu_backward(std::vector<mkldnn::memory*>& mkldnn_memories,
                                        std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                        std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                        const mkldnn::eltwise_backward::desc& bwd_desc,
                                        const mkldnn::eltwise_forward::desc& fwd_desc,
                                        const std::vector<size_t>& deps,
                                        size_t relu_index)
{
    /* create forward relu primitive descriptor*/
    auto relu_fwd_pd =
        mkldnn::eltwise_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    /* create backward relu primitive_descriptor */
    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto relu_bwd_pd = mkldnn::eltwise_backward::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, relu_fwd_pd);
    mkldnn_scratchpad_mds[relu_index] = new mkldnn::memory::desc(relu_bwd_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(mkldnn_memories, relu_bwd_pd.src_desc(), input_index);
    size_t delta_index = deps[1];
    build_memory(mkldnn_memories, relu_bwd_pd.diff_dst_desc(), delta_index);
    size_t result_index = deps[2];
    build_memory(mkldnn_memories, relu_bwd_pd.diff_src_desc(), result_index);

    mkldnn_primitives[relu_index] = new mkldnn::eltwise_backward(relu_bwd_pd);
}

void MKLDNNEmitter::build_sigmoid_forward(std::vector<mkldnn::memory*>& mkldnn_memories,
                                          std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                          std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                          const mkldnn::eltwise_forward::desc& sigmoid_desc,
                                          const std::vector<size_t>& deps,
                                          size_t sigmoid_index)
{
    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto sigmoid_pd =
        mkldnn::eltwise_forward::primitive_desc(sigmoid_desc, attr, executor::global_cpu_engine);
    mkldnn_scratchpad_mds[sigmoid_index] = new mkldnn::memory::desc(sigmoid_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(mkldnn_memories, sigmoid_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(mkldnn_memories, sigmoid_pd.dst_desc(), result_index);

    mkldnn_primitives[sigmoid_index] = new mkldnn::eltwise_forward(sigmoid_pd);
}

void MKLDNNEmitter::build_sigmoid_backward(
    std::vector<mkldnn::memory*>& mkldnn_memories,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
    const mkldnn::eltwise_backward::desc& bwd_desc,
    const mkldnn::eltwise_forward::desc& fwd_desc,
    const std::vector<size_t>& deps,
    size_t sigmoid_index)
{
    // sigmoid forward primitive desc
    auto sigmoid_fwd_pd =
        mkldnn::eltwise_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto sigmoid_bwd_pd = mkldnn::eltwise_backward::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, sigmoid_fwd_pd);
    mkldnn_scratchpad_mds[sigmoid_index] =
        new mkldnn::memory::desc(sigmoid_bwd_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(mkldnn_memories, sigmoid_bwd_pd.src_desc(), input_index);
    size_t delta_index = deps[1];
    build_memory(mkldnn_memories, sigmoid_bwd_pd.diff_dst_desc(), delta_index);
    size_t result_index = deps[2];
    build_memory(mkldnn_memories, sigmoid_bwd_pd.diff_dst_desc(), result_index);

    mkldnn_primitives[sigmoid_index] = new mkldnn::eltwise_backward(sigmoid_bwd_pd);
}

void MKLDNNEmitter::build_elementwise_add(std::vector<mkldnn::memory*>& mkldnn_memories,
                                          std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                          std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                          const mkldnn::sum::primitive_desc& sum_pd,
                                          const std::vector<size_t>& deps,
                                          size_t add_index)
{
    size_t input0_data_index = deps[0];
    build_memory(mkldnn_memories, sum_pd.dst_desc(), input0_data_index);
    size_t input1_data_index = deps[1];
    build_memory(mkldnn_memories, sum_pd.dst_desc(), input1_data_index);
    size_t result_index = deps[2];
    build_memory(mkldnn_memories, sum_pd.dst_desc(), result_index);

    mkldnn_scratchpad_mds[add_index] = new mkldnn::memory::desc(sum_pd.scratchpad_desc());

    // sum primitive
    mkldnn_primitives[add_index] = new mkldnn::sum(sum_pd);
}

void MKLDNNEmitter::build_batchnorm_forward(
    std::vector<mkldnn::memory*>& mkldnn_memories,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
    const mkldnn::batch_normalization_forward::desc& batchnorm_desc,
    const mkldnn::memory::desc& weights_desc,
    bool bn_training_flag,
    const std::vector<size_t>& deps,
    size_t batchnorm_index,
    const mkldnn::post_ops& pops)
{
    mkldnn::primitive_attr bn_attr;
    bn_attr.set_post_ops(pops);
    bn_attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto batchnorm_pd = mkldnn::batch_normalization_forward::primitive_desc(
        batchnorm_desc, bn_attr, executor::global_cpu_engine);
    mkldnn_scratchpad_mds[batchnorm_index] =
        new mkldnn::memory::desc(batchnorm_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(mkldnn_memories, batchnorm_pd.src_desc(), input_index);

    auto use_global_stats = batchnorm_desc.data.flags & 0x1U;
    if (bn_training_flag && !use_global_stats)
    {
        size_t weights_index = deps[1];
        build_memory(mkldnn_memories, weights_desc, weights_index);
        size_t result_index = deps[2];
        build_memory(mkldnn_memories, batchnorm_pd.dst_desc(), result_index);
        size_t mean_index = deps[3];
        build_memory(mkldnn_memories, batchnorm_pd.mean_desc(), mean_index);
        size_t variance_index = deps[4];
        build_memory(mkldnn_memories, batchnorm_pd.variance_desc(), variance_index);

        mkldnn_primitives[batchnorm_index] = new mkldnn::batch_normalization_forward(batchnorm_pd);
    }
    else
    {
        size_t weights_index = deps[3];
        build_memory(mkldnn_memories, weights_desc, weights_index);
        size_t result_index = deps[4];
        build_memory(mkldnn_memories, batchnorm_pd.dst_desc(), result_index);
        size_t mean_index = deps[1];
        build_memory(mkldnn_memories, batchnorm_pd.mean_desc(), mean_index);
        size_t variance_index = deps[2];
        build_memory(mkldnn_memories, batchnorm_pd.variance_desc(), variance_index);

        mkldnn_primitives[batchnorm_index] = new mkldnn::batch_normalization_forward(batchnorm_pd);
    }
}

void MKLDNNEmitter::build_batchnorm_backward(
    std::vector<mkldnn::memory*>& mkldnn_memories,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
    const mkldnn::batch_normalization_backward::desc& batchnorm_desc,
    const mkldnn::memory::desc& input_desc,
    const mkldnn::memory::desc& weights_desc,
    const mkldnn::memory::desc& dweights_desc,
    float epsilon,
    const std::vector<size_t>& deps,
    size_t batchnorm_index)
{
    auto batchnorm_fdesc =
        mkldnn::batch_normalization_forward::desc(mkldnn::prop_kind::forward_training,
                                                  input_desc,
                                                  epsilon,
                                                  mkldnn::BN_FLAG_CLASS::use_scale_shift);
    auto batchnorm_fpd = mkldnn::batch_normalization_forward::primitive_desc(
        batchnorm_fdesc, executor::global_cpu_engine);

    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto batchnorm_pd = mkldnn::batch_normalization_backward::primitive_desc(
        batchnorm_desc, attr, executor::global_cpu_engine, batchnorm_fpd);
    mkldnn_scratchpad_mds[batchnorm_index] =
        new mkldnn::memory::desc(batchnorm_pd.scratchpad_desc());

    size_t weights_index = deps[0];
    build_memory(mkldnn_memories, weights_desc, weights_index);
    size_t input_index = deps[1];
    build_memory(mkldnn_memories, batchnorm_pd.src_desc(), input_index);
    size_t mean_index = deps[2];
    build_memory(mkldnn_memories, batchnorm_pd.mean_desc(), mean_index);
    size_t variance_index = deps[3];
    build_memory(mkldnn_memories, batchnorm_pd.variance_desc(), variance_index);
    size_t delta_index = deps[4];
    build_memory(mkldnn_memories, batchnorm_pd.diff_src_desc(), delta_index);
    size_t dinput_index = deps[5];
    build_memory(mkldnn_memories, batchnorm_pd.dst_desc(), dinput_index);
    size_t dweights_index = deps[6];
    build_memory(mkldnn_memories, dweights_desc, dweights_index);

    mkldnn_primitives[batchnorm_index] = new mkldnn::batch_normalization_backward(batchnorm_pd);
}

void MKLDNNEmitter::build_rnn_forward(std::vector<mkldnn::memory*>& mkldnn_memories,
                                      std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                      std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                      std::vector<char*>& mkldnn_workspaces,
                                      const mkldnn::lstm_forward::desc& rnn_desc,
                                      std::vector<size_t>& deps,
                                      size_t rnn_index)
{
    size_t src_layer_index = deps[0];
    build_memory(mkldnn_memories, rnn_desc.data.src_layer_desc, src_layer_index);
    size_t src_iter_index = deps[1];
    build_memory(mkldnn_memories, rnn_desc.data.src_iter_desc, src_iter_index);
    size_t src_iter_c_index = deps[2];
    build_memory(mkldnn_memories, rnn_desc.data.src_iter_c_desc, src_iter_c_index);
    size_t weights_layer_index = deps[3];
    build_memory(mkldnn_memories, rnn_desc.data.weights_layer_desc, weights_layer_index);
    size_t weights_iter_index = deps[4];
    build_memory(mkldnn_memories, rnn_desc.data.weights_iter_desc, weights_iter_index);
    size_t bias_index = deps[5];
    build_memory(mkldnn_memories, rnn_desc.data.bias_desc, bias_index);
    size_t dst_layer_index = deps[6];
    build_memory(mkldnn_memories, rnn_desc.data.dst_layer_desc, dst_layer_index);
    size_t dst_iter_index = deps[7];
    build_memory(mkldnn_memories, rnn_desc.data.dst_iter_desc, dst_iter_index);
    size_t dst_iter_c_index = deps[8];
    build_memory(mkldnn_memories, rnn_desc.data.dst_iter_c_desc, dst_iter_c_index);

    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto rnn_layer_prim_desc =
        mkldnn::lstm_forward::primitive_desc(rnn_desc, attr, executor::global_cpu_engine);
    mkldnn_scratchpad_mds[rnn_index] =
        new mkldnn::memory::desc(rnn_layer_prim_desc.scratchpad_desc());

    size_t workspace_index = deps[9];
    build_memory(mkldnn_memories, rnn_layer_prim_desc.workspace_desc(), workspace_index);
    auto workspace = std::unique_ptr<MKLDNNWorkspace>(
        new MKLDNNWorkspace(rnn_layer_prim_desc.workspace_desc().get_size()));
    auto workspace_buf_index = insert_workspace(mkldnn_workspaces, workspace);
    deps[10] = workspace_buf_index;

    mkldnn_primitives[rnn_index] = new mkldnn::lstm_forward(rnn_layer_prim_desc);
}

void MKLDNNEmitter::build_concat(std::vector<mkldnn::memory*>& mkldnn_memories,
                                 std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                 std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                 const mkldnn::concat::primitive_desc& concat_pd,
                                 const std::vector<mkldnn::memory::desc>& inputs_data_desc,
                                 const std::vector<size_t>& deps,
                                 size_t concat_index)
{
    for (size_t i = 0; i < inputs_data_desc.size(); i++)
    {
        size_t inputs_data_index = deps[i];
        build_memory(mkldnn_memories, inputs_data_desc[i], inputs_data_index);
    }
    size_t result_index = deps[inputs_data_desc.size()];
    build_memory(mkldnn_memories, concat_pd.dst_desc(), result_index);

    mkldnn_scratchpad_mds[concat_index] = new mkldnn::memory::desc(concat_pd.scratchpad_desc());

    // concat primitive
    mkldnn_primitives[concat_index] = new mkldnn::concat(concat_pd);
}

void MKLDNNEmitter::build_slice(std::vector<mkldnn::memory*>& mkldnn_memories,
                                std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                mkldnn::memory::desc input_desc,
                                const mkldnn::memory::desc& result_desc,
                                const ngraph::Coordinate& lower_bounds,
                                const ngraph::Shape& result_shape,
                                const std::vector<size_t>& deps,
                                size_t slice_index)
{
    auto dims = mkldnn::memory::dims(result_shape.begin(), result_shape.end());
    auto offsets = mkldnn::memory::dims(lower_bounds.begin(), lower_bounds.end());
    auto input_sub_desc = input_desc.submemory_desc(dims, offsets);
    size_t input_index = deps[0];
    build_memory(mkldnn_memories, input_sub_desc, input_index);

    size_t result_index = deps[1];
    build_memory(mkldnn_memories, result_desc, result_index);

    // reorder primitive
    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto reorder_pd = mkldnn::reorder::primitive_desc(
        *mkldnn_memories[input_index], *mkldnn_memories[result_index], attr);
    mkldnn_scratchpad_mds[slice_index] = new mkldnn::memory::desc(reorder_pd.scratchpad_desc());

    mkldnn_primitives[slice_index] = new mkldnn::reorder(reorder_pd);
}

void MKLDNNEmitter::build_softmax_forward(std::vector<mkldnn::memory*>& mkldnn_memories,
                                          std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                          std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                          const mkldnn::softmax_forward::desc& softmax_desc,
                                          const std::vector<size_t>& deps,
                                          size_t softmax_index)
{
    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto softmax_pd =
        mkldnn::softmax_forward::primitive_desc(softmax_desc, attr, executor::global_cpu_engine);
    mkldnn_scratchpad_mds[softmax_index] = new mkldnn::memory::desc(softmax_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(mkldnn_memories, softmax_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(mkldnn_memories, softmax_pd.dst_desc(), result_index);

    mkldnn_primitives[softmax_index] = new mkldnn::softmax_forward(softmax_pd);
}

void MKLDNNEmitter::build_leaky_relu(std::vector<mkldnn::memory*>& mkldnn_memories,
                                     std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                     std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                     const mkldnn::eltwise_forward::desc& leaky_relu_desc,
                                     const std::vector<size_t>& deps,
                                     size_t leaky_relu_index)
{
    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto leaky_relu_pd =
        mkldnn::eltwise_forward::primitive_desc(leaky_relu_desc, attr, executor::global_cpu_engine);
    mkldnn_scratchpad_mds[leaky_relu_index] =
        new mkldnn::memory::desc(leaky_relu_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(mkldnn_memories, leaky_relu_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(mkldnn_memories, leaky_relu_pd.dst_desc(), result_index);

    mkldnn_primitives[leaky_relu_index] = new mkldnn::eltwise_forward(leaky_relu_pd);
}

void MKLDNNEmitter::build_bounded_relu(std::vector<mkldnn::memory*>& mkldnn_memories,
                                       std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                       std::vector<mkldnn::memory::desc*>& mkldnn_scratchpad_mds,
                                       const mkldnn::eltwise_forward::desc& bounded_relu_desc,
                                       const std::vector<size_t>& deps,
                                       size_t bounded_relu_index)
{
    mkldnn::primitive_attr attr;
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto bounded_relu_pd = mkldnn::eltwise_forward::primitive_desc(
        bounded_relu_desc, attr, executor::global_cpu_engine);
    mkldnn_scratchpad_mds[bounded_relu_index] =
        new mkldnn::memory::desc(bounded_relu_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(mkldnn_memories, bounded_relu_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(mkldnn_memories, bounded_relu_pd.dst_desc(), result_index);

    mkldnn_primitives[bounded_relu_index] = new mkldnn::eltwise_forward(bounded_relu_pd);
}

void MKLDNNEmitter::query_scratchpad_sum(const mkldnn::sum::primitive_desc pd)
{
    mkldnn::memory::desc scratchpad_md = pd.scratchpad_desc();
    auto size = scratchpad_md.get_size();
    m_max_scratchpad_size = size > m_max_scratchpad_size ? size : m_max_scratchpad_size;
}

void MKLDNNEmitter::query_scratchpad_concat(const mkldnn::concat::primitive_desc pd)
{
    mkldnn::memory::desc scratchpad_md = pd.scratchpad_desc();
    auto size = scratchpad_md.get_size();
    m_max_scratchpad_size = size > m_max_scratchpad_size ? size : m_max_scratchpad_size;
}

void MKLDNNEmitter::query_scratchpad_pooling_forward(const mkldnn::pooling_forward::desc& desc)
{
    ATTR_S
    auto pd = mkldnn::pooling_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_avg_pooling_backward(
    const mkldnn::pooling_forward::desc& fwd_desc, const mkldnn::pooling_backward::desc& bwd_desc)
{
    ATTR_S
    auto fwd_pd = mkldnn::pooling_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);
    auto pd = mkldnn::pooling_backward::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_max_pooling_backward(
    const mkldnn::pooling_forward::desc& fwd_desc, const mkldnn::pooling_backward::desc& bwd_desc)
{
    ATTR_S
    auto fwd_pd =
        mkldnn::pooling_forward::primitive_desc(fwd_desc, attr, executor::global_cpu_engine);
    auto pd = mkldnn::pooling_backward::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
    mkldnn::memory::desc fwd_scratchpad_md = fwd_pd.scratchpad_desc();
    size = fwd_scratchpad_md.get_size();
    m_max_scratchpad_size = size > m_max_scratchpad_size ? size : m_max_scratchpad_size;
}

void MKLDNNEmitter::query_scratchpad_max_pooling_with_indices_backward(
    const mkldnn::pooling_forward::desc& fwd_desc, const mkldnn::pooling_backward::desc& bwd_desc)
{
    ATTR_S
    auto fwd_pd =
        mkldnn::pooling_forward::primitive_desc(fwd_desc, attr, executor::global_cpu_engine);
    auto pd = mkldnn::pooling_backward::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_batchnorm_forward(
    const mkldnn::batch_normalization_forward::desc& desc, const mkldnn::post_ops& pops)
{
    ATTR_S
    attr.set_post_ops(pops);
    auto pd = mkldnn::batch_normalization_forward::primitive_desc(
        desc, attr, executor::global_cpu_engine);
    GET_SIZE
}
void MKLDNNEmitter::query_scratchpad_batchnorm_backward(
    const mkldnn::batch_normalization_backward::desc& desc,
    const mkldnn::memory::desc& input_desc,
    float epsilon)
{
    ATTR_S
    auto fwd_desc =
        mkldnn::batch_normalization_forward::desc(mkldnn::prop_kind::forward_training,
                                                  input_desc,
                                                  epsilon,
                                                  mkldnn::BN_FLAG_CLASS::use_scale_shift);
    auto fwd_pd =
        mkldnn::batch_normalization_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);
    auto pd = mkldnn::batch_normalization_backward::primitive_desc(
        desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_convolution_forward(
    const mkldnn::convolution_forward::desc& desc, mkldnn::primitive_attr& attr)
{
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto pd = mkldnn::convolution_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_convolution_backward_data(
    const mkldnn::convolution_forward::desc& fwd_desc,
    const mkldnn::convolution_backward_data::desc& bwd_desc)
{
    ATTR_S
    auto fwd_pd =
        mkldnn::convolution_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);
    auto pd = mkldnn::convolution_backward_data::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_convolution_backward_weights(
    const mkldnn::convolution_forward::desc& fwd_desc,
    const mkldnn::convolution_backward_weights::desc& bwd_desc)
{
    ATTR_S
    auto fwd_pd =
        mkldnn::convolution_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);
    auto pd = mkldnn::convolution_backward_weights::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_deconvolution_forward(
    const mkldnn::deconvolution_forward::desc& desc)
{
    ATTR_S
    auto pd =
        mkldnn::deconvolution_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_eltwise_forward(const mkldnn::eltwise_forward::desc& desc)
{
    ATTR_S
    auto pd = mkldnn::eltwise_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_eltwise_backward(
    const mkldnn::eltwise_forward::desc& fwd_desc, const mkldnn::eltwise_backward::desc& bwd_desc)
{
    ATTR_S
    auto fwd_pd = mkldnn::eltwise_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);
    auto pd = mkldnn::eltwise_backward::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
}
void MKLDNNEmitter::query_scratchpad_quantize(const mkldnn::memory::desc& input_desc,
                                              const mkldnn::memory::desc& output_desc)
{
}
void MKLDNNEmitter::query_scratchpad_dequantize(const mkldnn::memory::desc& input_desc,
                                                const mkldnn::memory::desc& output_desc)
{
}

void MKLDNNEmitter::query_scratchpad_ip_forward(const mkldnn::inner_product_forward::desc& desc,
                                                mkldnn::primitive_attr& attr)
{
    attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);
    auto pd =
        mkldnn::inner_product_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_reorder(const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& result_desc)
{
    ATTR_S
    auto pd = mkldnn::reorder::primitive_desc(
        executor::global_cpu_engine, input_desc, executor::global_cpu_engine, result_desc, attr);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_rnn_forward(const mkldnn::lstm_forward::desc& desc)
{
    ATTR_S
    auto pd = mkldnn::lstm_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_lrn_forward(const mkldnn::lrn_forward::desc& desc)
{
    ATTR_S
    auto pd = mkldnn::lrn_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_slice(mkldnn::memory::desc& input_desc,
                                           const mkldnn::memory::desc& output_desc,
                                           const ngraph::Coordinate& lower_bounds,
                                           const ngraph::Shape& result_shape)
{
    ATTR_S
    auto dims = mkldnn::memory::dims(result_shape.begin(), result_shape.end());
    auto offsets = mkldnn::memory::dims(lower_bounds.begin(), lower_bounds.end());
    auto input_sub_desc = input_desc.submemory_desc(dims, offsets);
    auto input = mkldnn::memory(input_sub_desc, executor::global_cpu_engine);
    auto output = mkldnn::memory(output_desc, executor::global_cpu_engine);
    auto pd = mkldnn::reorder::primitive_desc(input, output, attr);
    GET_SIZE
}

void MKLDNNEmitter::query_scratchpad_softmax_forward(const mkldnn::softmax_forward::desc& desc)
{
    ATTR_S
    auto pd = mkldnn::softmax_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

#else
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

void MKLDNNEmitter::build_quantize_reorder(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

    mkldnn_primitives[deconv_index] =
        new mkldnn::deconvolution_forward({deconv_desc, executor::global_cpu_engine},
                                          *mkldnn_primitives[delta_index],
                                          *mkldnn_primitives[weights_index],
                                          *mkldnn_primitives[bias_index],
                                          *mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_convolution_backward_weights_bias(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_pooling_forward(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_pooling_backward(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_max_pooling_backward(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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
        *mkldnn_primitives[diff_src_index], // HACK - Uses diff_src buffer. Safe since diff_src >
        *mkldnn_primitives[ws_index]);      //        fprop_dst

    mkldnn_primitives[bwd_pool_index] =
        new mkldnn::pooling_backward({bwd_pool_desc, executor::global_cpu_engine, fwd_pd},
                                     *mkldnn_primitives[diff_dst_index],
                                     *mkldnn_primitives[ws_index],
                                     *mkldnn_primitives[diff_src_index]);
}

void MKLDNNEmitter::build_max_pooling_with_indices_forward(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_reorder(std::vector<mkldnn::memory*>& /* mkldnn_memories */,
                                  std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                  std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_lrn_forward(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_relu_forward(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_relu_backward(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

    // create forward relu primitive descriptor
    auto relu_pd = mkldnn::eltwise_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    // create backward relu primitive_descriptor
    auto relu_bwd_pd =
        mkldnn::eltwise_backward::primitive_desc(bwd_desc, executor::global_cpu_engine, relu_pd);

    mkldnn_primitives[relu_index] = new mkldnn::eltwise_backward(relu_bwd_pd,
                                                                 *mkldnn_primitives[input_index],
                                                                 *mkldnn_primitives[delta_index],
                                                                 *mkldnn_primitives[result_index]);
}

void MKLDNNEmitter::build_sigmoid_forward(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_sigmoid_backward(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_elementwise_add(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_batchnorm_backward(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
    const mkldnn::batch_normalization_backward::desc& batchnorm_desc,
    const mkldnn::memory::desc& /* input_desc */,
    const mkldnn::memory::desc& weights_desc,
    const mkldnn::memory::desc& dweights_desc,
    float /* epsilon */,
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

void MKLDNNEmitter::build_rnn_forward(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_concat(std::vector<mkldnn::memory*>& /* mkldnn_memories */,
                                 std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                 std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_slice(std::vector<mkldnn::memory*>& /* mkldnn_memories */,
                                std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
                                mkldnn::memory::desc input_desc,
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

void MKLDNNEmitter::build_softmax_forward(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_leaky_relu(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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

void MKLDNNEmitter::build_bounded_relu(
    std::vector<mkldnn::memory*>& /* mkldnn_memories */,
    std::vector<mkldnn::primitive*>& mkldnn_primitives,
    std::vector<mkldnn::memory::desc*>& /* mkldnn_scratchpad_mds */,
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
#endif
