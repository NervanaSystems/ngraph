//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "dnnl_emitter.hpp"

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
#include "ngraph/runtime/cpu/cpu_tensor_wrapper.hpp"
#include "ngraph/runtime/cpu/dnnl_invoke.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/deconv.hpp"
#include "ngraph/runtime/cpu/op/gelu_backprop.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;
using namespace ngraph::op;
using namespace ngraph::runtime::cpu;

DNNLEmitter::~DNNLEmitter()
{
    for (auto p : m_dnnl_primitives)
        delete p;
    for (auto s : m_dnnl_scratchpad_mds)
        delete s;
#ifndef _WIN32
    // To avoid memory leak in dnnl, release any buffers that are not free'd yet.
    // https://software.intel.com/en-us/mkl-linux-developer-guide-avoiding-memory-leaks-in-intel-mkl
    // mkl_free_buffers() is not exposed at this point, hence using mkl_serv_free_buffers()
    dnnl_utils::mkl_serv_free_buffers();
#endif
}

std::vector<dnnl::primitive*>& DNNLEmitter::get_dnnl_primitives()
{
    return m_dnnl_primitives;
}

const std::vector<dnnl::memory*>& DNNLEmitter::get_dnnl_memories() const
{
    return m_dnnl_memories;
}

const std::vector<char*>& DNNLEmitter::get_dnnl_workspaces()
{
    return m_workspace_bufs;
}

const std::vector<dnnl::memory::desc*>& DNNLEmitter::get_dnnl_scratchpad_mds() const
{
    return m_dnnl_scratchpad_mds;
}

size_t DNNLEmitter::insert_primitive(dnnl::primitive* primitive)
{
    m_dnnl_primitives.emplace_back(primitive);
    return (m_dnnl_primitives.size() - 1);
}

size_t DNNLEmitter::insert_memory(dnnl::memory* memory)
{
    m_dnnl_memories.emplace_back(memory);
    return (m_dnnl_memories.size() - 1);
}

size_t DNNLEmitter::insert_workspace(std::unique_ptr<DNNLWorkspace>& workspace)
{
    m_workspace_bufs.push_back(workspace.get()->buf);
    m_workspaces.push_back(std::move(workspace));
    return (m_workspaces.size() - 1);
}

size_t DNNLEmitter::reserve_workspace()
{
    m_workspaces_size++;
    return m_workspaces_size - 1;
}

size_t DNNLEmitter::insert_scratchpad_md(dnnl::memory::desc* md)
{
    m_dnnl_scratchpad_mds.emplace_back(md);
    return (m_dnnl_scratchpad_mds.size() - 1);
}

void DNNLEmitter::reserve_descriptor_space(size_t count)
{
    m_dnnl_descriptors_size += count;
}

size_t DNNLEmitter::get_dnnl_descriptors_size()
{
    return m_dnnl_descriptors_size;
}

size_t DNNLEmitter::insert_workspace(std::vector<char*>& dnnl_workspaces,
                                     std::unique_ptr<DNNLWorkspace>& workspace)
{
    dnnl_workspaces.push_back(workspace.get()->buf);
    m_workspaces.push_back(std::move(workspace));
    return (dnnl_workspaces.size() - 1);
}

const std::vector<size_t>& DNNLEmitter::get_primitive_deps(size_t index) const
{
    return m_primitive_deps.at(index);
}

std::vector<size_t>& DNNLEmitter::get_primitive_deps(size_t index)
{
    return m_primitive_deps.at(index);
}

size_t DNNLEmitter::get_max_scratchpad_size() const
{
    return m_max_scratchpad_size;
}

dnnl::memory::desc DNNLEmitter::build_blocked_memory_descriptor(const dnnl::memory::dims& dim,
                                                                const dnnl::memory::dims& strides,
                                                                dnnl::memory::data_type dtype) const
{
    return dnnl::memory::desc(dim, dtype, strides);
}

size_t DNNLEmitter::build_quantize_reorder(const dnnl::memory::desc& input_desc,
                                           const dnnl::memory::desc& result_desc,
                                           const std::vector<float>& scales)
{
    dnnl::primitive_attr attr;
    attr.set_output_scales(0, scales);
    size_t input_index, result_index, primitive_index;

    input_index = build_memory(input_desc);
    result_index = build_memory(result_desc);
    auto reorder_prim_desc = dnnl::reorder::primitive_desc(
        executor::global_cpu_engine, input_desc, executor::global_cpu_engine, result_desc, attr);
    primitive_index = insert_primitive(new dnnl::reorder(reorder_prim_desc));

    NGRAPH_CHECK(m_primitive_deps.find(primitive_index) == m_primitive_deps.end(),
                 "Dependencies already created for node");

    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
}

size_t DNNLEmitter::build_dequantization(const ngraph::Node* node,
                                         const dnnl::memory::desc& input_desc,
                                         const dnnl::memory::desc& result_desc)
{
    auto dequantize = static_cast<const ngraph::op::v0::Dequantize*>(node);
    auto scale_const_op = as_type_ptr<ngraph::op::v0::Constant>(dequantize->get_argument(1));
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

size_t DNNLEmitter::build_reorder(const dnnl::memory::desc& input_desc,
                                  const dnnl::memory::desc& result_desc)
{
    size_t input_index, result_index, primitive_index;

    input_index = build_memory(input_desc);
    result_index = build_memory(result_desc);
    primitive_index = insert_primitive(
        new dnnl::reorder(*m_dnnl_memories[input_index], *m_dnnl_memories[result_index]));

    NGRAPH_CHECK(m_primitive_deps.find(primitive_index) == m_primitive_deps.end(),
                 "Dependencies already created for node");

    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
}

dnnl::lrn_forward::desc DNNLEmitter::get_lrn_forward_desc(const ngraph::Node* node)
{
    const ngraph::op::v0::LRN* lrn = static_cast<const ngraph::op::v0::LRN*>(node);

    auto alpha = static_cast<float>(lrn->get_alpha());
    auto beta = static_cast<float>(lrn->get_beta());
    auto bias = static_cast<float>(lrn->get_bias());
    auto nsize = static_cast<int>(lrn->get_nsize());

    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);

    return dnnl::lrn_forward::desc(dnnl::prop_kind::forward_scoring,
                                   dnnl::algorithm::lrn_across_channels,
                                   input_desc,
                                   nsize,
                                   alpha,
                                   beta,
                                   bias);
}

dnnl::eltwise_forward::desc DNNLEmitter::get_relu_forward_desc(const ngraph::Node* node)
{
    const float negative_slope = 0.0f;

    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);

    return dnnl::eltwise_forward::desc(
        dnnl::prop_kind::forward, dnnl::algorithm::eltwise_relu, input_desc, negative_slope);
}

dnnl::eltwise_backward::desc DNNLEmitter::get_relu_backward_desc(const ngraph::Node* node)
{
    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

    const float negative_slope = 0.0f;
    return dnnl::eltwise_backward::desc(
        dnnl::algorithm::eltwise_relu, result_desc, input_desc, negative_slope);
}

dnnl::eltwise_forward::desc DNNLEmitter::get_sigmoid_forward_desc(const ngraph::Node* node,
                                                                  bool backward_op)
{
    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);

    if (backward_op)
    {
        return dnnl::eltwise_forward::desc(
            dnnl::prop_kind::forward, dnnl::algorithm::eltwise_logistic, input_desc, 0, 0);
    }
    else
    {
        return dnnl::eltwise_forward::desc(
            dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_logistic, input_desc, 0, 0);
    }
}

dnnl::eltwise_backward::desc DNNLEmitter::get_sigmoid_backward_desc(const ngraph::Node* node)
{
    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
    auto delta_desc = dnnl_utils::get_input_dnnl_md(node, 1);

    return dnnl::eltwise_backward::desc(
        dnnl::algorithm::eltwise_logistic, delta_desc, input_desc, 0, 0);
}

dnnl::batch_normalization_backward::desc
    DNNLEmitter::get_batchnorm_backward_desc(const ngraph::Node* node)
{
    const ngraph::op::v0::BatchNormTrainingBackprop* batchnorm =
        static_cast<const ngraph::op::v0::BatchNormTrainingBackprop*>(node);
    auto eps = batchnorm->get_eps_value();

    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 2);
    auto delta_desc = dnnl_utils::get_input_dnnl_md(node, 5);

    return dnnl::batch_normalization_backward::desc(dnnl::prop_kind::backward,
                                                    delta_desc,
                                                    input_desc,
                                                    eps,
                                                    dnnl::BN_FLAG_CLASS::use_scale_shift);
}

dnnl::softmax_forward::desc DNNLEmitter::get_softmax_forward_desc(const ngraph::Node* node)
{
    auto softmax = static_cast<const ngraph::op::v0::Softmax*>(node);

    auto axes = softmax->get_axes();
    if (axes.size() != 1)
    {
        throw ngraph_error("DNNL supports softmax only across single axis");
    }
    int softmax_axis = static_cast<int>(*(axes.begin()));

    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);

    return dnnl::softmax_forward::desc(dnnl::prop_kind::forward_scoring, input_desc, softmax_axis);
}

dnnl::eltwise_forward::desc DNNLEmitter::get_leaky_relu_desc(const ngraph::Node* node)
{
    auto alpha = static_cast<const ngraph::op::CPULeakyRelu*>(node)->get_alpha();

    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);

    return dnnl::eltwise_forward::desc(
        dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_relu, input_desc, alpha, 0.0f);
}

dnnl::eltwise_forward::desc DNNLEmitter::get_bounded_relu_desc(const ngraph::Node* node)
{
    auto alpha = static_cast<const ngraph::op::BoundedRelu*>(node)->get_alpha();

    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);

    return dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training,
                                       dnnl::algorithm::eltwise_bounded_relu,
                                       input_desc,
                                       alpha,
                                       0.0f);
}

dnnl::eltwise_forward::desc DNNLEmitter::get_gelu_forward_desc(const ngraph::Node* node)
{
    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);

    return dnnl::eltwise_forward::desc(
        dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_gelu, input_desc, 1.0f, 0.0f);
}

dnnl::eltwise_backward::desc DNNLEmitter::get_gelu_backward_desc(const ngraph::Node* node)
{
    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

    const float negative_slope = 0.0f;
    return dnnl::eltwise_backward::desc(
        dnnl::algorithm::eltwise_gelu, result_desc, input_desc, negative_slope);
}

size_t DNNLEmitter::convolution_forward_init(bool with_bias)
{
    size_t size = m_dnnl_primitives.size();
    if (with_bias)
    {
        // Inputs, Weights, Bias, Results, Conv
        size_t mem_size = m_dnnl_memories.size();
        m_dnnl_primitives.resize(size + 1, nullptr);
        m_dnnl_scratchpad_mds.resize(size + 1, nullptr);
        m_dnnl_memories.resize(mem_size + 4, nullptr);
        m_primitive_deps[m_dnnl_primitives.size() - 1] = {
            mem_size, mem_size + 1, mem_size + 2, mem_size + 3};
    }
    else
    {
        // Inputs, Weights, Results, Conv
        size_t mem_size = m_dnnl_memories.size();
        m_dnnl_primitives.resize(size + 1, nullptr);
        m_dnnl_scratchpad_mds.resize(size + 1, nullptr);
        m_dnnl_memories.resize(mem_size + 3, nullptr);
        m_primitive_deps[m_dnnl_primitives.size() - 1] = {mem_size, mem_size + 1, mem_size + 2};
    }
    return m_dnnl_primitives.size() - 1;
}

size_t DNNLEmitter::inner_product_forward_init(bool with_bias)
{
    size_t size = m_dnnl_primitives.size();
    if (with_bias)
    {
        // Inputs, Weights, Bias, Results, inner_product
        size_t mem_size = m_dnnl_memories.size();
        m_dnnl_primitives.resize(size + 1, nullptr);
        m_dnnl_scratchpad_mds.resize(size + 1, nullptr);
        m_dnnl_memories.resize(mem_size + 4, nullptr);
        m_primitive_deps[m_dnnl_primitives.size() - 1] = {
            mem_size, mem_size + 1, mem_size + 2, mem_size + 3};
    }
    else
    {
        // Inputs, Weights, Results, inner_product
        size_t mem_size = m_dnnl_memories.size();
        m_dnnl_primitives.resize(size + 1, nullptr);
        m_dnnl_scratchpad_mds.resize(size + 1, nullptr);
        m_dnnl_memories.resize(mem_size + 3, nullptr);
        m_primitive_deps[m_dnnl_primitives.size() - 1] = {mem_size, mem_size + 1, mem_size + 2};
    }
    return m_dnnl_primitives.size() - 1;
}

size_t DNNLEmitter::reserve_primitive_space(size_t count, bool fwd_bwd, bool new_workspace)
{
    size_t size = m_dnnl_primitives.size();
    size_t mem_size = m_dnnl_memories.size();
    if (fwd_bwd)
    {
        m_dnnl_primitives.resize(size + 2, nullptr);
        m_dnnl_memories.resize(mem_size + count - 2, nullptr);
        m_dnnl_scratchpad_mds.resize(size + 2, nullptr);
        for (auto i = 0; i < count - 2; i++)
        {
            m_primitive_deps[m_dnnl_primitives.size() - 2].push_back(mem_size + i);
            m_primitive_deps[m_dnnl_primitives.size() - 1].push_back(mem_size + i);
        }
    }
    else
    {
        m_dnnl_primitives.resize(size + 1, nullptr);
        m_dnnl_memories.resize(mem_size + count - 1, nullptr);
        m_dnnl_scratchpad_mds.resize(size + 1, nullptr);
        for (auto i = 0; i < count - 1; i++)
        {
            m_primitive_deps[m_dnnl_primitives.size() - 1].push_back(mem_size + i);
        }
    }

    if (new_workspace)
    {
        m_primitive_deps[m_dnnl_primitives.size() - 1].push_back(0);
        if (fwd_bwd)
        {
            m_primitive_deps[m_dnnl_primitives.size() - 2].push_back(0);
        }
    }
    return m_dnnl_primitives.size() - 1;
}

size_t DNNLEmitter::build_quantized_inner_product_forward(const dnnl::memory::desc& input_data_desc,
                                                          const dnnl::memory::desc& weights_desc,
                                                          const dnnl::memory::desc& bias_desc,
                                                          const dnnl::memory::desc& result_desc,
                                                          const float scale,
                                                          const dnnl::post_ops& pops)
{
    std::vector<float> output_scale;
    output_scale.push_back(scale);
    // dnnl inner_product attr
    dnnl::primitive_attr ip_attr;
    ip_attr.set_post_ops(pops);
    /* Specify the scales array and corresponding mask */
    ip_attr.set_output_scales(0, output_scale);

    size_t ip_index;
    size_t input_data_index = build_memory(input_data_desc);
    size_t weights_index = build_memory(weights_desc);
    size_t bias_index = build_memory(bias_desc);
    size_t result_index = build_memory(result_desc);

    // dnnl inner_product
    ip_index =
        insert_primitive(new dnnl::inner_product_forward({{
                                                              dnnl::prop_kind::forward_scoring,
                                                              input_data_desc,
                                                              weights_desc,
                                                              bias_desc,
                                                              result_desc,
                                                          },
                                                          ip_attr,
                                                          executor::global_cpu_engine}));
    m_primitive_deps[ip_index] = {input_data_index, weights_index, bias_index, result_index};
    return ip_index;
}

size_t DNNLEmitter::build_quantized_inner_product_forward(const dnnl::memory::desc& input_data_desc,
                                                          const dnnl::memory::desc& weights_desc,
                                                          const dnnl::memory::desc& result_desc,
                                                          const float scale,
                                                          const dnnl::post_ops& pops)
{
    std::vector<float> output_scale;
    output_scale.push_back(scale);
    // dnnl inner_product attr
    dnnl::primitive_attr ip_attr;
    ip_attr.set_post_ops(pops);
    /* Specify the scales array and corresponding mask */
    ip_attr.set_output_scales(0, output_scale);

    size_t ip_index;
    size_t input_data_index = build_memory(input_data_desc);
    size_t weights_index = build_memory(weights_desc);
    size_t result_index = build_memory(result_desc);

    // dnnl inner_product
    ip_index =
        insert_primitive(new dnnl::inner_product_forward({{
                                                              dnnl::prop_kind::forward_scoring,
                                                              input_data_desc,
                                                              weights_desc,
                                                              result_desc,
                                                          },
                                                          ip_attr,
                                                          executor::global_cpu_engine}));
    m_primitive_deps[ip_index] = {input_data_index, weights_index, result_index};
    return ip_index;
}

dnnl::memory::desc DNNLEmitter::build_memory_descriptor(const TensorWrapper& tvw,
                                                        dnnl::memory::format_tag fmt_tag) const
{
    return dnnl::memory::desc(dnnl::memory::dims(tvw.get_shape().begin(), tvw.get_shape().end()),
                              dnnl_utils::get_dnnl_data_type(tvw.get_element_type()),
                              fmt_tag);
}

dnnl::memory::desc DNNLEmitter::build_memory_descriptor(const ngraph::Shape& shape,
                                                        const ngraph::element::Type& et,
                                                        dnnl::memory::format_tag fmt) const
{
    return dnnl::memory::desc(
        dnnl::memory::dims(shape.begin(), shape.end()), dnnl_utils::get_dnnl_data_type(et), fmt);
}

size_t DNNLEmitter::build_memory(const dnnl::memory::desc& desc)
{
    size_t index = insert_memory(new dnnl::memory(desc, executor::global_cpu_engine, nullptr));
    return index;
}

void DNNLEmitter::build_memory(const dnnl::memory::desc& desc, size_t index)
{
    m_dnnl_memories[index] = new dnnl::memory(desc, executor::global_cpu_engine, nullptr);
}

void DNNLEmitter::build_memory(std::vector<dnnl::memory*>& dnnl_memories,
                               const dnnl::memory::desc& desc,
                               size_t index)
{
    dnnl_memories[index] = new dnnl::memory(desc, executor::global_cpu_engine, nullptr);
}

dnnl::sum::primitive_desc DNNLEmitter::get_elementwise_add_desc(const ngraph::Node* node)
{
    std::vector<float> scale_vector(2, 1);
    auto input0_data_desc = dnnl_utils::get_input_dnnl_md(node, 0);
    auto input1_data_desc = dnnl_utils::get_input_dnnl_md(node, 1);
    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);
    std::vector<dnnl::memory::desc> inputs_desc{input0_data_desc, input1_data_desc};

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // elementwise sum primtive descriptor
    dnnl::sum::primitive_desc sum_pd = dnnl::sum::primitive_desc(
        result_desc, scale_vector, inputs_desc, executor::global_cpu_engine, attr);

    return sum_pd;
}

void DNNLEmitter::build_quantize_reorder(std::vector<dnnl::memory*>& dnnl_memories,
                                         std::vector<dnnl::primitive*>& dnnl_primitives,
                                         std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                         const dnnl::memory::desc& input_desc,
                                         const dnnl::memory::desc& result_desc,
                                         const std::vector<float>& scales,
                                         const std::vector<size_t>& deps,
                                         size_t quantize_index,
                                         const int mask)
{
    size_t input_index = deps[0];
    build_memory(dnnl_memories, input_desc, input_index);
    size_t result_index = deps[1];
    build_memory(dnnl_memories, result_desc, result_index);

    dnnl::primitive_attr attr;
    attr.set_output_scales(mask, scales);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto reorder_prim_desc = dnnl::reorder::primitive_desc(
        executor::global_cpu_engine, input_desc, executor::global_cpu_engine, result_desc, attr);

    dnnl_scratchpad_mds[quantize_index] =
        new dnnl::memory::desc(reorder_prim_desc.scratchpad_desc());
    dnnl_primitives[quantize_index] = new dnnl::reorder(reorder_prim_desc);
}

void DNNLEmitter::build_deconvolutionbias_forward(
    std::vector<dnnl::memory*>& dnnl_memories,
    std::vector<dnnl::primitive*>& dnnl_primitives,
    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
    const dnnl::deconvolution_forward::desc& deconv_desc,
    const std::vector<size_t>& deps,
    size_t deconv_index,
    const dnnl::memory::desc& weights_desc)
{
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto deconv_pd =
        dnnl::deconvolution_forward::primitive_desc(deconv_desc, attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[deconv_index] = new dnnl::memory::desc(deconv_pd.scratchpad_desc());

    size_t weights_index = deps[0];
    build_memory(dnnl_memories, weights_desc, weights_index);
    size_t delta_index = deps[1];
    build_memory(dnnl_memories, deconv_pd.src_desc(), delta_index);
    size_t bias_index = deps[2];
    build_memory(dnnl_memories, deconv_pd.bias_desc(), bias_index);
    size_t result_index = deps[3];
    build_memory(dnnl_memories, deconv_pd.dst_desc(), result_index);

    dnnl_primitives[deconv_index] = new dnnl::deconvolution_forward(deconv_pd);
}

void DNNLEmitter::build_convolution_backward_weights_bias(
    std::vector<dnnl::memory*>& dnnl_memories,
    std::vector<dnnl::primitive*>& dnnl_primitives,
    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
    const dnnl::convolution_backward_weights::desc& bwd_desc,
    const dnnl::convolution_forward::desc& fwd_desc,
    const std::vector<size_t>& deps,
    size_t conv_index)
{
    auto conv_fwd_pd =
        dnnl::convolution_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto conv_bwd_pd = dnnl::convolution_backward_weights::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, conv_fwd_pd);
    dnnl_scratchpad_mds[conv_index] = new dnnl::memory::desc(conv_bwd_pd.scratchpad_desc());

    size_t src_index = deps[0];
    build_memory(dnnl_memories, conv_bwd_pd.src_desc(), src_index);
    size_t diff_dst_index = deps[1];
    build_memory(dnnl_memories, conv_bwd_pd.diff_dst_desc(), diff_dst_index);
    size_t diff_weights_index = deps[2];
    build_memory(dnnl_memories, conv_bwd_pd.diff_weights_desc(), diff_weights_index);
    size_t diff_bias_index = deps[3];
    build_memory(dnnl_memories, conv_bwd_pd.diff_bias_desc(), diff_bias_index);

    dnnl_primitives[conv_index] = new dnnl::convolution_backward_weights(conv_bwd_pd);
}

void DNNLEmitter::build_convolution_backward_weights(
    std::vector<dnnl::memory*>& dnnl_memories,
    std::vector<dnnl::primitive*>& dnnl_primitives,
    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
    const dnnl::convolution_backward_weights::desc& bwd_desc,
    const dnnl::convolution_forward::desc& fwd_desc,
    const std::vector<size_t>& deps,
    size_t conv_index)
{
    // Forward primitive descriptor corresponding to this backward weights descriptor
    auto conv_fwd_pd =
        dnnl::convolution_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto conv_bwd_pd = dnnl::convolution_backward_weights::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, conv_fwd_pd);
    dnnl_scratchpad_mds[conv_index] = new dnnl::memory::desc(conv_bwd_pd.scratchpad_desc());

    size_t src_index = deps[0];
    build_memory(dnnl_memories, conv_bwd_pd.src_desc(), src_index);
    size_t diff_dst_index = deps[1];
    build_memory(dnnl_memories, conv_bwd_pd.diff_dst_desc(), diff_dst_index);
    size_t diff_weights_index = deps[2];
    build_memory(dnnl_memories, conv_bwd_pd.diff_weights_desc(), diff_weights_index);

    dnnl_primitives[conv_index] = new dnnl::convolution_backward_weights(conv_bwd_pd);
}

void DNNLEmitter::build_convolution_backward_data(
    std::vector<dnnl::memory*>& dnnl_memories,
    std::vector<dnnl::primitive*>& dnnl_primitives,
    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
    const dnnl::convolution_backward_data::desc& bwd_desc,
    const dnnl::convolution_forward::desc& fwd_desc,
    const std::vector<size_t>& deps,
    size_t conv_index)
{
    // Forward primitive descriptor corresponding to this backward weights descriptor
    auto conv_fwd_pd =
        dnnl::convolution_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto conv_bwd_pd = dnnl::convolution_backward_data::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, conv_fwd_pd);
    dnnl_scratchpad_mds[conv_index] = new dnnl::memory::desc(conv_bwd_pd.scratchpad_desc());

    size_t weights_index = deps[0];
    build_memory(dnnl_memories, conv_bwd_pd.weights_desc(), weights_index);
    size_t diff_dst_index = deps[1];
    build_memory(dnnl_memories, conv_bwd_pd.diff_dst_desc(), diff_dst_index);
    size_t diff_src_index = deps[2];
    build_memory(dnnl_memories, conv_bwd_pd.diff_src_desc(), diff_src_index);

    dnnl_primitives[conv_index] = new dnnl::convolution_backward_data(conv_bwd_pd);
}

void DNNLEmitter::build_pooling_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                        std::vector<dnnl::primitive*>& dnnl_primitives,
                                        std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                        const dnnl::pooling_forward::desc& pool_desc,
                                        const std::vector<size_t>& deps,
                                        size_t pool_index)
{
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pool_pd =
        dnnl::pooling_forward::primitive_desc(pool_desc, attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[pool_index] = new dnnl::memory::desc(pool_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, pool_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(dnnl_memories, pool_pd.dst_desc(), result_index);

    dnnl_primitives[pool_index] = new dnnl::pooling_forward(pool_pd);
}

void DNNLEmitter::build_pooling_backward(std::vector<dnnl::memory*>& dnnl_memories,
                                         std::vector<dnnl::primitive*>& dnnl_primitives,
                                         std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                         const dnnl::pooling_backward::desc& pool_desc,
                                         const dnnl::pooling_forward::desc& pool_fwd_desc,
                                         const std::vector<size_t>& deps,
                                         size_t pool_index)
{
    auto pool_fwd_pd =
        dnnl::pooling_forward::primitive_desc(pool_fwd_desc, executor::global_cpu_engine);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pool_bwd_pd = dnnl::pooling_backward::primitive_desc(
        pool_desc, attr, executor::global_cpu_engine, pool_fwd_pd);
    dnnl_scratchpad_mds[pool_index] = new dnnl::memory::desc(pool_bwd_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, pool_bwd_pd.diff_dst_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(dnnl_memories, pool_bwd_pd.diff_src_desc(), result_index);

    dnnl_primitives[pool_index] = new dnnl::pooling_backward(pool_bwd_pd);
}

void DNNLEmitter::build_max_pooling_backward(std::vector<dnnl::memory*>& dnnl_memories,
                                             std::vector<dnnl::primitive*>& dnnl_primitives,
                                             std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                             std::vector<char*>& dnnl_workspaces,
                                             const dnnl::pooling_backward::desc& bwd_pool_desc,
                                             const dnnl::pooling_forward::desc& fwd_pool_desc,
                                             const dnnl::memory::desc& fprop_src_desc,
                                             std::vector<size_t>& fdeps,
                                             std::vector<size_t>& bdeps,
                                             size_t fwd_pool_index,
                                             size_t bwd_pool_index)
{
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pool_fwd_pd =
        dnnl::pooling_forward::primitive_desc(fwd_pool_desc, attr, executor::global_cpu_engine);
    auto pool_bwd_pd = dnnl::pooling_backward::primitive_desc(
        bwd_pool_desc, attr, executor::global_cpu_engine, pool_fwd_pd);
    dnnl_scratchpad_mds[fwd_pool_index] = new dnnl::memory::desc(pool_fwd_pd.scratchpad_desc());
    dnnl_scratchpad_mds[bwd_pool_index] = new dnnl::memory::desc(pool_bwd_pd.scratchpad_desc());

    size_t fprop_src_index = fdeps[0];
    build_memory(dnnl_memories, fprop_src_desc, fprop_src_index);
    size_t diff_dst_index = bdeps[0];
    build_memory(dnnl_memories, pool_bwd_pd.diff_dst_desc(), diff_dst_index);
    size_t diff_src_index = fdeps[1];
    build_memory(dnnl_memories, pool_bwd_pd.diff_src_desc(), diff_src_index);
    bdeps[2] = diff_src_index;

    size_t ws_index = fdeps[2];
    build_memory(dnnl_memories, pool_fwd_pd.workspace_desc(), ws_index);
    bdeps[1] = ws_index;

    // Allocate workspace
    // TODO (jbobba): Might need to align memory
    auto ws =
        std::unique_ptr<DNNLWorkspace>(new DNNLWorkspace(pool_fwd_pd.workspace_desc().get_size()));
    auto ws_buf_index = insert_workspace(dnnl_workspaces, ws);
    fdeps[3] = ws_buf_index;
    bdeps[3] = ws_buf_index;

    dnnl_primitives[fwd_pool_index] = new dnnl::pooling_forward(pool_fwd_pd);

    dnnl_primitives[bwd_pool_index] = new dnnl::pooling_backward(pool_bwd_pd);
}

void DNNLEmitter::build_max_pooling_with_indices_forward(
    std::vector<dnnl::memory*>& dnnl_memories,
    std::vector<dnnl::primitive*>& dnnl_primitives,
    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
    const dnnl::pooling_forward::desc& max_pool_desc,
    const std::vector<size_t>& deps,
    size_t max_pool_index)
{
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pool_pd =
        dnnl::pooling_forward::primitive_desc(max_pool_desc, attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[max_pool_index] = new dnnl::memory::desc(pool_pd.scratchpad_desc());

    size_t src_index = deps[0];
    build_memory(dnnl_memories, pool_pd.src_desc(), src_index);
    size_t dst_index = deps[1];
    build_memory(dnnl_memories, pool_pd.dst_desc(), dst_index);

    size_t ws_index = deps[2];
    build_memory(dnnl_memories, pool_pd.workspace_desc(), ws_index);

    dnnl_primitives[max_pool_index] = new dnnl::pooling_forward(pool_pd);
}

void DNNLEmitter::build_max_pooling_with_indices_backward(
    std::vector<dnnl::memory*>& dnnl_memories,
    std::vector<dnnl::primitive*>& dnnl_primitives,
    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
    const dnnl::pooling_backward::desc& bwd_pool_desc,
    const dnnl::pooling_forward::desc& fwd_pool_desc,
    const std::vector<size_t>& deps,
    size_t max_pool_index)
{
    auto pool_fwd_pd =
        dnnl::pooling_forward::primitive_desc(fwd_pool_desc, executor::global_cpu_engine);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pool_bwd_pd = dnnl::pooling_backward::primitive_desc(
        bwd_pool_desc, attr, executor::global_cpu_engine, pool_fwd_pd);
    dnnl_scratchpad_mds[max_pool_index] = new dnnl::memory::desc(pool_bwd_pd.scratchpad_desc());

    size_t diff_dst_index = deps[0];
    build_memory(dnnl_memories, pool_bwd_pd.diff_dst_desc(), diff_dst_index);
    size_t diff_src_index = deps[2];
    build_memory(dnnl_memories, pool_bwd_pd.diff_src_desc(), diff_src_index);

    size_t fprop_ws_index = deps[1];
    build_memory(dnnl_memories, pool_fwd_pd.workspace_desc(), fprop_ws_index);

    dnnl_primitives[max_pool_index] = new dnnl::pooling_backward(pool_bwd_pd);
}

void DNNLEmitter::build_reorder(std::vector<dnnl::memory*>& dnnl_memories,
                                std::vector<dnnl::primitive*>& dnnl_primitives,
                                std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                const dnnl::memory::desc& input_desc,
                                const dnnl::memory::desc& result_desc,
                                const std::vector<size_t>& deps,
                                size_t reorder_index)
{
    size_t input_index = deps[0];
    build_memory(dnnl_memories, input_desc, input_index);
    size_t result_index = deps[1];
    build_memory(dnnl_memories, result_desc, result_index);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto reorder_pd = dnnl::reorder::primitive_desc(
        *dnnl_memories[input_index], *dnnl_memories[result_index], attr);
    dnnl_scratchpad_mds[reorder_index] = new dnnl::memory::desc(reorder_pd.scratchpad_desc());
    dnnl_primitives[reorder_index] = new dnnl::reorder(reorder_pd);
}

void DNNLEmitter::build_lrn_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                    std::vector<dnnl::primitive*>& dnnl_primitives,
                                    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                    const dnnl::lrn_forward::desc& lrn_desc,
                                    const std::vector<size_t>& deps,
                                    size_t lrn_index)
{
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto lrn_pd = dnnl::lrn_forward::primitive_desc(lrn_desc, attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[lrn_index] = new dnnl::memory::desc(lrn_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, lrn_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(dnnl_memories, lrn_pd.dst_desc(), result_index);

    dnnl_primitives[lrn_index] = new dnnl::lrn_forward(lrn_pd);
}

void DNNLEmitter::build_relu_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                     std::vector<dnnl::primitive*>& dnnl_primitives,
                                     std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                     const dnnl::eltwise_forward::desc& relu_desc,
                                     const std::vector<size_t>& deps,
                                     size_t relu_index)
{
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto relu_pd =
        dnnl::eltwise_forward::primitive_desc(relu_desc, attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[relu_index] = new dnnl::memory::desc(relu_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, relu_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(dnnl_memories, relu_pd.dst_desc(), result_index);

    dnnl_primitives[relu_index] = new dnnl::eltwise_forward(relu_pd);
}

void DNNLEmitter::build_relu_backward(std::vector<dnnl::memory*>& dnnl_memories,
                                      std::vector<dnnl::primitive*>& dnnl_primitives,
                                      std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                      const dnnl::eltwise_backward::desc& bwd_desc,
                                      const dnnl::eltwise_forward::desc& fwd_desc,
                                      const std::vector<size_t>& deps,
                                      size_t relu_index)
{
    /* create forward relu primitive descriptor*/
    auto relu_fwd_pd = dnnl::eltwise_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    /* create backward relu primitive_descriptor */
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto relu_bwd_pd = dnnl::eltwise_backward::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, relu_fwd_pd);
    dnnl_scratchpad_mds[relu_index] = new dnnl::memory::desc(relu_bwd_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, relu_bwd_pd.src_desc(), input_index);
    size_t delta_index = deps[1];
    build_memory(dnnl_memories, relu_bwd_pd.diff_dst_desc(), delta_index);
    size_t result_index = deps[2];
    build_memory(dnnl_memories, relu_bwd_pd.diff_src_desc(), result_index);

    dnnl_primitives[relu_index] = new dnnl::eltwise_backward(relu_bwd_pd);
}

void DNNLEmitter::build_sigmoid_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                        std::vector<dnnl::primitive*>& dnnl_primitives,
                                        std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                        const dnnl::eltwise_forward::desc& sigmoid_desc,
                                        const std::vector<size_t>& deps,
                                        size_t sigmoid_index)
{
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto sigmoid_pd =
        dnnl::eltwise_forward::primitive_desc(sigmoid_desc, attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[sigmoid_index] = new dnnl::memory::desc(sigmoid_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, sigmoid_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(dnnl_memories, sigmoid_pd.dst_desc(), result_index);

    dnnl_primitives[sigmoid_index] = new dnnl::eltwise_forward(sigmoid_pd);
}

void DNNLEmitter::build_sigmoid_backward(std::vector<dnnl::memory*>& dnnl_memories,
                                         std::vector<dnnl::primitive*>& dnnl_primitives,
                                         std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                         const dnnl::eltwise_backward::desc& bwd_desc,
                                         const dnnl::eltwise_forward::desc& fwd_desc,
                                         const std::vector<size_t>& deps,
                                         size_t sigmoid_index)
{
    // sigmoid forward primitive desc
    auto sigmoid_fwd_pd =
        dnnl::eltwise_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto sigmoid_bwd_pd = dnnl::eltwise_backward::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, sigmoid_fwd_pd);
    dnnl_scratchpad_mds[sigmoid_index] = new dnnl::memory::desc(sigmoid_bwd_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, sigmoid_bwd_pd.src_desc(), input_index);
    size_t delta_index = deps[1];
    build_memory(dnnl_memories, sigmoid_bwd_pd.diff_dst_desc(), delta_index);
    size_t result_index = deps[2];
    build_memory(dnnl_memories, sigmoid_bwd_pd.diff_dst_desc(), result_index);

    dnnl_primitives[sigmoid_index] = new dnnl::eltwise_backward(sigmoid_bwd_pd);
}

void DNNLEmitter::build_elementwise_add(std::vector<dnnl::memory*>& dnnl_memories,
                                        std::vector<dnnl::primitive*>& dnnl_primitives,
                                        std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                        const dnnl::sum::primitive_desc& sum_pd,
                                        const std::vector<size_t>& deps,
                                        size_t add_index)
{
    size_t input0_data_index = deps[0];
    build_memory(dnnl_memories, sum_pd.dst_desc(), input0_data_index);
    size_t input1_data_index = deps[1];
    build_memory(dnnl_memories, sum_pd.dst_desc(), input1_data_index);
    size_t result_index = deps[2];
    build_memory(dnnl_memories, sum_pd.dst_desc(), result_index);

    dnnl_scratchpad_mds[add_index] = new dnnl::memory::desc(sum_pd.scratchpad_desc());

    // sum primitive
    dnnl_primitives[add_index] = new dnnl::sum(sum_pd);
}

void DNNLEmitter::build_batchnorm_forward(
    std::vector<dnnl::memory*>& dnnl_memories,
    std::vector<dnnl::primitive*>& dnnl_primitives,
    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
    const dnnl::batch_normalization_forward::desc& batchnorm_desc,
    const dnnl::memory::desc& weights_desc,
    bool bn_training_flag,
    const std::vector<size_t>& deps,
    size_t batchnorm_index,
    const dnnl::post_ops& pops)
{
    dnnl::primitive_attr bn_attr;
    bn_attr.set_post_ops(pops);
    bn_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto batchnorm_pd = dnnl::batch_normalization_forward::primitive_desc(
        batchnorm_desc, bn_attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[batchnorm_index] = new dnnl::memory::desc(batchnorm_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, batchnorm_pd.src_desc(), input_index);

    auto use_global_stats = batchnorm_desc.data.flags & 0x1U;
    if (bn_training_flag && !use_global_stats)
    {
        size_t weights_index = deps[1];
        build_memory(dnnl_memories, weights_desc, weights_index);
        size_t result_index = deps[2];
        build_memory(dnnl_memories, batchnorm_pd.dst_desc(), result_index);
        size_t mean_index = deps[3];
        build_memory(dnnl_memories, batchnorm_pd.mean_desc(), mean_index);
        size_t variance_index = deps[4];
        build_memory(dnnl_memories, batchnorm_pd.variance_desc(), variance_index);

        dnnl_primitives[batchnorm_index] = new dnnl::batch_normalization_forward(batchnorm_pd);
    }
    else
    {
        size_t weights_index = deps[3];
        build_memory(dnnl_memories, weights_desc, weights_index);
        size_t result_index = deps[4];
        build_memory(dnnl_memories, batchnorm_pd.dst_desc(), result_index);
        size_t mean_index = deps[1];
        build_memory(dnnl_memories, batchnorm_pd.mean_desc(), mean_index);
        size_t variance_index = deps[2];
        build_memory(dnnl_memories, batchnorm_pd.variance_desc(), variance_index);

        dnnl_primitives[batchnorm_index] = new dnnl::batch_normalization_forward(batchnorm_pd);
    }
}

void DNNLEmitter::build_batchnorm_backward(
    std::vector<dnnl::memory*>& dnnl_memories,
    std::vector<dnnl::primitive*>& dnnl_primitives,
    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
    const dnnl::batch_normalization_backward::desc& batchnorm_desc,
    const dnnl::memory::desc& input_desc,
    const dnnl::memory::desc& weights_desc,
    const dnnl::memory::desc& dweights_desc,
    float epsilon,
    const std::vector<size_t>& deps,
    size_t batchnorm_index)
{
    auto batchnorm_fdesc =
        dnnl::batch_normalization_forward::desc(dnnl::prop_kind::forward_training,
                                                input_desc,
                                                epsilon,
                                                dnnl::BN_FLAG_CLASS::use_scale_shift);
    auto batchnorm_fpd = dnnl::batch_normalization_forward::primitive_desc(
        batchnorm_fdesc, executor::global_cpu_engine);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto batchnorm_pd = dnnl::batch_normalization_backward::primitive_desc(
        batchnorm_desc, attr, executor::global_cpu_engine, batchnorm_fpd);
    dnnl_scratchpad_mds[batchnorm_index] = new dnnl::memory::desc(batchnorm_pd.scratchpad_desc());

    size_t weights_index = deps[0];
    build_memory(dnnl_memories, weights_desc, weights_index);
    size_t input_index = deps[1];
    build_memory(dnnl_memories, batchnorm_pd.src_desc(), input_index);
    size_t mean_index = deps[2];
    build_memory(dnnl_memories, batchnorm_pd.mean_desc(), mean_index);
    size_t variance_index = deps[3];
    build_memory(dnnl_memories, batchnorm_pd.variance_desc(), variance_index);
    size_t delta_index = deps[4];
    build_memory(dnnl_memories, batchnorm_pd.diff_src_desc(), delta_index);
    size_t dinput_index = deps[5];
    build_memory(dnnl_memories, batchnorm_pd.dst_desc(), dinput_index);
    size_t dweights_index = deps[6];
    build_memory(dnnl_memories, dweights_desc, dweights_index);

    dnnl_primitives[batchnorm_index] = new dnnl::batch_normalization_backward(batchnorm_pd);
}

void DNNLEmitter::build_vanilla_rnn_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                            std::vector<dnnl::primitive*>& dnnl_primitives,
                                            std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                            std::vector<char*>& dnnl_workspaces,
                                            const dnnl::vanilla_rnn_forward::desc& rnn_desc,
                                            std::vector<size_t>& deps,
                                            size_t rnn_index)
{
    size_t src_layer_index = deps[0];
    build_memory(dnnl_memories, rnn_desc.data.src_layer_desc, src_layer_index);
    size_t src_iter_index = deps[1];
    build_memory(dnnl_memories, rnn_desc.data.src_iter_desc, src_iter_index);
    size_t weights_layer_index = deps[2];
    build_memory(dnnl_memories, rnn_desc.data.weights_layer_desc, weights_layer_index);
    size_t weights_iter_index = deps[3];
    build_memory(dnnl_memories, rnn_desc.data.weights_iter_desc, weights_iter_index);
    size_t bias_index = deps[4];
    build_memory(dnnl_memories, rnn_desc.data.bias_desc, bias_index);
    size_t dst_layer_index = deps[5];
    build_memory(dnnl_memories, rnn_desc.data.dst_layer_desc, dst_layer_index);
    size_t dst_iter_index = deps[6];
    build_memory(dnnl_memories, rnn_desc.data.dst_iter_desc, dst_iter_index);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto rnn_layer_prim_desc =
        dnnl::vanilla_rnn_forward::primitive_desc(rnn_desc, attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[rnn_index] = new dnnl::memory::desc(rnn_layer_prim_desc.scratchpad_desc());
    size_t workspace_index = deps[7];
    build_memory(dnnl_memories, rnn_layer_prim_desc.workspace_desc(), workspace_index);
    auto workspace = std::unique_ptr<DNNLWorkspace>(
        new DNNLWorkspace(rnn_layer_prim_desc.workspace_desc().get_size()));
    auto workspace_buf_index = insert_workspace(dnnl_workspaces, workspace);
    deps[8] = workspace_buf_index;

    dnnl_primitives[rnn_index] = new dnnl::vanilla_rnn_forward(rnn_layer_prim_desc);
}

void DNNLEmitter::build_rnn_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                    std::vector<dnnl::primitive*>& dnnl_primitives,
                                    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                    std::vector<char*>& dnnl_workspaces,
                                    const dnnl::lstm_forward::desc& rnn_desc,
                                    std::vector<size_t>& deps,
                                    size_t rnn_index)
{
    size_t src_layer_index = deps[0];
    build_memory(dnnl_memories, rnn_desc.data.src_layer_desc, src_layer_index);
    size_t src_iter_index = deps[1];
    build_memory(dnnl_memories, rnn_desc.data.src_iter_desc, src_iter_index);
    size_t src_iter_c_index = deps[2];
    build_memory(dnnl_memories, rnn_desc.data.src_iter_c_desc, src_iter_c_index);
    size_t weights_layer_index = deps[3];
    build_memory(dnnl_memories, rnn_desc.data.weights_layer_desc, weights_layer_index);
    size_t weights_iter_index = deps[4];
    build_memory(dnnl_memories, rnn_desc.data.weights_iter_desc, weights_iter_index);
    size_t bias_index = deps[5];
    build_memory(dnnl_memories, rnn_desc.data.bias_desc, bias_index);
    size_t dst_layer_index = deps[6];
    build_memory(dnnl_memories, rnn_desc.data.dst_layer_desc, dst_layer_index);
    size_t dst_iter_index = deps[7];
    build_memory(dnnl_memories, rnn_desc.data.dst_iter_desc, dst_iter_index);
    size_t dst_iter_c_index = deps[8];
    build_memory(dnnl_memories, rnn_desc.data.dst_iter_c_desc, dst_iter_c_index);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto rnn_layer_prim_desc =
        dnnl::lstm_forward::primitive_desc(rnn_desc, attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[rnn_index] = new dnnl::memory::desc(rnn_layer_prim_desc.scratchpad_desc());

    size_t workspace_index = deps[9];
    build_memory(dnnl_memories, rnn_layer_prim_desc.workspace_desc(), workspace_index);
    auto workspace = std::unique_ptr<DNNLWorkspace>(
        new DNNLWorkspace(rnn_layer_prim_desc.workspace_desc().get_size()));
    auto workspace_buf_index = insert_workspace(dnnl_workspaces, workspace);
    deps[10] = workspace_buf_index;

    dnnl_primitives[rnn_index] = new dnnl::lstm_forward(rnn_layer_prim_desc);
}

void DNNLEmitter::build_concat(std::vector<dnnl::memory*>& dnnl_memories,
                               std::vector<dnnl::primitive*>& dnnl_primitives,
                               std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                               const dnnl::concat::primitive_desc& concat_pd,
                               const std::vector<dnnl::memory::desc>& inputs_data_desc,
                               const std::vector<size_t>& deps,
                               size_t concat_index)
{
    for (size_t i = 0; i < inputs_data_desc.size(); i++)
    {
        size_t inputs_data_index = deps[i];
        build_memory(dnnl_memories, inputs_data_desc[i], inputs_data_index);
    }
    size_t result_index = deps[inputs_data_desc.size()];
    build_memory(dnnl_memories, concat_pd.dst_desc(), result_index);

    dnnl_scratchpad_mds[concat_index] = new dnnl::memory::desc(concat_pd.scratchpad_desc());

    // concat primitive
    dnnl_primitives[concat_index] = new dnnl::concat(concat_pd);
}

void DNNLEmitter::build_slice(std::vector<dnnl::memory*>& dnnl_memories,
                              std::vector<dnnl::primitive*>& dnnl_primitives,
                              std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                              dnnl::memory::desc input_desc,
                              const dnnl::memory::desc& result_desc,
                              const ngraph::Coordinate& lower_bounds,
                              const ngraph::Shape& result_shape,
                              const std::vector<size_t>& deps,
                              size_t slice_index)
{
    auto dims = dnnl::memory::dims(result_shape.begin(), result_shape.end());
    auto offsets = dnnl::memory::dims(lower_bounds.begin(), lower_bounds.end());
    auto input_sub_desc = input_desc.submemory_desc(dims, offsets);
    size_t input_index = deps[0];
    build_memory(dnnl_memories, input_sub_desc, input_index);

    size_t result_index = deps[1];
    build_memory(dnnl_memories, result_desc, result_index);

    // reorder primitive
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto reorder_pd = dnnl::reorder::primitive_desc(
        *dnnl_memories[input_index], *dnnl_memories[result_index], attr);
    dnnl_scratchpad_mds[slice_index] = new dnnl::memory::desc(reorder_pd.scratchpad_desc());

    dnnl_primitives[slice_index] = new dnnl::reorder(reorder_pd);
}

void DNNLEmitter::build_softmax_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                        std::vector<dnnl::primitive*>& dnnl_primitives,
                                        std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                        const dnnl::softmax_forward::desc& softmax_desc,
                                        const std::vector<size_t>& deps,
                                        size_t softmax_index)
{
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto softmax_pd =
        dnnl::softmax_forward::primitive_desc(softmax_desc, attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[softmax_index] = new dnnl::memory::desc(softmax_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, softmax_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(dnnl_memories, softmax_pd.dst_desc(), result_index);

    dnnl_primitives[softmax_index] = new dnnl::softmax_forward(softmax_pd);
}

void DNNLEmitter::build_leaky_relu(std::vector<dnnl::memory*>& dnnl_memories,
                                   std::vector<dnnl::primitive*>& dnnl_primitives,
                                   std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                   const dnnl::eltwise_forward::desc& leaky_relu_desc,
                                   const std::vector<size_t>& deps,
                                   size_t leaky_relu_index)
{
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto leaky_relu_pd =
        dnnl::eltwise_forward::primitive_desc(leaky_relu_desc, attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[leaky_relu_index] = new dnnl::memory::desc(leaky_relu_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, leaky_relu_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(dnnl_memories, leaky_relu_pd.dst_desc(), result_index);

    dnnl_primitives[leaky_relu_index] = new dnnl::eltwise_forward(leaky_relu_pd);
}

void DNNLEmitter::build_bounded_relu(std::vector<dnnl::memory*>& dnnl_memories,
                                     std::vector<dnnl::primitive*>& dnnl_primitives,
                                     std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                     const dnnl::eltwise_forward::desc& bounded_relu_desc,
                                     const std::vector<size_t>& deps,
                                     size_t bounded_relu_index)
{
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto bounded_relu_pd =
        dnnl::eltwise_forward::primitive_desc(bounded_relu_desc, attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[bounded_relu_index] =
        new dnnl::memory::desc(bounded_relu_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, bounded_relu_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(dnnl_memories, bounded_relu_pd.dst_desc(), result_index);

    dnnl_primitives[bounded_relu_index] = new dnnl::eltwise_forward(bounded_relu_pd);
}

void DNNLEmitter::build_gelu(std::vector<dnnl::memory*>& dnnl_memories,
                             std::vector<dnnl::primitive*>& dnnl_primitives,
                             std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                             const dnnl::eltwise_forward::desc& gelu_desc,
                             const std::vector<size_t>& deps,
                             size_t gelu_index)
{
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto gelu_pd =
        dnnl::eltwise_forward::primitive_desc(gelu_desc, attr, executor::global_cpu_engine);
    dnnl_scratchpad_mds[gelu_index] = new dnnl::memory::desc(gelu_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, gelu_pd.src_desc(), input_index);
    size_t result_index = deps[1];
    build_memory(dnnl_memories, gelu_pd.dst_desc(), result_index);

    dnnl_primitives[gelu_index] = new dnnl::eltwise_forward(gelu_pd);
}

void DNNLEmitter::build_gelu_backward(std::vector<dnnl::memory*>& dnnl_memories,
                                      std::vector<dnnl::primitive*>& dnnl_primitives,
                                      std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                      const dnnl::eltwise_backward::desc& bwd_desc,
                                      const dnnl::eltwise_forward::desc& fwd_desc,
                                      const std::vector<size_t>& deps,
                                      size_t gelu_bprop_index)
{
    // gelu forward primitive desc
    auto gelu_fwd_pd = dnnl::eltwise_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto gelu_bwd_pd = dnnl::eltwise_backward::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, gelu_fwd_pd);
    dnnl_scratchpad_mds[gelu_bprop_index] = new dnnl::memory::desc(gelu_bwd_pd.scratchpad_desc());

    size_t input_index = deps[0];
    build_memory(dnnl_memories, gelu_bwd_pd.src_desc(), input_index);
    size_t delta_index = deps[1];
    build_memory(dnnl_memories, gelu_bwd_pd.diff_dst_desc(), delta_index);
    size_t result_index = deps[2];
    build_memory(dnnl_memories, gelu_bwd_pd.diff_dst_desc(), result_index);

    dnnl_primitives[gelu_bprop_index] = new dnnl::eltwise_backward(gelu_bwd_pd);
}

size_t DNNLEmitter::query_scratchpad_sum(const dnnl::sum::primitive_desc pd)
{
    dnnl::memory::desc scratchpad_md = pd.scratchpad_desc();
    auto size = scratchpad_md.get_size();
    m_max_scratchpad_size = size > m_max_scratchpad_size ? size : m_max_scratchpad_size;
    return size;
}

size_t DNNLEmitter::query_scratchpad_concat(const dnnl::concat::primitive_desc pd)
{
    dnnl::memory::desc scratchpad_md = pd.scratchpad_desc();
    auto size = scratchpad_md.get_size();
    m_max_scratchpad_size = size > m_max_scratchpad_size ? size : m_max_scratchpad_size;
    return size;
}

size_t DNNLEmitter::query_scratchpad_pooling_forward(const dnnl::pooling_forward::desc& desc)
{
    ATTR_S
    auto pd = dnnl::pooling_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

size_t
    DNNLEmitter::query_scratchpad_avg_pooling_backward(const dnnl::pooling_forward::desc& fwd_desc,
                                                       const dnnl::pooling_backward::desc& bwd_desc)
{
    ATTR_S
    auto fwd_pd = dnnl::pooling_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);
    auto pd =
        dnnl::pooling_backward::primitive_desc(bwd_desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
}

size_t
    DNNLEmitter::query_scratchpad_max_pooling_backward(const dnnl::pooling_forward::desc& fwd_desc,
                                                       const dnnl::pooling_backward::desc& bwd_desc)
{
    ATTR_S
    auto fwd_pd =
        dnnl::pooling_forward::primitive_desc(fwd_desc, attr, executor::global_cpu_engine);
    auto pd =
        dnnl::pooling_backward::primitive_desc(bwd_desc, attr, executor::global_cpu_engine, fwd_pd);
    dnnl::memory::desc scratchpad_md = pd.scratchpad_desc();
    size_t size = scratchpad_md.get_size();
    m_max_scratchpad_size = size > m_max_scratchpad_size ? size : m_max_scratchpad_size;
    dnnl::memory::desc fwd_scratchpad_md = fwd_pd.scratchpad_desc();
    size_t f_size = fwd_scratchpad_md.get_size();
    m_max_scratchpad_size = f_size > m_max_scratchpad_size ? f_size : m_max_scratchpad_size;
    return size > f_size ? size : f_size;
}

size_t DNNLEmitter::query_scratchpad_max_pooling_with_indices_backward(
    const dnnl::pooling_forward::desc& fwd_desc, const dnnl::pooling_backward::desc& bwd_desc)
{
    ATTR_S
    auto fwd_pd =
        dnnl::pooling_forward::primitive_desc(fwd_desc, attr, executor::global_cpu_engine);
    auto pd =
        dnnl::pooling_backward::primitive_desc(bwd_desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
}

size_t DNNLEmitter::query_scratchpad_batchnorm_forward(
    const dnnl::batch_normalization_forward::desc& desc, const dnnl::post_ops& pops)
{
    ATTR_S
    attr.set_post_ops(pops);
    auto pd =
        dnnl::batch_normalization_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}
size_t DNNLEmitter::query_scratchpad_batchnorm_backward(
    const dnnl::batch_normalization_backward::desc& desc,
    const dnnl::memory::desc& input_desc,
    float epsilon)
{
    ATTR_S
    auto fwd_desc = dnnl::batch_normalization_forward::desc(dnnl::prop_kind::forward_training,
                                                            input_desc,
                                                            epsilon,
                                                            dnnl::BN_FLAG_CLASS::use_scale_shift);
    auto fwd_pd =
        dnnl::batch_normalization_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);
    auto pd = dnnl::batch_normalization_backward::primitive_desc(
        desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
}

size_t
    DNNLEmitter::query_scratchpad_convolution_forward(const dnnl::convolution_forward::desc& desc,
                                                      dnnl::primitive_attr& attr)
{
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pd = dnnl::convolution_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

size_t DNNLEmitter::query_scratchpad_convolution_backward_data(
    const dnnl::convolution_forward::desc& fwd_desc,
    const dnnl::convolution_backward_data::desc& bwd_desc)
{
    ATTR_S
    auto fwd_pd = dnnl::convolution_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);
    auto pd = dnnl::convolution_backward_data::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
}

size_t DNNLEmitter::query_scratchpad_convolution_backward_weights(
    const dnnl::convolution_forward::desc& fwd_desc,
    const dnnl::convolution_backward_weights::desc& bwd_desc)
{
    ATTR_S
    auto fwd_pd = dnnl::convolution_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);
    auto pd = dnnl::convolution_backward_weights::primitive_desc(
        bwd_desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
}

size_t DNNLEmitter::query_scratchpad_deconvolution_forward(
    const dnnl::deconvolution_forward::desc& desc)
{
    ATTR_S
    auto pd = dnnl::deconvolution_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

size_t DNNLEmitter::query_scratchpad_eltwise_forward(const dnnl::eltwise_forward::desc& desc)
{
    ATTR_S
    auto pd = dnnl::eltwise_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

size_t DNNLEmitter::query_scratchpad_eltwise_backward(const dnnl::eltwise_forward::desc& fwd_desc,
                                                      const dnnl::eltwise_backward::desc& bwd_desc)
{
    ATTR_S
    auto fwd_pd = dnnl::eltwise_forward::primitive_desc(fwd_desc, executor::global_cpu_engine);
    auto pd =
        dnnl::eltwise_backward::primitive_desc(bwd_desc, attr, executor::global_cpu_engine, fwd_pd);
    GET_SIZE
}

size_t DNNLEmitter::query_scratchpad_ip_forward(const dnnl::inner_product_forward::desc& desc,
                                                dnnl::primitive_attr& attr)
{
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pd = dnnl::inner_product_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

size_t DNNLEmitter::query_scratchpad_reorder(const dnnl::memory::desc& input_desc,
                                             const dnnl::memory::desc& result_desc)
{
    ATTR_S
    auto pd = dnnl::reorder::primitive_desc(
        executor::global_cpu_engine, input_desc, executor::global_cpu_engine, result_desc, attr);
    GET_SIZE
}

size_t DNNLEmitter::query_scratchpad_rnn_forward(const dnnl::lstm_forward::desc& desc)
{
    ATTR_S
    auto pd = dnnl::lstm_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

size_t
    DNNLEmitter::query_scratchpad_vanilla_rnn_forward(const dnnl::vanilla_rnn_forward::desc& desc)
{
    ATTR_S
    auto pd = dnnl::vanilla_rnn_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

size_t DNNLEmitter::query_scratchpad_lrn_forward(const dnnl::lrn_forward::desc& desc)
{
    ATTR_S
    auto pd = dnnl::lrn_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}

size_t DNNLEmitter::query_scratchpad_slice(dnnl::memory::desc& input_desc,
                                           const dnnl::memory::desc& output_desc,
                                           const ngraph::Coordinate& lower_bounds,
                                           const ngraph::Shape& result_shape)
{
    ATTR_S
    auto dims = dnnl::memory::dims(result_shape.begin(), result_shape.end());
    auto offsets = dnnl::memory::dims(lower_bounds.begin(), lower_bounds.end());
    auto input_sub_desc = input_desc.submemory_desc(dims, offsets);
    auto input = dnnl::memory(input_sub_desc, executor::global_cpu_engine);
    auto output = dnnl::memory(output_desc, executor::global_cpu_engine);
    auto pd = dnnl::reorder::primitive_desc(input, output, attr);
    GET_SIZE
}

size_t DNNLEmitter::query_scratchpad_softmax_forward(const dnnl::softmax_forward::desc& desc)
{
    ATTR_S
    auto pd = dnnl::softmax_forward::primitive_desc(desc, attr, executor::global_cpu_engine);
    GET_SIZE
}
