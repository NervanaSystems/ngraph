/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <memory>
#include <string>

#include "mkldnn_emitter.hpp"

#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph::runtime::cpu;

MKLDNNEmitter::~MKLDNNEmitter()
{
    for (auto p : m_mkldnn_primitives)
        delete p;
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
    return mkldnn::memory::desc(
        mkldnn::memory::dims(tvw.get_shape().begin(), tvw.get_shape().end()),
        mkldnn_utils::get_mkldnn_data_type(tvw.get_element_type()),
        fmt);
}

mkldnn::memory::desc MKLDNNEmitter::build_memory_descriptor(const TensorViewWrapper& tvw) const
{
    auto layout =
        std::static_pointer_cast<LayoutDescriptor>(tvw.get_tensor_view()->get_tensor_view_layout());

    return build_memory_descriptor(tvw, layout->get_mkldnn_format());
}

mkldnn::memory::desc MKLDNNEmitter::build_memory_descriptor(const Shape& shape,
                                                            const ngraph::element::Type& et,
                                                            mkldnn::memory::format fmt) const
{
    return mkldnn::memory::desc(mkldnn::memory::dims(shape.begin(), shape.end()),
                                mkldnn_utils::get_mkldnn_data_type(et),
                                fmt);
}

mkldnn::memory MKLDNNEmitter::build_memory_primitive(const TensorViewWrapper& tvw) const
{
    return mkldnn::memory({build_memory_descriptor(tvw), mkldnn_utils::global_cpu_engine}, nullptr);
}

size_t MKLDNNEmitter::build_memory_primitive(const mkldnn::memory::desc& desc)
{
    // The MKL-DNN C++ API forces proper initialization of a memory primitive
    // with a non-null pointer (unlike the C API)
    // Primitives are initialized at runtime so we use a known-invalid address here
    // to bypass this check
    return insert_primitive(
        new mkldnn::memory({desc, mkldnn_utils::global_cpu_engine}, reinterpret_cast<void*>(0x42)));
}

size_t MKLDNNEmitter::build_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                                const mkldnn::memory::desc& weights_desc,
                                                const mkldnn::memory::desc& result_desc,
                                                const ngraph::Strides& strides,
                                                const ngraph::Strides& dilation_strides,
                                                const ngraph::CoordinateDiff& padding_below,
                                                const ngraph::CoordinateDiff& padding_above)
{
    size_t input_data_index = build_memory_primitive(input_data_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t conv_index = insert_primitive(new mkldnn::convolution_forward(
        {{mkldnn::prop_kind::forward,
          mkldnn::algorithm::convolution_direct,
          input_data_desc,
          weights_desc,
          result_desc,
          mkldnn::memory::dims(strides.begin(), strides.end()),
          mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
          mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
          mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
          mkldnn::padding_kind::zero},
         mkldnn_utils::global_cpu_engine},
        *m_mkldnn_primitives[input_data_index],
        *m_mkldnn_primitives[weights_index],
        *m_mkldnn_primitives[result_index]));

    m_primitive_deps[conv_index] = {input_data_index, weights_index, result_index};
    return conv_index;
}

size_t MKLDNNEmitter::build_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                                const mkldnn::memory::desc& weights_desc,
                                                const mkldnn::memory::desc& bias_desc,
                                                const mkldnn::memory::desc& result_desc,
                                                const ngraph::Strides& strides,
                                                const ngraph::Strides& dilation_strides,
                                                const ngraph::CoordinateDiff& padding_below,
                                                const ngraph::CoordinateDiff& padding_above)
{
    const size_t input_data_index = build_memory_primitive(input_data_desc);
    const size_t weights_index = build_memory_primitive(weights_desc);
    const size_t bias_index = build_memory_primitive(bias_desc);
    const size_t result_index = build_memory_primitive(result_desc);

    const size_t conv_index = insert_primitive(new mkldnn::convolution_forward(
        {{mkldnn::prop_kind::forward,
          mkldnn::algorithm::convolution_direct,
          input_data_desc,
          weights_desc,
          bias_desc,
          result_desc,
          mkldnn::memory::dims(strides.begin(), strides.end()),
          mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
          mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
          mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
          mkldnn::padding_kind::zero},
         mkldnn_utils::global_cpu_engine},
        *m_mkldnn_primitives[input_data_index],
        *m_mkldnn_primitives[weights_index],
        *m_mkldnn_primitives[bias_index],
        *m_mkldnn_primitives[result_index]));

    m_primitive_deps[conv_index] = {input_data_index, weights_index, bias_index, result_index};
    return conv_index;
}

size_t MKLDNNEmitter::build_convolution_backward_weights_bias(
    const mkldnn::memory::desc& in_data_desc,
    const mkldnn::memory::desc& in_delta_desc,
    const mkldnn::memory::desc& out_weights_delta_desc,
    const mkldnn::memory::desc& out_bias_delta_desc,
    const ngraph::Strides& ng_strides,
    const ngraph::Strides& ng_dilation_strides,
    const ngraph::CoordinateDiff& ng_padding_below,
    const ngraph::CoordinateDiff& ng_padding_above)
{
    const size_t in_data_index = build_memory_primitive(in_data_desc);
    const size_t in_delta_index = build_memory_primitive(in_delta_desc);
    const size_t out_weights_delta_index = build_memory_primitive(out_weights_delta_desc);
    const size_t out_bias_delta_index = build_memory_primitive(out_bias_delta_desc);

    mkldnn::memory::dims strides(ng_strides.begin(), ng_strides.end());
    mkldnn::memory::dims dilation(ng_dilation_strides.begin(), ng_dilation_strides.end());
    mkldnn::memory::dims padding_l(ng_padding_below.begin(), ng_padding_below.end());
    mkldnn::memory::dims padding_r(ng_padding_above.begin(), ng_padding_above.end());
    mkldnn::convolution_forward::primitive_desc fwd_pd{{mkldnn::prop_kind::forward,
                                                        mkldnn::algorithm::convolution_direct,
                                                        in_data_desc,
                                                        out_weights_delta_desc,
                                                        out_bias_delta_desc,
                                                        in_delta_desc,
                                                        strides,
                                                        dilation,
                                                        padding_l,
                                                        padding_r,
                                                        mkldnn::padding_kind::zero},
                                                       mkldnn_utils::global_cpu_engine};

    mkldnn::convolution_backward_weights::primitive_desc bwd_pd{
        {mkldnn::algorithm::convolution_direct,
         in_data_desc,
         out_weights_delta_desc,
         out_bias_delta_desc,
         in_delta_desc,
         strides,
         dilation,
         padding_l,
         padding_r,
         mkldnn::padding_kind::zero},
        mkldnn_utils::global_cpu_engine,
        fwd_pd};

    const size_t conv_index = insert_primitive(
        new mkldnn::convolution_backward_weights(bwd_pd,
                                                 *m_mkldnn_primitives[in_data_index],
                                                 *m_mkldnn_primitives[in_delta_index],
                                                 *m_mkldnn_primitives[out_weights_delta_index],
                                                 *m_mkldnn_primitives[out_bias_delta_index]));

    m_primitive_deps[conv_index] = {
        in_data_index, in_delta_index, out_weights_delta_index, out_bias_delta_index};
    return conv_index;
}

size_t
    MKLDNNEmitter::build_convolution_backward_weights(const mkldnn::memory::desc& input_desc,
                                                      const mkldnn::memory::desc& delta_desc,
                                                      const mkldnn::memory::desc& result_desc,
                                                      const ngraph::Strides& strides,
                                                      const ngraph::Strides& dilation_strides,
                                                      const ngraph::CoordinateDiff& padding_below,
                                                      const ngraph::CoordinateDiff& padding_above)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t delta_index = build_memory_primitive(delta_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index = insert_primitive(new mkldnn::convolution_backward_weights(
        {{mkldnn::algorithm::convolution_direct,
          input_desc,
          result_desc,
          delta_desc,
          mkldnn::memory::dims(strides.begin(), strides.end()),
          mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
          mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
          mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
          mkldnn::padding_kind::zero},
         mkldnn_utils::global_cpu_engine,
         // Forward primitive descriptor corresponding to this backward weights descriptor
         {{mkldnn::prop_kind::forward,
           mkldnn::algorithm::convolution_direct,
           input_desc,
           result_desc,
           delta_desc,
           mkldnn::memory::dims(strides.begin(), strides.end()),
           mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
           mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
           mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
           mkldnn::padding_kind::zero},
          mkldnn_utils::global_cpu_engine}},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[delta_index],
        *m_mkldnn_primitives[result_index]));

    m_primitive_deps[primitive_index] = {input_index, delta_index, result_index};
    return primitive_index;
}

size_t MKLDNNEmitter::build_convolution_backward_data(const mkldnn::memory::desc& weights_desc,
                                                      const mkldnn::memory::desc& delta_desc,
                                                      const mkldnn::memory::desc& result_desc,
                                                      const ngraph::Strides& strides,
                                                      const ngraph::Strides& dilation_strides,
                                                      const ngraph::CoordinateDiff& padding_below,
                                                      const ngraph::CoordinateDiff& padding_above)
{
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t delta_index = build_memory_primitive(delta_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index = insert_primitive(new mkldnn::convolution_backward_data(
        {{mkldnn::algorithm::convolution_direct,
          result_desc,
          weights_desc,
          delta_desc,
          mkldnn::memory::dims(strides.begin(), strides.end()),
          mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
          mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
          mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
          mkldnn::padding_kind::zero},
         mkldnn_utils::global_cpu_engine,
         // Forward primitive descriptor corresponding to this backward data descriptor
         {{mkldnn::prop_kind::forward,
           mkldnn::algorithm::convolution_direct,
           result_desc,
           weights_desc,
           delta_desc,
           mkldnn::memory::dims(strides.begin(), strides.end()),
           mkldnn::memory::dims(dilation_strides.begin(), dilation_strides.end()),
           mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
           mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
           mkldnn::padding_kind::zero},
          mkldnn_utils::global_cpu_engine}},
        *m_mkldnn_primitives[delta_index],
        *m_mkldnn_primitives[weights_index],
        *m_mkldnn_primitives[result_index]));

    m_primitive_deps[primitive_index] = {weights_index, delta_index, result_index};
    return primitive_index;
}

size_t MKLDNNEmitter::build_pooling_forward(mkldnn::algorithm pooling_algorithm,
                                            const mkldnn::memory::desc& input_desc,
                                            const mkldnn::memory::desc& result_desc,
                                            const ngraph::Strides& window_strides,
                                            const ngraph::Shape& window_shape,
                                            const ngraph::Shape& padding_below,
                                            const ngraph::Shape& padding_above)
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
         mkldnn_utils::global_cpu_engine},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[result_index]));

    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
}

size_t MKLDNNEmitter::build_pooling_backward(mkldnn::algorithm pooling_algorithm,
                                             const mkldnn::memory::desc& diff_dst_desc,
                                             const mkldnn::memory::desc& diff_src_desc,
                                             const ngraph::Strides& window_strides,
                                             const ngraph::Shape& window_shape,
                                             const ngraph::Shape& padding_below,
                                             const ngraph::Shape& padding_above)
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
         mkldnn_utils::global_cpu_engine,
         {{mkldnn::prop_kind::forward_training,
           pooling_algorithm,
           diff_src_desc,
           diff_dst_desc,
           mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
           mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
           mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
           mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
           mkldnn::padding_kind::zero},
          mkldnn_utils::global_cpu_engine}},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[result_index]));

    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
}

size_t MKLDNNEmitter::build_max_pooling_backward(mkldnn::algorithm pooling_algorithm,
                                                 const mkldnn::memory::desc& fprop_src_desc,
                                                 const mkldnn::memory::desc& diff_dst_desc,
                                                 const mkldnn::memory::desc& diff_src_desc,
                                                 const ngraph::Strides& window_strides,
                                                 const ngraph::Shape& window_shape,
                                                 const ngraph::Shape& padding_below,
                                                 const ngraph::Shape& padding_above)
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
        mkldnn_utils::global_cpu_engine};

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
         mkldnn_utils::global_cpu_engine,
         fwd_pd},
        *m_mkldnn_primitives[diff_dst_index],
        *m_mkldnn_primitives[ws_index],
        *m_mkldnn_primitives[diff_src_index]));

    m_primitive_deps[fwd_primitive_index] = {
        fprop_src_index, diff_src_index, ws_index, ws_buf_index};
    m_primitive_deps[bwd_primitive_index] = {
        diff_dst_index, ws_index, diff_src_index, ws_buf_index};
    return bwd_primitive_index;
}

size_t MKLDNNEmitter::build_reorder(const mkldnn::memory::desc& input_desc,
                                    const mkldnn::memory::desc& result_desc)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index = insert_primitive(
        new mkldnn::reorder(*m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]));

    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
}

size_t MKLDNNEmitter::build_relu_forward(const mkldnn::memory::desc& input_desc,
                                         const mkldnn::memory::desc& result_desc)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index = insert_primitive(new mkldnn::relu_forward(
        {{mkldnn::prop_kind::forward_training, mkldnn::algorithm::eltwise_relu, input_desc, 0, 0},
         mkldnn_utils::global_cpu_engine},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[result_index]));

    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
}

size_t MKLDNNEmitter::build_relu_backward(const mkldnn::memory::desc& input_desc,
                                          const mkldnn::memory::desc& delta_desc,
                                          const mkldnn::memory::desc& result_desc)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t delta_index = build_memory_primitive(delta_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index = insert_primitive(new mkldnn::relu_backward(
        {{mkldnn::algorithm::eltwise_relu, delta_desc, input_desc, 0, 0},
         mkldnn_utils::global_cpu_engine,
         {{mkldnn::prop_kind::forward, mkldnn::algorithm::eltwise_relu, input_desc, 0, 0},
          mkldnn_utils::global_cpu_engine}},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[delta_index],
        *m_mkldnn_primitives[result_index]));

    m_primitive_deps[primitive_index] = {input_index, delta_index, result_index};
    return primitive_index;
}

size_t MKLDNNEmitter::build_sigmoid_forward(const mkldnn::memory::desc& input_desc,
                                            const mkldnn::memory::desc& result_desc)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index =
        insert_primitive(new mkldnn::eltwise_forward({{mkldnn::prop_kind::forward_training,
                                                       mkldnn::algorithm::eltwise_logistic,
                                                       input_desc,
                                                       0,
                                                       0},
                                                      mkldnn_utils::global_cpu_engine},
                                                     *m_mkldnn_primitives[input_index],
                                                     *m_mkldnn_primitives[result_index]));

    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
}

size_t MKLDNNEmitter::build_sigmoid_backward(const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& delta_desc,
                                             const mkldnn::memory::desc& result_desc)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t delta_index = build_memory_primitive(delta_desc);
    size_t result_index = build_memory_primitive(result_desc);

    // sigmoid forward primitive desc
    mkldnn::eltwise_forward::primitive_desc sigmoid_fwd_pd =
        mkldnn::eltwise_forward::primitive_desc(
            {mkldnn::prop_kind::forward, mkldnn::algorithm::eltwise_logistic, input_desc, 0, 0},
            mkldnn_utils::global_cpu_engine);

    size_t primitive_index = insert_primitive(new mkldnn::eltwise_backward(
        {{mkldnn::algorithm::eltwise_logistic, delta_desc, input_desc, 0, 0},
         mkldnn_utils::global_cpu_engine,
         sigmoid_fwd_pd},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[delta_index],
        *m_mkldnn_primitives[result_index]));

    m_primitive_deps[primitive_index] = {input_index, delta_index, result_index};
    return primitive_index;
}

size_t MKLDNNEmitter::build_elementwise_add(
    const mkldnn::memory::desc& input0_data_desc,
    const mkldnn::memory::desc& input1_data_desc,
    const mkldnn::memory::desc& result_desc,
    const std::vector<float>& scale_vector,
    const std::vector<mkldnn::memory::primitive_desc>& inputs_pd)

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

    m_primitive_deps[add_index] = {input0_data_index, input1_data_index, result_index};
    return add_index;
}

size_t MKLDNNEmitter::build_batchnorm_forward(const mkldnn::memory::desc& input_desc,
                                              const mkldnn::memory::desc& weights_desc,
                                              const mkldnn::memory::desc& result_desc,
                                              const mkldnn::memory::desc& mean_desc,
                                              const mkldnn::memory::desc& variance_desc,
                                              const double eps)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t result_index = build_memory_primitive(result_desc);
    size_t mean_index = build_memory_primitive(mean_desc);
    size_t variance_index = build_memory_primitive(variance_desc);

    size_t batchnorm_index = insert_primitive(new mkldnn::batch_normalization_forward(
        {{mkldnn::prop_kind::forward_training,
          input_desc,
          eps,
          mkldnn::batch_normalization_flag::use_scale_shift},
         mkldnn_utils::global_cpu_engine},
        mkldnn::primitive::at(*m_mkldnn_primitives[input_index]),
        mkldnn::primitive::at(*m_mkldnn_primitives[weights_index]),
        static_cast<mkldnn::memory>(*m_mkldnn_primitives[result_index]),
        *m_mkldnn_primitives[mean_index],
        *m_mkldnn_primitives[variance_index]));

    m_primitive_deps[batchnorm_index] = {
        input_index, weights_index, result_index, mean_index, variance_index};
    return batchnorm_index;
}

size_t MKLDNNEmitter::build_batchnorm_backward(const mkldnn::memory::desc& weights_desc,
                                               const mkldnn::memory::desc& input_desc,
                                               const mkldnn::memory::desc& mean_desc,
                                               const mkldnn::memory::desc& variance_desc,
                                               const mkldnn::memory::desc& delta_desc,
                                               const mkldnn::memory::desc& dinput_desc,
                                               const mkldnn::memory::desc& dweights_desc,
                                               const double eps)
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
         mkldnn_utils::global_cpu_engine,
         {{mkldnn::prop_kind::forward_training,
           input_desc,
           eps,
           mkldnn::batch_normalization_flag::use_scale_shift},
          mkldnn_utils::global_cpu_engine}},
        *m_mkldnn_primitives[input_index],
        *m_mkldnn_primitives[mean_index],
        *m_mkldnn_primitives[variance_index],
        *m_mkldnn_primitives[delta_index],
        *m_mkldnn_primitives[weights_index],
        *m_mkldnn_primitives[dinput_index],
        *m_mkldnn_primitives[dweights_index]));

    m_primitive_deps[batchnorm_index] = {weights_index,
                                         input_index,
                                         mean_index,
                                         variance_index,
                                         delta_index,
                                         dinput_index,
                                         dweights_index};
    return batchnorm_index;
}
