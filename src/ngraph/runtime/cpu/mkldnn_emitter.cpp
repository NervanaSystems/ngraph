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
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace ngraph::runtime::cpu;

const std::vector<mkldnn::primitive*>& MKLDNNEmitter::get_mkldnn_primitives() const
{
    return m_mkldnn_primitives;
}

size_t MKLDNNEmitter::insert_primitive(mkldnn::primitive* primitive)
{
    m_mkldnn_primitives.emplace_back(primitive);
    return (m_mkldnn_primitives.size() - 1);
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


size_t MKLDNNEmitter::build_convolution_backward_data(const mkldnn::memory::desc &in_weights_desc,
                                                      const mkldnn::memory::desc &in_delta_desc,
                                                      const mkldnn::memory::desc &out_data_delta_desc,
                                                      const ngraph::Strides &ng_strides,
                                                      const ngraph::Strides &ng_dilation_strides,
                                                      const ngraph::CoordinateDiff &ng_padding_below,
                                                      const ngraph::CoordinateDiff &ng_padding_above)

{
    const size_t in_weights_index = build_memory_primitive(in_weights_desc);
    const size_t in_delta_index = build_memory_primitive(in_delta_desc);
    const size_t out_data_delta_index = build_memory_primitive(out_data_delta_desc);

    mkldnn::memory::dims strides(ng_strides.begin(), ng_strides.end());
    mkldnn::memory::dims dilation(ng_dilation_strides.begin(), ng_dilation_strides.end());
    mkldnn::memory::dims padding_l(ng_padding_below.begin(), ng_padding_below.end());
    mkldnn::memory::dims padding_r(ng_padding_above.begin(), ng_padding_above.end());
    mkldnn::convolution_forward::primitive_desc fwd_pd{{mkldnn::prop_kind::forward,
                                                        mkldnn::algorithm::convolution_direct,
                                                        out_data_delta_desc,
                                                        in_weights_desc,
                                                        in_delta_desc,
                                                        strides,
                                                        dilation,
                                                        padding_l,
                                                        padding_r,
                                                        mkldnn::padding_kind::zero},
                                                       mkldnn_utils::global_cpu_engine};

    mkldnn::convolution_backward_data::primitive_desc bwd_pd{{mkldnn::algorithm::convolution_direct,
                                                                 out_data_delta_desc,
                                                                 in_weights_desc,
                                                                 in_delta_desc,
                                                                 strides,
                                                                 dilation,
                                                                 padding_l,
                                                                 padding_r,
                                                                 mkldnn::padding_kind::zero},
                                                                mkldnn_utils::global_cpu_engine,
                                                                fwd_pd};

    const size_t conv_index = insert_primitive(new mkldnn::convolution_backward_data(bwd_pd,
                                                                                     *m_mkldnn_primitives[in_delta_index],
                                                                                     *m_mkldnn_primitives[in_weights_index],
                                                                                     *m_mkldnn_primitives[out_data_delta_index]));

    m_primitive_deps[conv_index] = {in_weights_index, in_delta_index, out_data_delta_index};
    return conv_index;
}

size_t MKLDNNEmitter::build_convolution_backward_filters_bias(const mkldnn::memory::desc &in_data_desc,
                                                              const mkldnn::memory::desc &in_delta_desc,
                                                              const mkldnn::memory::desc &out_weights_delta_desc,
                                                              const mkldnn::memory::desc &out_bias_delta_desc,
                                                              const ngraph::Strides &ng_strides,
                                                              const ngraph::Strides &ng_dilation_strides,
                                                              const ngraph::CoordinateDiff &ng_padding_below,
                                                              const ngraph::CoordinateDiff &ng_padding_above)
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

    mkldnn::convolution_backward_weights::primitive_desc bwd_pd{{mkldnn::algorithm::convolution_direct,
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

    const size_t conv_index = insert_primitive(new mkldnn::convolution_backward_weights(bwd_pd,
            *m_mkldnn_primitives[in_data_index],
            *m_mkldnn_primitives[in_delta_index],
            *m_mkldnn_primitives[out_weights_delta_index],
            *m_mkldnn_primitives[out_bias_delta_index]));

    m_primitive_deps[conv_index] = {in_data_index, in_delta_index, out_weights_delta_index, out_bias_delta_index};
    return conv_index;
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
