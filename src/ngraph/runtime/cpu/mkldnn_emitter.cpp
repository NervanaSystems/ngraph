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

#include <memory>
#include <string>

#include "mkldnn_emitter.hpp"

#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/dequantize.hpp"
#include "ngraph/runtime/cpu/op/quantized_avg_pool.hpp"
#include "ngraph/runtime/cpu/op/quantized_max_pool.hpp"
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
    if (fmt == mkldnn::memory::format::blocked)
    {
        throw ngraph_error("Cannot created blocked descriptor.");
    }
    return mkldnn::memory::desc(
        mkldnn::memory::dims(tvw.get_shape().begin(), tvw.get_shape().end()),
        mkldnn_utils::get_mkldnn_data_type(tvw.get_element_type()),
        fmt);
}

mkldnn::memory::desc MKLDNNEmitter::build_memory_descriptor(const Shape& shape,
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
        insert_primitive(new mkldnn::memory({desc, mkldnn_utils::global_cpu_engine}, nullptr));
    return index;
}

size_t MKLDNNEmitter::build_dequantization(const ngraph::Node* node,
                                           const mkldnn::memory::desc& input_desc,
                                           const mkldnn::memory::desc& result_desc)
{
    auto dequantize = static_cast<const ngraph::op::Dequantize*>(node);
    auto min_const_op = std::static_pointer_cast<ngraph::op::Constant>(dequantize->get_argument(1));
    auto max_const_op = std::static_pointer_cast<ngraph::op::Constant>(dequantize->get_argument(2));
    float min_range = *(static_cast<float const*>(min_const_op->get_data_ptr()));
    float max_range = *(static_cast<float const*>(max_const_op->get_data_ptr()));

    const float max_abs = std::max(std::abs(min_range), std::abs(max_range));
    bool is_signed = (dequantize->get_dequantize_et()).is_signed();
    const float target_range =
        static_cast<float>((is_signed ? std::pow(2, 7) : std::pow(2, 8)) - 1);
    const float scale_factor = max_abs / target_range;
    std::vector<float> scales;
    scales.push_back(scale_factor);

    size_t dequantize_index = 0;
    dequantize_index = this->build_quantize_reorder(input_desc, result_desc, scales);
    return dequantize_index;
}

void MKLDNNEmitter::build_quantized_max_pool(const ngraph::Node* node,
                                             std::vector<float>& quant_util)
{
    auto qmax_pool = static_cast<const ngraph::op::QuantizedMaxPool*>(node);
    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
    size_t qmax_pool_index = this->build_pooling_forward(mkldnn::algorithm::pooling_max,
                                                         input_desc,
                                                         result_desc,
                                                         qmax_pool->get_window_movement_strides(),
                                                         qmax_pool->get_window_shape(),
                                                         qmax_pool->get_padding_below(),
                                                         qmax_pool->get_padding_above());
    auto min_const_op = std::static_pointer_cast<ngraph::op::Constant>(qmax_pool->get_argument(1));
    auto max_const_op = std::static_pointer_cast<ngraph::op::Constant>(qmax_pool->get_argument(2));
    float min = *(static_cast<float const*>(min_const_op->get_data_ptr()));
    float max = *(static_cast<float const*>(max_const_op->get_data_ptr()));
    quant_util.push_back(min);
    quant_util.push_back(max);
    quant_util.push_back(qmax_pool_index);
}

void MKLDNNEmitter::build_quantized_avg_pool(const ngraph::Node* node,
                                             std::vector<float>& quant_util)
{
    auto qavg_pool = static_cast<const ngraph::op::QuantizedAvgPool*>(node);
    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
    size_t qavg_pool_index =
        this->build_pooling_forward((qavg_pool->get_include_padding_in_avg_computation()
                                         ? mkldnn::algorithm::pooling_avg_include_padding
                                         : mkldnn::algorithm::pooling_avg_exclude_padding),
                                    input_desc,
                                    result_desc,
                                    qavg_pool->get_window_movement_strides(),
                                    qavg_pool->get_window_shape(),
                                    qavg_pool->get_padding_below(),
                                    qavg_pool->get_padding_above());
    auto min_const_op = std::static_pointer_cast<ngraph::op::Constant>(qavg_pool->get_argument(1));
    auto max_const_op = std::static_pointer_cast<ngraph::op::Constant>(qavg_pool->get_argument(2));
    float min = *(static_cast<float const*>(min_const_op->get_data_ptr()));
    float max = *(static_cast<float const*>(max_const_op->get_data_ptr()));
    quant_util.push_back(min);
    quant_util.push_back(max);
    quant_util.push_back(qavg_pool_index);
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

    mkldnn::engine cpu_engine(mkldnn::engine::cpu, 0);
    mkldnn::convolution_forward::desc conv_desc_layout(
        mkldnn::prop_kind::forward,
        mkldnn::algorithm::convolution_direct,
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

size_t MKLDNNEmitter::build_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                                const mkldnn::memory::desc& weights_desc,
                                                const mkldnn::memory::desc& result_desc,
                                                const ngraph::Strides& strides,
                                                const ngraph::Strides& dilation_strides,
                                                const ngraph::CoordinateDiff& padding_below,
                                                const ngraph::CoordinateDiff& padding_above,
                                                const mkldnn::post_ops& pops)
{
    size_t input_data_index = build_memory_primitive(input_data_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t result_index = build_memory_primitive(result_desc);

    mkldnn::primitive_attr conv_attr;
    conv_attr.set_post_ops(pops);

    size_t conv_index = 0;
    try
    {
        auto conv_prim = new mkldnn::convolution_forward(
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
             conv_attr,
             mkldnn_utils::global_cpu_engine},
            *m_mkldnn_primitives[input_data_index],
            *m_mkldnn_primitives[weights_index],
            *m_mkldnn_primitives[result_index]);

        conv_index = insert_primitive(conv_prim);

        m_primitive_deps[conv_index] = {input_data_index, weights_index, result_index};
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not create mkldnn convolution " + e.message);
    }
    return conv_index;
}

size_t MKLDNNEmitter::build_quantized_convolution(const mkldnn::memory::desc& input_data_desc,
                                                  const mkldnn::memory::desc& weights_desc,
                                                  const mkldnn::memory::desc& result_desc,
                                                  const ngraph::Strides& strides,
                                                  const ngraph::Strides& dilation_strides,
                                                  const ngraph::CoordinateDiff& padding_below,
                                                  const ngraph::CoordinateDiff& padding_above,
                                                  const float scale)
{
    size_t input_data_index = build_memory_primitive(input_data_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t result_index = build_memory_primitive(result_desc);
    std::vector<float> output_scale;
    output_scale.push_back(scale);
    mkldnn::primitive_attr conv_attr;
    /* Specify the rounding mode */
    conv_attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
    /* Specify the scales array and corresponding mask */
    conv_attr.set_output_scales(0, output_scale);
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
         conv_attr,
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
                                                const ngraph::CoordinateDiff& padding_above,
                                                const mkldnn::post_ops& pops)
{
    const size_t input_data_index = build_memory_primitive(input_data_desc);
    const size_t weights_index = build_memory_primitive(weights_desc);
    const size_t bias_index = build_memory_primitive(bias_desc);
    const size_t result_index = build_memory_primitive(result_desc);

    mkldnn::primitive_attr conv_attr;
    conv_attr.set_post_ops(pops);

    size_t conv_index = -1;
    try
    {
        conv_index = insert_primitive(new mkldnn::convolution_forward(
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
             conv_attr,
             mkldnn_utils::global_cpu_engine},
            *m_mkldnn_primitives[input_data_index],
            *m_mkldnn_primitives[weights_index],
            *m_mkldnn_primitives[bias_index],
            *m_mkldnn_primitives[result_index]));

        m_primitive_deps[conv_index] = {input_data_index, weights_index, bias_index, result_index};
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not create convolution " + e.message);
    }
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

size_t MKLDNNEmitter::build_max_pooling_with_indices_forward(mkldnn::algorithm pooling_algorithm,
                                                             const mkldnn::memory::desc& src_desc,
                                                             const mkldnn::memory::desc& dst_desc,
                                                             const ngraph::Strides& window_strides,
                                                             const ngraph::Shape& window_shape,
                                                             const ngraph::Shape& padding_below,
                                                             const ngraph::Shape& padding_above)
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
        mkldnn_utils::global_cpu_engine};

    auto ws_index = build_memory_primitive(fwd_pd.workspace_primitive_desc().desc());

    size_t fwd_primitive_index =
        insert_primitive(new mkldnn::pooling_forward(fwd_pd,
                                                     *m_mkldnn_primitives[src_index],
                                                     *m_mkldnn_primitives[dst_index],
                                                     *m_mkldnn_primitives[ws_index]));

    m_primitive_deps[fwd_primitive_index] = {src_index, dst_index, ws_index};
    return fwd_primitive_index;
}

size_t MKLDNNEmitter::build_max_pooling_with_indices_backward(
    mkldnn::algorithm pooling_algorithm,
    const mkldnn::memory::desc& diff_dst_desc,
    const mkldnn::memory::desc& diff_src_desc,
    const ngraph::Strides& window_strides,
    const ngraph::Shape& window_shape,
    const ngraph::Shape& padding_below,
    const ngraph::Shape& padding_above)
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
        mkldnn_utils::global_cpu_engine};

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
         mkldnn_utils::global_cpu_engine,
         fwd_pd},
        *m_mkldnn_primitives[diff_dst_index],
        *m_mkldnn_primitives[fprop_ws_index],
        *m_mkldnn_primitives[diff_src_index]));

    m_primitive_deps[bwd_primitive_index] = {diff_dst_index, fprop_ws_index, diff_src_index};
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

size_t MKLDNNEmitter::build_quantize_reorder(const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& result_desc,
                                             const std::vector<float>& scales)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    mkldnn::primitive_attr attr;
    attr.set_output_scales(0, scales);
    attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);

    auto reorder_desc =
        mkldnn::reorder::primitive_desc({input_desc, mkldnn_utils::global_cpu_engine},
                                        {result_desc, mkldnn_utils::global_cpu_engine},
                                        attr);

    size_t primitive_index = insert_primitive(new mkldnn::reorder(
        reorder_desc, *m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]));
    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
}

size_t MKLDNNEmitter::build_lrn_forward(const mkldnn::memory::desc& input_desc,
                                        const mkldnn::memory::desc& result_desc,
                                        float alpha,
                                        float beta,
                                        float bias,
                                        int nsize)
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
    auto lrn_prim_desc =
        mkldnn::lrn_forward::primitive_desc(lrn_desc, mkldnn_utils::global_cpu_engine);

    size_t primitive_index = insert_primitive(new mkldnn::lrn_forward(
        lrn_prim_desc, *m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]));

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
                                              const double eps,
                                              bool use_global_stats,
                                              bool bn_training_flag,
                                              const mkldnn::post_ops& pops)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t weights_index = build_memory_primitive(weights_desc);
    size_t result_index = build_memory_primitive(result_desc);
    size_t mean_index = build_memory_primitive(mean_desc);
    size_t variance_index = build_memory_primitive(variance_desc);

    mkldnn::primitive_attr bn_attr;
    bn_attr.set_post_ops(pops);

    if (bn_training_flag && !use_global_stats)
    {
        size_t batchnorm_index = insert_primitive(new mkldnn::batch_normalization_forward(
            {{mkldnn::prop_kind::forward_training,
              input_desc,
              eps,
              mkldnn::batch_normalization_flag::use_scale_shift},
             bn_attr,
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
    else
    {
        size_t batchnorm_index = insert_primitive(new mkldnn::batch_normalization_forward(
            {{mkldnn::prop_kind::forward_training,
              input_desc,
              eps,
              mkldnn::batch_normalization_flag::use_scale_shift |
                  mkldnn::batch_normalization_flag::use_global_stats},
             bn_attr,
             mkldnn_utils::global_cpu_engine},
            mkldnn::primitive::at(*m_mkldnn_primitives[input_index]),
            mkldnn::primitive::at(*m_mkldnn_primitives[mean_index]),
            mkldnn::primitive::at(*m_mkldnn_primitives[variance_index]),
            mkldnn::primitive::at(*m_mkldnn_primitives[weights_index]),
            static_cast<mkldnn::memory>(*m_mkldnn_primitives[result_index])));

        m_primitive_deps[batchnorm_index] = {
            input_index, mean_index, variance_index, weights_index, result_index};
        return batchnorm_index;
    }
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

size_t MKLDNNEmitter::build_rnn_forward(const mkldnn::memory::desc& src_layer_desc,
                                        const mkldnn::memory::desc& src_iter_desc,
                                        const mkldnn::memory::desc& weights_layer_desc,
                                        const mkldnn::memory::desc& weights_iter_desc,
                                        const mkldnn::memory::desc& bias_desc,
                                        const mkldnn::memory::desc& dst_layer_desc,
                                        const mkldnn::memory::desc& dst_iter_desc)
{
    size_t src_layer_index = build_memory_primitive(src_layer_desc);
    size_t src_iter_index = build_memory_primitive(src_iter_desc);
    size_t weights_layer_index = build_memory_primitive(weights_layer_desc);
    size_t weights_iter_index = build_memory_primitive(weights_iter_desc);
    size_t bias_index = build_memory_primitive(bias_desc);
    size_t dst_layer_index = build_memory_primitive(dst_layer_desc);
    size_t dst_iter_index = build_memory_primitive(dst_iter_desc);

    mkldnn::rnn_cell::desc rnn_cell(mkldnn::algorithm::vanilla_lstm);
    mkldnn::rnn_forward::desc rnn_layer_desc(mkldnn::prop_kind::forward_training,
                                             rnn_cell,
                                             mkldnn::rnn_direction::unidirectional_left2right,
                                             src_layer_desc,
                                             src_iter_desc,
                                             weights_layer_desc,
                                             weights_iter_desc,
                                             bias_desc,
                                             dst_layer_desc,
                                             dst_iter_desc);
    auto rnn_layer_prim_desc =
        mkldnn::rnn_forward::primitive_desc(rnn_layer_desc, mkldnn_utils::global_cpu_engine);
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
    m_primitive_deps[rnn_index] = {src_layer_index,
                                   src_iter_index,
                                   weights_layer_index,
                                   weights_iter_index,
                                   bias_index,
                                   dst_layer_index,
                                   dst_iter_index,
                                   workspace_index,
                                   workspace_buf_index};

    return rnn_index;
}

size_t MKLDNNEmitter::build_concat(const std::vector<mkldnn::memory::desc>& inputs_data_desc,
                                   const mkldnn::memory::desc& result_desc,
                                   const size_t concat_dim)
{
    std::vector<mkldnn::memory::primitive::at> inputs_primitive;
    std::vector<size_t> inputs_data_index;
    std::vector<size_t> in_out_index;
    std::vector<mkldnn::memory::primitive_desc> inputs_pd;

    for (size_t i = 0; i < inputs_data_desc.size(); i++)
    {
        inputs_pd.push_back(mkldnn::memory::primitive_desc(
            inputs_data_desc[i], runtime::cpu::mkldnn_utils::global_cpu_engine));
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
    m_primitive_deps[concat_index] = in_out_index;
    return concat_index;
}

size_t MKLDNNEmitter::build_slice(const mkldnn::memory::desc& input_desc,
                                  const mkldnn::memory::desc& result_desc,
                                  const ngraph::Coordinate& lower_bounds,
                                  const ngraph::Shape& result_shape)
{
    std::vector<size_t> in_out_index;
    mkldnn::memory::primitive_desc input_pd =
        mkldnn::memory::primitive_desc(input_desc, runtime::cpu::mkldnn_utils::global_cpu_engine);
    size_t input_index = build_memory_primitive(input_desc);

    auto dims = mkldnn::memory::dims(result_shape.begin(), result_shape.end());
    auto offsets = mkldnn::memory::dims(lower_bounds.begin(), lower_bounds.end());
    auto view_pd = mkldnn::view::primitive_desc(input_pd, dims, offsets).dst_primitive_desc();

    mkldnn::memory::primitive_desc result_pd =
        mkldnn::memory::primitive_desc(result_desc, runtime::cpu::mkldnn_utils::global_cpu_engine);
    size_t result_index = build_memory_primitive(result_desc);

    // reorder primitive descriptor
    mkldnn::reorder::primitive_desc reorder_pd =
        mkldnn::reorder::primitive_desc(view_pd, result_pd);
    // reorder primitive
    size_t reorder_index = insert_primitive(new mkldnn::reorder(
        reorder_pd, *m_mkldnn_primitives[input_index], *m_mkldnn_primitives[result_index]));

    in_out_index.push_back(input_index);
    in_out_index.push_back(result_index);
    m_primitive_deps[reorder_index] = in_out_index;
    return reorder_index;
}

size_t MKLDNNEmitter::build_softmax_forward(const mkldnn::memory::desc& input_desc,
                                            const mkldnn::memory::desc& result_desc,
                                            int softmax_axis)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index = insert_primitive(
        new mkldnn::softmax_forward({{mkldnn::prop_kind::forward_scoring, input_desc, softmax_axis},
                                     mkldnn_utils::global_cpu_engine},
                                    *m_mkldnn_primitives[input_index],
                                    *m_mkldnn_primitives[result_index]));

    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
}

size_t MKLDNNEmitter::build_bounded_relu(const mkldnn::memory::desc& input_desc,
                                         const mkldnn::memory::desc& result_desc,
                                         float alpha)
{
    size_t input_index = build_memory_primitive(input_desc);
    size_t result_index = build_memory_primitive(result_desc);

    size_t primitive_index =
        insert_primitive(new mkldnn::eltwise_forward({{mkldnn::prop_kind::forward_training,
                                                       mkldnn::algorithm::eltwise_bounded_relu,
                                                       input_desc,
                                                       alpha,
                                                       0.0f},
                                                      mkldnn_utils::global_cpu_engine},
                                                     *m_mkldnn_primitives[input_index],
                                                     *m_mkldnn_primitives[result_index]));

    m_primitive_deps[primitive_index] = {input_index, result_index};
    return primitive_index;
}
