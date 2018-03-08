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

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <mkldnn.hpp>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_ExternalFunction;
            class TensorViewWrapper;

            class MKLDNNEmitter
            {
            public:
                MKLDNNEmitter() {}
                const std::vector<mkldnn::primitive*>& get_mkldnn_primitives() const;

                size_t insert_primitive(mkldnn::primitive* primitive);
                const std::vector<size_t>& get_primitive_deps(size_t index) const;

                // TODO(jmenon): Get rid of TensorViewWrappers at some point
                mkldnn::memory::desc build_memory_descriptor(const TensorViewWrapper& tvw,
                                                             mkldnn::memory::format fmt) const;
                mkldnn::memory::desc build_memory_descriptor(const TensorViewWrapper& tvw) const;
                mkldnn::memory build_memory_primitive(const TensorViewWrapper& tvw) const;
                size_t build_memory_primitive(const mkldnn::memory::desc& desc);

                size_t build_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                                 const mkldnn::memory::desc& weights_desc,
                                                 const mkldnn::memory::desc& result_desc,
                                                 const ngraph::Strides& strides,
                                                 const ngraph::CoordinateDiff& padding_below,
                                                 const ngraph::CoordinateDiff& padding_above);

                size_t build_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                                 const mkldnn::memory::desc& weights_desc,
                                                 const mkldnn::memory::desc& result_desc,
                                                 const ngraph::Strides& strides,
                                                 const ngraph::Strides& dilation_strides,
                                                 const ngraph::CoordinateDiff& padding_below,
                                                 const ngraph::CoordinateDiff& padding_above);

                size_t
                    build_convolution_backward_weights(const mkldnn::memory::desc& input_desc,
                                                       const mkldnn::memory::desc& delta_desc,
                                                       const mkldnn::memory::desc& result_desc,
                                                       const ngraph::Strides& strides,
                                                       const ngraph::Strides& dilation_strides,
                                                       const ngraph::CoordinateDiff& padding_below,
                                                       const ngraph::CoordinateDiff& padding_above);

                size_t build_convolution_backward_data(const mkldnn::memory::desc& weights_desc,
                                                       const mkldnn::memory::desc& delta_desc,
                                                       const mkldnn::memory::desc& result_desc,
                                                       const ngraph::Strides& strides,
                                                       const ngraph::Strides& dilation_strides,
                                                       const ngraph::CoordinateDiff& padding_below,
                                                       const ngraph::CoordinateDiff& padding_above);

                size_t build_pooling_forward(mkldnn::algorithm pooling_algorithm,
                                             const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& result_desc,
                                             const ngraph::Strides& window_strides,
                                             const ngraph::Shape& window_shape,
                                             const ngraph::Shape& padding_below,
                                             const ngraph::Shape& padding_above);

                size_t build_reorder(const mkldnn::memory::desc& input_desc,
                                     const mkldnn::memory::desc& result_desc);

                size_t build_relu_forward(const mkldnn::memory::desc& input_desc,
                                          const mkldnn::memory::desc& result_desc);
                
                size_t build_sigmoid_forward(const mkldnn::memory::desc& input_desc,
                                          const mkldnn::memory::desc& result_desc);

                size_t build_elementwise_add(
                    const mkldnn::memory::desc& input0_data_desc,
                    const mkldnn::memory::desc& input1_data_desc,
                    const mkldnn::memory::desc& result_desc,
                    const std::vector<float>& scale_vector,
                    const std::vector<mkldnn::memory::primitive_desc>& input_pd);

            private:
                std::vector<mkldnn::primitive*> m_mkldnn_primitives;
                std::vector<mkldnn::stream> m_mkldnn_streams;
                std::unordered_map<size_t, std::vector<size_t>> m_primitive_deps;
            };
        }
    }
}
