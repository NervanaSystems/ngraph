// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>

#include "mkldnn_emitter.hpp"

#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace ngraph::runtime::cpu;

mkldnn::memory::desc MKLDNNEmitter::build_memory_descriptor(const TensorViewWrapper& tvw, mkldnn::memory::format fmt)
{
    return mkldnn::memory::desc(mkldnn::memory::dims(tvw.get_shape().begin(), tvw.get_shape().end()),
                                mkldnn_utils::GetDataType(tvw.get_element_type()),
                                fmt);
}

mkldnn::memory::desc MKLDNNEmitter::build_memory_descriptor(const TensorViewWrapper& tvw)
{
    auto layout = std::static_pointer_cast<LayoutDescriptor>(tvw.get_tensor_view()->get_tensor_view_layout());

    return build_memory_descriptor(tvw, layout->get_mkldnn_format());
}

mkldnn::memory MKLDNNEmitter::build_memory_primitive(const TensorViewWrapper& tvw)
{
    return mkldnn::memory({build_memory_descriptor(tvw), mkldnn_utils::global_cpu_engine}, nullptr);
}

mkldnn::memory MKLDNNEmitter::build_memory_primitive(const mkldnn::memory::desc& desc)
{
    return mkldnn::memory({desc, mkldnn_utils::global_cpu_engine}, nullptr);
}

size_t MKLDNNEmitter::build_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                                                     const mkldnn::memory::desc& weights_desc,
                                                                     const mkldnn::memory::desc& result_desc,
                                                                     const ngraph::Strides& strides,
                                                                     const ngraph::CoordinateDiff& padding_below,
                                                                     const ngraph::CoordinateDiff& padding_above)

{
    mkldnn_primitives.push_back(mkldnn::convolution_forward(
    {
        {
            mkldnn::prop_kind::forward, mkldnn::algorithm::convolution_direct,
            input_data_desc, weights_desc, result_desc,
            mkldnn::memory::dims(strides.begin(), strides.end()),
            mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
            mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
            mkldnn::padding_kind::zero
        },
        mkldnn_utils::global_cpu_engine
    },
    build_memory_primitive(input_data_desc),
    build_memory_primitive(weights_desc),
    build_memory_primitive(result_desc)));

    return (mkldnn_primitives.size() - 1);
}
