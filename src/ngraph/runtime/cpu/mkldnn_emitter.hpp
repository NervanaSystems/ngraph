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

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <mkldnn.hpp>

#include "ngraph/common.hpp"

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
                MKLDNNEmitter(std::shared_ptr<CPU_ExternalFunction> ef)
                    : external_function(ef)
                {
                }

                // TODO(jmenon): Get rid of TensorViewWrappers at some point
                size_t insert_primitive(mkldnn::primitive* primitive);
                mkldnn::memory::desc build_memory_descriptor(const TensorViewWrapper& tvw, mkldnn::memory::format fmt);
                mkldnn::memory::desc build_memory_descriptor(const TensorViewWrapper& tvw);
                mkldnn::memory build_memory_primitive(const TensorViewWrapper& tvw);
                size_t build_memory_primitive(const mkldnn::memory::desc& desc);

                size_t build_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                                       const mkldnn::memory::desc& weights_desc,
                                                       const mkldnn::memory::desc& result_desc,
                                                       const ngraph::Strides& strides,
                                                       const ngraph::CoordinateDiff& padding_below,
                                                       const ngraph::CoordinateDiff& padding_above);
            private:
                std::shared_ptr<CPU_ExternalFunction> external_function;
                std::vector<mkldnn::primitive*> mkldnn_primitives;
                std::vector<mkldnn::stream> mkldnn_streams;
                std::unordered_map<size_t, std::vector<size_t>> primitive_deps;
            };
        }
    }
}
