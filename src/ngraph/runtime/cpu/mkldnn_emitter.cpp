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

#include "mkldnn_emitter.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace ngraph::runtime::cpu;

mkldnn::memory::desc MKLDNNEmitter::build_memory_descriptor(const TensorViewWrapper& tvw)
{
    return mkldnn::memory::desc(mkldnn::memory::dims(tvw.get_shape().begin(), tvw.get_shape().end()),
                                mkldnn_utils::GetDataType(tvw.get_element_type()),
                                mkldnn::memory::format::nchw);
}

