// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
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

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/eigen/utils.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"

ngraph::runtime::TH2
    ngraph::runtime::eigen::get_tensor_header(const TensorViewInfo& tensor_view_info, bool flatten)
{
    const std::shared_ptr<ngraph::descriptor::TensorView>& descriptor = tensor_view_info.descriptor;
    const Shape& shape = descriptor->get_tensor_view_type()->get_shape();
    auto layout = std::dynamic_pointer_cast<ngraph::descriptor::layout::DenseTensorViewLayout>(
        descriptor->get_tensor_view_layout());
    if (nullptr == layout)
    {
        throw ngraph_error("Dense tensor view layout required");
    }
    if (flatten)
    {
        return TH2{tensor_view_info.index, layout->get_size(), 1, 1, 1};
    }
    const ngraph::Strides& strides = layout->get_strides();

    switch (strides.size())
    {
    case 1: return TH2{tensor_view_info.index, shape[0], strides[0], 1, 1};
    case 2: return TH2{tensor_view_info.index, shape[0], strides[0], shape[1], strides[1]};
    default: throw ngraph_error("Only 1 and 2d supported");
    }
}
