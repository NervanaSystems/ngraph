/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

runtime::intelgpu::IntelGPULayout::IntelGPULayout(const descriptor::TensorView& tv,
                                                  const cldnn::layout& layout)
    : TensorViewLayout(tv)
    , cldnn_layout(layout)
{
}

size_t runtime::intelgpu::IntelGPULayout::get_index_offset(const std::vector<size_t>& indices)
{
    if (indices.size() != strides.size())
    {
        throw ngraph_error("Indices have incorrect rank");
    }
    size_t result = 0;
    for (int i = 0; i < indices.size(); i++)
    {
        result += strides[i] + indices[i];
    }
    return result;
}

bool runtime::intelgpu::IntelGPULayout::
    operator==(const descriptor::layout::TensorViewLayout& other) const
{
    const IntelGPULayout* p_other = dynamic_cast<const IntelGPULayout*>(&other);
    if (!p_other)
    {
        return false;
    }

    return (cldnn_layout == p_other->cldnn_layout);
}

cldnn::data_types
    runtime::intelgpu::IntelGPULayout::get_cldnn_type(const element::Type& element_type)
{
    if (element_type == ngraph::element::i8)
    {
        return cldnn::data_types::i8;
    }
    else if (element_type == ngraph::element::u8)
    {
        return cldnn::data_types::u8;
    }
    else if (element_type == ngraph::element::f32)
    {
        return cldnn::data_types::f32;
    }
    else
    {
        ostringstream os;
        os << "IntelGPUTensorView::get_cldnn_type: Unknown type " << element_type;
        throw std::invalid_argument(os.str());
    }
}

cldnn::layout runtime::intelgpu::IntelGPULayout::create_cldnn_layout(
    const ngraph::element::Type& element_type, const Shape& element_shape)
{
    const size_t mem_size = shape_size(element_shape);
    const cldnn::data_types data_type = get_cldnn_type(element_type);
    const cldnn::tensor tensor(1, mem_size, 1, 1);
    const cldnn::format::type format = cldnn::format::yxfb;

    return cldnn::layout(data_type, format, tensor);
}
