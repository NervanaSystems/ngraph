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

size_t runtime::intelgpu::IntelGPULayout::get_index_offset(const vector<size_t>& indices)
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
    if ((element_type == ngraph::element::i8) || (element_type == ngraph::element::boolean))
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
        os << "IntelGPULayout::get_cldnn_type: Unknown type " << element_type;
        throw invalid_argument(os.str());
    }
}

cldnn::tensor runtime::intelgpu::IntelGPULayout::create_cldnn_tensor(const Shape& element_shape)
{
    vector<size_t> idx(4, 1);
    size_t index = 0;
    const size_t total_size = shape_size<Shape>(element_shape);

    // clDNN requires at least scalar tensor size. We can't create zero sized tensors
    if (total_size != 0)
    {
        for (auto i = element_shape.crbegin(); i != element_shape.crend() && index < 3;
             ++i, ++index)
        {
            idx.at(index) = *i;
        }

        if (element_shape.size() > 3)
        {
            idx.at(3) = accumulate(
                element_shape.rbegin() + 3, element_shape.rend(), 1, multiplies<size_t>());
        }
    }

    // Parameters for this ctor: batch, feature, spatial_x, spatial_y
    const cldnn::tensor tns(idx.at(3), idx.at(2), idx.at(0), idx.at(1));

    return tns;
}

cldnn::tensor runtime::intelgpu::IntelGPULayout::create_cldnn_offset(const Shape& pad_below)
{
    vector<cldnn::tensor::value_type> offset({0, 0, 0, 0});
    size_t ridx = 4;

    for (auto i = pad_below.crbegin(); i != pad_below.crend() && ridx > 0; ++i, --ridx)
    {
        offset.at(ridx - 1) = -(*i);
    }

    const cldnn::tensor input_offset(offset.at(0), offset.at(1), offset.at(3), offset.at(2));
    return input_offset;
}

cldnn::layout runtime::intelgpu::IntelGPULayout::create_cldnn_layout(
    const ngraph::element::Type& element_type, const Shape& element_shape)
{
    const cldnn::data_types data_type = get_cldnn_type(element_type);
    const cldnn::format::type format = cldnn::format::bfyx;
    const cldnn::tensor tensor = create_cldnn_tensor(element_shape);

    return cldnn::layout(data_type, format, tensor);
}

cldnn::concatenation::concatenation_axis
    runtime::intelgpu::IntelGPULayout::get_cldnn_axis(size_t shape_size, size_t axis)
{
    const size_t t_channel = shape_size - axis - 1;

    switch (t_channel)
    {
    case 0: return cldnn::concatenation::along_x;
    case 1: return cldnn::concatenation::along_y;
    case 2: return cldnn::concatenation::along_f;
    case 3:
        if (shape_size < 5)
        {
            return cldnn::concatenation::along_b;
        }
    /* no break */
    default:
        throw invalid_argument("IntelGPULayout::get_cldnn_axis: wrong tensor channel " +
                               to_string(t_channel));
    }
}
