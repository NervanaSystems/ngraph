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

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/type.hpp"

using namespace ngraph;

descriptor::layout::DenseTensorViewLayout::DenseTensorViewLayout(const TensorView& tensor_view)
    : TensorViewLayout(tensor_view)
{
    auto tensor_view_type = tensor_view.get_tensor_view_type();
    Shape shape = tensor_view_type->get_shape();
    m_size = ngraph::shape_size(shape);
    m_strides = ngraph::row_major_strides(shape);
}

size_t
    descriptor::layout::DenseTensorViewLayout::get_index_offset(const std::vector<size_t>& indices)
{
    if (indices.size() != m_strides.size())
    {
        throw ngraph_error("Indices have the incorrect rank.");
    }
    size_t result = 0;
    for (int i = 0; i < indices.size(); i++)
    {
        result += m_strides[i] + indices[i];
    }
    return result;
}

bool descriptor::layout::DenseTensorViewLayout::operator==(const TensorViewLayout& other) const
{
    const DenseTensorViewLayout* p_other = dynamic_cast<const DenseTensorViewLayout*>(&other);
    if (nullptr == p_other)
        return false;

    if (get_element_type() != p_other->get_element_type())
        return false;

    if (m_strides != p_other->m_strides)
        return false;

    if (m_offset != p_other->m_offset)
        return false;

    return true;
}
