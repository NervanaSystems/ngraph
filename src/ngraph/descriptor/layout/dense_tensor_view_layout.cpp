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

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph::descriptor::layout;
using ngraph::Shape;
using ngraph::descriptor::TensorView;
using ngraph::TensorViewType;

DenseTensorViewLayout::DenseTensorViewLayout(const TensorView& tensor_view)
    : TensorViewLayout(tensor_view)
{
    auto tensor_view_type = tensor_view.get_tensor_view_type();
    Shape shape = tensor_view_type->get_shape();
    m_size = ngraph::shape_size(shape);
    m_strides = ngraph::row_major_strides(shape);
}

size_t DenseTensorViewLayout::get_index_offset(const std::vector<size_t>& indices)
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
