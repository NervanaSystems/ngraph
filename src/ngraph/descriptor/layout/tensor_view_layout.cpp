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

#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/type.hpp"

using namespace ngraph;

descriptor::layout::TensorViewLayout::TensorViewLayout(const descriptor::TensorView& tensor_view)
    : m_element_type(tensor_view.get_element_type())
    , m_shape(tensor_view.get_shape())
{
}

const element::Type& descriptor::layout::TensorViewLayout::get_element_type() const
{
    return m_element_type;
}

const Shape& descriptor::layout::TensorViewLayout::get_shape() const
{
    return m_shape;
}

void descriptor::layout::TensorViewLayout::set_tensor_view_type(const element::Type& element_type,
                                                                const Shape& shape)
{
    m_element_type = element_type;
    m_shape = shape;
}

size_t descriptor::layout::TensorViewLayout::get_allocated_size()
{
    return get_size() * get_element_type().size();
}
