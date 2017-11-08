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

#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/types/element_type.hpp"
#include "ngraph/types/type.hpp"

using namespace ngraph::descriptor::layout;

TensorViewLayout::TensorViewLayout(const ngraph::descriptor::TensorView& tensor_view)
    : m_tensor_view_type(tensor_view.get_tensor_view_type())
{
}

const ngraph::element::Type& TensorViewLayout::get_element_type() const
{
    return m_tensor_view_type->get_element_type();
}
const ngraph::Shape& TensorViewLayout::get_shape() const
{
    return m_tensor_view_type->get_shape();
}
