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

#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/type.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<const descriptor::TensorView> runtime::TensorView::get_tensor_view_descriptor() const
{
    return m_descriptor;
}

shared_ptr<descriptor::TensorView> runtime::TensorView::get_descriptor() const
{
    return m_descriptor;
}

const Shape& runtime::TensorView::get_shape() const
{
    return m_descriptor->get_tensor_view_type()->get_shape();
}

const Strides& runtime::TensorView::get_strides() const
{
    return m_descriptor->get_tensor_view_layout()->get_strides();
}

shared_ptr<descriptor::layout::TensorViewLayout> runtime::TensorView::get_tensor_view_layout() const
{
    return m_descriptor->get_tensor_view_layout();
}

size_t runtime::TensorView::get_element_count() const
{
    size_t rc = 1;
    for (size_t s : get_shape())
    {
        rc *= s;
    }
    return rc;
}

const descriptor::Tensor& runtime::TensorView::get_tensor() const
{
    return get_tensor_view_descriptor()->get_tensor();
}
