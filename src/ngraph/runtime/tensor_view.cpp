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

#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;
using namespace std;

const Shape& runtime::TensorView::get_shape() const
{
    return m_descriptor->get_shape();
}

Strides runtime::TensorView::get_strides() const
{
    return m_descriptor->get_tensor_layout()->get_strides();
}

const element::Type& runtime::TensorView::get_element_type() const
{
    return m_descriptor->get_element_type();
}

shared_ptr<descriptor::layout::TensorLayout> runtime::TensorView::get_tensor_layout() const
{
    return m_descriptor->get_tensor_layout();
}

void runtime::TensorView::set_tensor_layout(
    const shared_ptr<descriptor::layout::TensorLayout>& layout)
{
    m_descriptor->set_tensor_layout(layout);
}

size_t runtime::TensorView::get_size() const
{
    return get_tensor_layout()->get_size();
}

const std::string& runtime::TensorView::get_name() const
{
    return m_descriptor->get_name();
}

bool runtime::TensorView::get_stale() const
{
    return m_stale;
}

void runtime::TensorView::set_stale(bool val)
{
    m_stale = val;
}
