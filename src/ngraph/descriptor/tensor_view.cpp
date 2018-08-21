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

#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/type/type.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<const ngraph::TensorViewType> descriptor::TensorView::get_value_type() const
{
    return m_tensor_view_type;
}

const element::Type& descriptor::TensorView::get_element_type() const
{
    return m_tensor_view_type->get_element_type();
}

const Shape& descriptor::TensorView::get_shape() const
{
    return m_tensor_view_type->get_shape();
}

void descriptor::TensorView::set_tensor_view_type(const element::Type& element_type,
                                                  const Shape& shape)
{
    m_tensor_view_type = make_shared<ngraph::TensorViewType>(element_type, shape);
    if (nullptr != m_tensor_view_layout)
    {
        m_tensor_view_layout->set_tensor_view_type(element_type, shape);
    }
}
