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

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/node.hpp"

using namespace ngraph;
using namespace std;

descriptor::Tensor::Tensor(const element::Type& element_type,
                           PrimaryTensorView* primary_tensor_view,
                           const string& name)
    : m_element_type(element_type)
    , m_primary_tensor_view(primary_tensor_view)
    , m_name{name}
    , m_next_view_id{0}
{
}

string descriptor::Tensor::make_tensor_name(const Node* node, size_t value_index)
{
    return node->get_name() + "_" + to_string(value_index);
}

string descriptor::Tensor::get_next_view_name()
{
    return m_name + "_TV" + to_string(m_next_view_id++);
}

void descriptor::Tensor::set_pool_offset(size_t offset)
{
    m_pool_offset = offset;
}

size_t descriptor::Tensor::get_pool_offset() const
{
    return m_pool_offset;
}

size_t descriptor::Tensor::size() const
{
    if (auto tvl = m_primary_tensor_view->get_tensor_view_layout())
    {
        return tvl->get_allocated_size();
    }
    else
    {
        auto tvt = m_primary_tensor_view->get_tensor_view_type();
        return shape_size(tvt->get_shape()) * m_element_type.size();
    }
}

void descriptor::Tensor::set_element_type(const element::Type& element_type)
{
    m_element_type = element_type;
}

ostream& operator<<(ostream& out, const descriptor::Tensor& tensor)
{
    out << "Tensor(" << tensor.get_name() << ")";
    return out;
}
