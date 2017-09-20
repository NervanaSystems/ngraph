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

#include "descriptor/tensor.hpp"
#include "node.hpp"

namespace ngraph
{
    namespace descriptor
    {
        Tensor::Tensor(const element::Type& element_type,
                       PrimaryTensorView*   primary_tensor_view,
                       const Node*          parent,
                       size_t               value_index)
            : m_element_type(element_type)
            , m_primary_tensor_view(primary_tensor_view)
            , m_is_output{false}
            , m_is_input{parent->is_parameter()}
            , m_is_persistent{false}
            , m_name{parent->get_node_id() + "_" + std::to_string(value_index)}
            , m_next_view_id{0}
        {
        }

        std::string Tensor::get_next_view_name()
        {
            return m_name + "_TV" + std::to_string(m_next_view_id++);
        }

        std::ostream& operator<<(std::ostream& out, const Tensor& tensor)
        {
            out << "Tensor(" << tensor.get_name() << ")";
            return out;
        }
    }
}
