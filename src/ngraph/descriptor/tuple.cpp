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

#include <memory>
#include <vector>

#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/descriptor/tuple.hpp"
#include "ngraph/types/type.hpp"

using namespace ngraph::descriptor;

Tuple::Tuple(const std::vector<std::shared_ptr<ngraph::descriptor::Value>>& elements)
    : m_elements(elements)
{
    std::vector<std::shared_ptr<const ngraph::ValueType>> types;
    for (auto element : m_elements)
    {
        types.push_back(element->get_value_type());
    }
    m_tuple_type = std::make_shared<ngraph::TupleType>(types);
}

void Tuple::collect_tensor_views(std::vector<std::shared_ptr<TensorView>>& views,
                                 const std::shared_ptr<Value>&             value) const
{
    for (auto element : m_elements)
    {
        element->collect_tensor_views(views, element);
    }
}
