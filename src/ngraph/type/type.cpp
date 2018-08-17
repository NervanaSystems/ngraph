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

#include <memory>

#include "ngraph/type/type.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

bool TensorViewType::operator!=(const TensorViewType& that) const
{
    return !(*this == that);
}

bool TensorViewType::operator==(const TensorViewType& that) const
{
    bool rc = true;
    auto that_tvt = dynamic_cast<const TensorViewType*>(&that);
    if (that_tvt != nullptr)
    {
        rc = true;
        if (that_tvt->get_element_type() != m_element_type)
        {
            rc = false;
        }
        if (that_tvt->get_shape() != m_shape)
        {
            rc = false;
        }
    }
    return rc;
}

std::ostream& ngraph::operator<<(std::ostream& out, const TensorViewType& obj)
{
    out << "TensorViewType(" << obj.m_element_type << ", {" << join(obj.m_shape) << "})";
    return out;
}
