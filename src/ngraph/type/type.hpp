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

#pragma once

#include <memory>
#include <vector>

#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    class TensorViewType;

    std::ostream& operator<<(std::ostream&, const TensorViewType&);

    /// Describes a tensor view; an element type and a shape.
    class TensorViewType : public std::enable_shared_from_this<TensorViewType>
    {
    public:
        /// /param element_type The type of the tensor elements.
        /// /param shape The shape of the tensor.
        TensorViewType(const element::Type& element_type, const Shape& shape)
            : m_element_type(element_type)
            , m_shape(shape)
        {
        }

        const element::Type& get_element_type() const { return m_element_type; }
        const Shape& get_shape() const { return m_shape; }
        bool operator==(const TensorViewType& that) const;
        bool operator!=(const TensorViewType& that) const;

        friend std::ostream& operator<<(std::ostream&, const TensorViewType&);

    protected:
        const element::Type m_element_type;
        Shape m_shape;
    };
}
