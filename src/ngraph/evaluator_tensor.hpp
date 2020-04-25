//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#pragma once

#include <memory>
#include <vector>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"

namespace ngraph
{
    class Node;
    template <typename T>
    class Output;

    namespace runtime
    {
        class HostTensor;
    }

    /// \brief A generic handle to (potential) storage with element type and shape information
    class NGRAPH_API EvaluatorTensor
    {
    protected:
        EvaluatorTensor(const element::Type& element_type, const PartialShape& partial_shape);
        /// \brief Get type/shape from value
        EvaluatorTensor(const Output<Node>& value);

    public:
        virtual ~EvaluatorTensor();
        /// \return pointer to storage
        virtual void* get_data_ptr() = 0;

        const element::Type& get_element_type() const;
        const PartialShape& get_partial_shape() const;
        const Shape get_shape() const;
        size_t get_element_count();
        size_t get_size_in_bytes();

        template <element::Type_t ET>
        typename element_type_traits<ET>::value_type* get_data_ptr()
        {
            return static_cast<typename element_type_traits<ET>::value_type*>(get_data_ptr());
        }

    protected:
        element::Type m_element_type;
        PartialShape m_partial_shape;
    };

    // This is the name we want to use; a handle to the API implementation
    using EvaluatorTensorPtr = std::shared_ptr<EvaluatorTensor>;
    using EvaluatorTensorVector = std::vector<EvaluatorTensorPtr>;
}
