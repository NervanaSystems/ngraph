//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include <string>

#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    class Node;

    namespace descriptor
    {
        namespace layout
        {
            class TensorLayout;
        }

        /// \brief Compile-time descriptor of a first-class value that is a tensor.
        class NGRAPH_API Tensor
        {
            Tensor(const Tensor&) = delete;
            Tensor& operator=(const Tensor&) = delete;

        public:
            Tensor(const element::Type& element_type,
                   const PartialShape& pshape,
                   const std::string& name);
            Tensor(const element::Type& element_type,
                   const PartialShape& pshape,
                   Node* node,
                   size_t node_output_number);

            const std::string& get_name() const;
            void set_tensor_type(const element::Type& element_type, const PartialShape& pshape);

            const element::Type& get_element_type() const { return m_element_type; }
            const Shape& get_shape() const;
            const PartialShape& get_partial_shape() const { return m_partial_shape; }
            const std::shared_ptr<layout::TensorLayout>& get_tensor_layout() const
            {
                return m_tensor_layout;
            }

            void set_tensor_layout(const std::shared_ptr<layout::TensorLayout>& tensor_layout);

            void set_pool_offset(size_t);
            size_t get_pool_offset() const;

            size_t size() const;

        protected:
            element::Type m_element_type;

            // TODO(amprocte): For now we are maintaining both m_shape and m_partial_shape fields,
            //    with m_shape possibly being invalid (get_shape will throw an exception if it
            //    is). This is because get_shape() returns a const reference. I think ideally we
            //    should refactor so that get_shape returns by value.
            Shape m_shape;
            PartialShape m_partial_shape;
            Node* m_node{nullptr};
            size_t m_node_output_number{0};

            std::string m_name;
            std::shared_ptr<layout::TensorLayout> m_tensor_layout;
            size_t m_pool_offset{0};
        };

        std::ostream& operator<<(std::ostream&, const ngraph::descriptor::Tensor&);
    }
}
