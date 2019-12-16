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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    class Function;
    namespace op
    {
        /// \brief A function parameter.
        ///
        /// Parameters are nodes that represent the arguments that will be passed to user-defined
        /// functions. Function creation requires a sequence of parameters. Basic graph operations
        /// do not need parameters attached to a function.
        class NGRAPH_API Parameter : public op::Op
        {
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const OutputVector& deltas) override;

        public:
            static constexpr NodeTypeInfo type_info{"Parameter", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            /// \brief Constructions a tensor-typed parameter node.
            Parameter() = default;
            /// \brief Constructions a tensor-typed parameter node.
            ///
            /// \param element_type The element type of the parameter.
            /// \param pshape The partial shape of the parameter.
            /// \param cacheable True if the parameter is not expected to be frequently updated.
            Parameter(const ngraph::element::Type& element_type,
                      const PartialShape& pshape,
                      const bool cacheable = false);

            bool visit_attributes(AttributeVisitor& visitor) override;

            bool is_parameter() const override { return true; }
            void validate_and_infer_types() override;

            bool get_cacheable() const { return m_cacheable; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            bool is_relevant_to_shapes() const;
            void set_is_relevant_to_shapes(bool is_relevant);

            const PartialShape& get_partial_shape() const { return m_partial_shape; }
            PartialShape& get_partial_shape() { return m_partial_shape; }
            void set_partial_shape(const PartialShape& partial_shape)
            {
                m_partial_shape = partial_shape;
            }

            const element::Type& get_element_type() const { return m_element_type; }
            void set_element_type(const element::Type& element_type)
            {
                m_element_type = element_type;
            }

        protected:
            bool m_cacheable;
            PartialShape m_partial_shape;
            element::Type m_element_type;
            bool m_is_relevant_to_shapes;
        };
    }
    using ParameterVector = std::vector<std::shared_ptr<op::Parameter>>;
}
