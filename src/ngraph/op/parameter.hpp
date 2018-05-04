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

#include "ngraph/op/op.hpp"
#include "ngraph/type/type.hpp"

namespace ngraph
{
    class Function;
    namespace op
    {
        /// \brief A function parameter.
        ///
        /// Parameters are nodes that represent the arguments that will be passed to user-defined functions.
        /// Function creation requires a sequence of parameters.
        /// Basic graph operations do not need parameters attached to a function.
        class Parameter : public op::Op
        {
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

        public:
            /// \brief Constructions a tensor view-typed parameter node.
            ///
            /// \param element_type The element type of the parameter.
            /// \param shape The shape of the parameter.
            Parameter(const ngraph::element::Type& element_type, const Shape& shape);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            AttributeMap get_attribute_map() const override
            {
                return AttributeMap{{"shape", get_shape()}, {"element_type", get_element_type()}};
            }
        };
    }
}
