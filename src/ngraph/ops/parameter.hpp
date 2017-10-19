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

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/types/type.hpp"

namespace ngraph
{
    class Function;
    namespace op
    {
        ///
        /// Parameters are nodes that represent the arguments that will be passed to user-defined functions.
        /// Function creation requires a sequence of parameters.
        /// Basic graph operations do not need parameters attached to a function.
        ///
        class Parameter : public Node
        {
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override
            {
            }

        public:
            Parameter(const std::shared_ptr<const ValueType>& value_type = nullptr);
            Parameter(const ngraph::element::Type& element_type, const Shape& shape);

            std::string description() const override { return "Parameter"; }
            virtual void propagate_types() override;
        };
    }
}
