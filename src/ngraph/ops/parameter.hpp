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

#include "../node.hpp"
#include "../type.hpp"

namespace ngraph
{
    class Function;

    /**
     ** Parameters are nodes that represent the arguments that will be passed to user-defined functions.
     ** Function creation requires a sequence of parameters.
     ** Basic graph operations do not need parameters attached to a function.
     **/
    class Parameter : public Node
    {
        friend class Function;

    protected:
        // Called by the Function constructor to associate this parameter with the function.
        // It is an error to try to associate a parameter with more than one function.
        void assign_function(Function* function, size_t index);

    public:
        Parameter(const ValueType::ptr& value_type);

        std::string         description() const override { return "Parameter"; }
        virtual void        propagate_types() override;
        virtual std::string node_id() const override;
        virtual bool is_parameter() const override { return true; };

    protected:
        Function* m_function;
        size_t    m_index;
    };

    namespace op
    {
        /// Factory for frameworks
        std::shared_ptr<ngraph::Parameter> parameter(const ValueType::ptr& value_type = nullptr);
        /// Convenience factory for tests
        std::shared_ptr<ngraph::Parameter> parameter(const ngraph::element::Type element_type,
                                                     const Shape&                shape);
    }
}
