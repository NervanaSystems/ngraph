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
#include "ngraph/type.hpp"

namespace ngraph
{

    class Function;
    
    /**
        ** One parameter of a function. Within the function's graph
        ** the parameter is a node that represents the argument in a call.
        **/
    class Parameter : public Node
    {
        friend class Function;
    protected:
        void assign_function(Function* function, size_t index);

    public:
        Parameter(const ValueType::ptr& value_type);

        std::string description() const override { return "Parameter"; }

        virtual void propagate_types() override;

    protected:
        Function* m_function;
        size_t    m_index;
    };

    namespace op
    {
        std::shared_ptr<ngraph::Parameter> parameter(const ValueType::ptr& value_type=nullptr);
        std::shared_ptr<ngraph::Parameter> parameter(const ngraph::element::Type element_type, const Shape& shape);
    }
}
