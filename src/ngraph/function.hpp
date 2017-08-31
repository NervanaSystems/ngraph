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

#include "node.hpp"
#include "op.hpp"
#include "ops/parameter.hpp"
#include "type.hpp"

namespace ngraph
{
    /**
     ** A user-defined function.
     **/
    class Function
    {
    public:
        Function(const Node::ptr&                               result,
                 const std::vector<std::shared_ptr<Parameter>>& parameters);

        Node::ptr result() { return m_result; }

        Parameter::ptr parameter(size_t i) { return m_parameters[i]; }

        std::string name() const { return m_name; }

    protected:
        Node::ptr                                       m_result;
        std::vector<std::shared_ptr<ngraph::Parameter>> m_parameters;
        std::string                                     m_name;
    };

    namespace op
    {
        std::shared_ptr<Function>
            function(const Node::ptr&                                         result,
                     const std::initializer_list<std::shared_ptr<Parameter>>& parameters);
        std::shared_ptr<Function>
            function(const Node::ptr&                               result,
                     const std::vector<std::shared_ptr<Parameter>>& parameters);
    }
}
