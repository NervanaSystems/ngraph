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
#include "ngraph/op.hpp"
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
    public:
        using ptr = std::shared_ptr<Parameter>;

        Parameter(Function& function, size_t index);

    protected:
        Function& m_function;
        size_t    m_index;
    };

    /**
     ** The result of a function. The ndoe addociated with the result
     ** supplies the return value when the function is called.
     **/
    class Result : public TypedValueMixin
    {
    public:
        using ptr = std::shared_ptr<Result>;

        Node::ptr value() const { return m_value; }
        void      value(const Node::ptr& value) { m_value = value; }

    protected:
        Node::ptr m_value;
    };

    /**
     ** A user-defined function.
     **/
    class Function : public Op
    {
    public:
        Function(size_t n_parameters);

        Result* result() { return &m_result; }

        Parameter::ptr parameter(size_t i) { return m_parameters[i]; }

    protected:
        std::vector<Parameter::ptr> m_parameters;
        Result                      m_result;
    };
}
