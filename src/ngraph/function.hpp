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

#include <initializer_list>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/op.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/runtime/instruction.hpp"
#include "ngraph/types/type.hpp"

namespace ngraph
{
    /// A user-defined function.
    class Function
    {
    public:
        Function(const std::shared_ptr<Node>& result,
                 const std::shared_ptr<ValueType>& result_type,
                 const std::vector<std::shared_ptr<op::Parameter>>& parameters,
                 const std::string& name = "");

        std::shared_ptr<Node> get_result() { return m_result; }
        const std::vector<std::shared_ptr<op::Parameter>> get_parameters() const
        {
            return m_parameters;
        }
        const std::shared_ptr<ValueType> get_result_type() const { return m_result_type; }
        std::string get_name() const;
        void set_name(const std::string& name);
        std::list<Node*>& get_ops();
        const std::list<Node*>& get_ops() const;
        std::list<Node*>& get_ordered_ops();
        const std::list<Node*>& get_ordered_ops() const;
        void set_ordered_ops(const std::list<Node*>&);
        void set_ordered_ops_valid() { m_ordered_ops_valid = true; }
        void clear_ordered_ops_valid() { m_ordered_ops_valid = false; }
        friend std::ostream& operator<<(std::ostream&, const Function&);

    protected:
        std::shared_ptr<Node> m_result;
        std::vector<std::shared_ptr<ngraph::op::Parameter>> m_parameters;
        std::string m_name;
        std::shared_ptr<ValueType> m_result_type;
        bool m_ordered_ops_valid;
        std::list<Node*> m_ordered_ops;
        std::list<Node*> m_ops;

    private:
        Function(const Function&) = delete;
        Function(const Function&&) = delete;

        static size_t m_next_instance_id;
        size_t m_instance_id;
    };
}
