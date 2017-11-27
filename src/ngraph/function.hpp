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

#include <atomic>
#include <cassert>
#include <initializer_list>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/op.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/types/type.hpp"

namespace ngraph
{
    /// A user-defined function.
    class Function
    {
    public:
        Function(const Nodes& results,
                 const std::vector<std::shared_ptr<const ValueType>>& result_types,
                 const std::vector<std::shared_ptr<op::Parameter>>& parameters,
                 const std::string& name = "");

        Function(const std::shared_ptr<Node>& result,
                 const std::shared_ptr<const ValueType>& result_type,
                 const std::vector<std::shared_ptr<op::Parameter>>& parameters,
                 const std::string& name = "");

        Function(const std::shared_ptr<Node>& result,
                 const std::vector<std::shared_ptr<op::Parameter>>& parameters,
                 const std::string& name = "");

        virtual ~Function() {}
        std::shared_ptr<Node> get_result() //TODO: push up to XLAFunction
        {
            assert(m_results.size() < 2);
            return m_results[0];
        }

        const Nodes& get_results() const { return m_results; }
        std::vector<descriptor::Output*> get_outputs();

        const std::vector<std::shared_ptr<op::Parameter>>& get_parameters() const
        {
            return m_parameters;
        }
        std::shared_ptr<const ValueType> get_result_type() const //TODO: push up to XLAFunction
        {
            assert(m_results.size() < 2);
            return m_results[0]->get_value_type();
        }
        std::vector<std::shared_ptr<const ValueType>> get_result_types() const
        {
            std::vector<std::shared_ptr<const ValueType>> result_types{};
            for (auto r : m_results)
            {
                result_types.push_back(r->get_value_type());
            }
            return result_types;
        } //TODO: switch ValueType to TensorViewType
        std::string get_name() const;
        virtual void
            set_name(const std::string&
                         name); //so we can check if we are dealing with XLA or regular function
        std::list<std::shared_ptr<Node>>&
            get_ops(); //overriding get_result_type doesn't work because we would like different behaviours(checks) depending on a caller
        const std::list<std::shared_ptr<Node>>& get_ops() const;
        std::list<std::shared_ptr<Node>>& get_ordered_ops();
        const std::list<std::shared_ptr<Node>>& get_ordered_ops() const;
        void set_ordered_ops(const std::list<std::shared_ptr<Node>>&);
        void set_ordered_ops_valid() { m_ordered_ops_valid = true; }
        void clear_ordered_ops_valid() { m_ordered_ops_valid = false; }
        friend std::ostream& operator<<(std::ostream&, const Function&);
        size_t get_instance_id() { return m_instance_id; }
        size_t get_temporary_pool_size();
        void set_temporary_pool_size(size_t);

    protected:
        Nodes m_results;
        std::vector<std::shared_ptr<ngraph::op::Parameter>> m_parameters;
        std::string m_name;
        std::vector<std::shared_ptr<const ValueType>> m_result_types;
        bool m_ordered_ops_valid;
        std::list<std::shared_ptr<Node>> m_ordered_ops;
        std::list<std::shared_ptr<Node>> m_ops;
        size_t m_temporary_pool_size;

    private:
        Function(const Function&) = delete;
        Function(const Function&&) = delete;

        static std::atomic<size_t> m_next_instance_id;
        size_t m_instance_id;
    };
}
