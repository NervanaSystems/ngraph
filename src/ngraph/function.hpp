//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <atomic>
#include <initializer_list>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/parameter_vector.hpp"
#include "ngraph/op/result_vector.hpp"

namespace ngraph
{
    /// A user-defined function.
    class Function
    {
    public:
        Function(const NodeVector& results,
                 const op::ParameterVector& parameters,
                 const std::string& name = "");

        Function(const std::shared_ptr<Node>& result,
                 const op::ParameterVector& parameters,
                 const std::string& name = "");

        Function(const ResultVector& results,
                 const op::ParameterVector& parameters,
                 const std::string& name = "");

        void init();

        virtual ~Function() {}
    public:
        /// Return the number of outputs for this function.
        size_t get_output_size() const;

        /// Return the op that generates output i
        std::shared_ptr<Node> get_output_op(size_t i) const;

        /// Return the element type of output i
        const element::Type& get_output_element_type(size_t i) const;

        /// Return the shape of element i
        const Shape& get_output_shape(size_t i) const;

        /// Return the function parameters
        const op::ParameterVector& get_parameters() const { return m_parameters; }
        /// Return a list of function's outputs
        const ResultVector& get_results() const { return m_results; }
        /// Check that there is a single result and return it.
        std::shared_ptr<Node> get_result() const;

        const std::string& get_friendly_name() const;
        const std::string& get_name() const;
        // so we can use `dynamic_cast` in FunctionCall to double check if we are dealing with
        //  an XLA or regular function
        void set_name(const std::string& name);
        std::list<std::shared_ptr<Node>> get_ops(bool include_control_deps = true) const;
        std::list<std::shared_ptr<Node>> get_ordered_ops(bool include_control_deps = true) const;
        friend std::ostream& operator<<(std::ostream&, const Function&);
        size_t get_instance_id() { return m_instance_id; }
        size_t get_temporary_pool_size();
        void set_temporary_pool_size(size_t);
        // updates graph and m_results list
        void replace_node(std::shared_ptr<Node> old, std::shared_ptr<Node> repl);

        void validate_nodes_and_infer_types();

    protected:
        ResultVector m_results;
        op::ParameterVector m_parameters;
        size_t m_temporary_pool_size;

    private:
        Function(const Function&) = delete;
        Function(const Function&&) = delete;
        Function& operator=(const Function&) = delete;

        static std::atomic<size_t> m_next_instance_id;
        size_t m_instance_id;
        std::string m_name;
        const std::string m_unique_name;
    };
}
