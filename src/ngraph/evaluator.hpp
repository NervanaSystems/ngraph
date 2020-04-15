//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <map>
#include <stack>
#include <utility>

#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type_traits.hpp"

namespace ngraph
{
    /// \brief Execute handlers on a subgraph to compute values
    ///
    ///
    template <typename V>
    class Evaluator
    {
    public:
        /// \brief values we compute for outputs
        using value_map = std::map<RawNodeOutput, V>;

        /// \brief Handler for a computation of a value about an op
        ///
        /// A handler is passed a Node* and a vector of computed input values. The handler should
        /// return a vector of computed output values.
        using op_handler = std::function<std::vector<V>(Node* op, std::vector<V>& inputs)>;

        /// \brief Table of ops with handlers
        using op_handler_map = std::map<Node::type_info_t, op_handler>;

        /// \brief construct  handler using the provided op handlers.
        ///
        /// Evaluations share previously computed values so that calls on multiple nodes can share
        /// work. All state is kept in the value map, which is accessible for clearing or seeding
        /// with
        /// Evaluator::get_value_map().
        ///
        /// \param Handlers for ops. Pairs of Node::type_info_t and handler functions.
        Evaluator(const op_handler_map& handlers)
            : m_handlers(handlers)
        {
        }

        /// \brief Retrieves the value_map, which holds all Output<Node> value associations.
        value_map& get_value_map() { return m_value_map; }
        const value_map& get_value_map() const { return m_value_map; }
    protected:
        /// \brief Intstructions for evaluations state machine
        class Inst
        {
        public:
            /// \brief Ensure value has been analyzed
            Inst(const Output<Node>& value)
                : m_node(value.get_node())
                , m_index(value.get_index())
            {
            }

            Inst(const RawNodeOutput& value)
                : m_node(value.node)
                , m_index(value.index)
            {
            }

            /// \brief All arguments have been handled; execute the node handler
            Inst(Node* node)
                : m_node(node)
            {
            }

            /// \brief True if this is a value instruction
            bool is_value() const { return m_index >= 0; }
            RawNodeOutput get_value() const { return RawNodeOutput(m_node, m_index); }
            Node* get_node() const { return m_node; }
        private:
            Node* m_node;
            int64_t m_index{-1};
        };

    public:
        /// \brief Determine information about value
        V evaluate(const Output<Node>& value)
        {
            std::stack<Inst> inst_stack;
            inst_stack.push(value);
            while (!inst_stack.empty())
            {
                auto inst = inst_stack.top();
                inst_stack.pop();
                auto node = inst.get_node();
                if (m_value_map.find(node->output(0)) != m_value_map.end())
                {
                    // Already computed
                    continue;
                }
                if (inst.is_value())
                {
                    // Request to analyze this value if we can
                    if (m_handlers.find(node->get_type_info()) != m_handlers.end())
                    {
                        // Ensure the inputs are processed and then execute the op handler
                        inst_stack.push(node);
                        for (auto v : node->input_values())
                        {
                            inst_stack.push(v);
                        }
                    }
                    else
                    {
                        // We don't know how to handle this op, so mark the outputs as unknown
                        for (auto output : node->outputs())
                        {
                            m_value_map[output] = V();
                        }
                    }
                }
                else
                {
                    // Request to execute the handleer. Pass what we know about the inputs to the
                    // handler and associate the results with the outputs
                    std::vector<V> inputs;
                    for (auto v : node->input_values())
                    {
                        inputs.push_back(m_value_map.at(v));
                    }
                    std::vector<V> outputs = m_handlers.at(node->get_type_info())(node, inputs);
                    for (size_t i = 0; i < outputs.size(); ++i)
                    {
                        m_value_map[node->output(i)] = outputs[i];
                    }
                }
            }
            return m_value_map.at(value);
        }

    protected:
        op_handler_map m_handlers;
        value_map m_value_map;
    };
}
