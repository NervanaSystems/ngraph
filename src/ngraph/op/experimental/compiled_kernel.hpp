//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/op/op.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief CompiledKernel represents a sub-graph that can be compiled and executed
        /// independently.
        ///
        /// This op can be used to delimit sub-graphs that with special compilation requirements
        /// within a function. For example, we currently use it to delimit sub-graphs that will be
        /// independently compiled and executed by MLIR backend.
        class CompiledKernel : public ngraph::op::Op
        {
        public:
            CompiledKernel(const NodeVector& node_list,
                           const NodeVector& outputs,
                           const NodeVector& args);
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const NodeVector& get_node_list() const { return m_node_list; }
            const NodeVector& get_kernel_outputs() const { return m_output_nodes; }
            descriptor::Tensor* get_head_node_input(Node* node, int arg_no)
            {
                auto it = m_node_inputs_map.find(node);
                if (it == m_node_inputs_map.end())
                {
                    return nullptr;
                }

                auto vector = it->second;
                return vector[arg_no];
            };
            void add_head_node_input(Node* node, descriptor::Tensor* tensor)
            {
                m_node_inputs_map[node].push_back(tensor);
            }
        private:
            NodeVector m_node_list;
            NodeVector m_output_nodes;
            // Maps head nodes to their input tensor ptrs
            // Needed because we remove all inputs to the sub-graph to make it unreachable. 
            std::unordered_map<Node*, std::vector<descriptor::Tensor*>> m_node_inputs_map;
        };
    }
}
