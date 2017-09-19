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

#include <memory>

#include "descriptor/tensor.hpp"

namespace ngraph
{
    class Node;

    namespace descriptor
    {
        class Output;

        // Describes a tensor that is an input to an op, directly or indirectly via a tuple
        class Input
        {
            friend class Node;

        public:
            /// @param node The node that owns this input; not shared to prevent owner loop
            /// @param index The position of this this tensor in all input tensors
            /// @param argno The position of the argument with this tensor
            /// @param arg_index The position of the tensor within the argument's tensors
            /// @param output The output that supplies a value for this input
            Input(Node*   node,
                  size_t  index,
                  size_t  argno,
                  size_t  arg_index,
                  Output& output);

            std::shared_ptr<Node> get_node();
            size_t                get_argno() const { return m_argno; }
            size_t                get_arg_index() const { return m_arg_index; }
            size_t                get_index() const { return m_index; }
            const Output&         get_output() const { return m_output; }
            Output&               get_output() { return m_output; }
            const Tensor&         get_tensor() const;
            Tensor&               get_tensor();

        protected:
            Node*   m_node;      // The node we are an input for
            size_t  m_index;     // Index into all input tensors
            size_t  m_argno;     // Arg number for this input
            size_t  m_arg_index; // Index into arg's tensors
            Output& m_output;

        private:
            // Input(const Input&) = default;
            // Input(Input&&) = default;
            // Input& operator=(const Input&) = delete;
        };
    }
}
