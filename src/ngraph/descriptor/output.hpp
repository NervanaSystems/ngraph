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
#include <set>

#include "ngraph/descriptor/tensor_view.hpp"

namespace ngraph
{
    namespace descriptor
    {
        // Describes an output tensor of an op
        class Output
        {
            // For some odd reason emplace_back is requiring a copy constructor
            // it should not. See issue #111 for details
            // Output(const Output&) = delete;
            // Output& operator=(const Output&) = delete;

        public:
            /// @param node Node that owns this output.
            /// @param index Position of the output tensor in all output tensors
            /// @param tensor_view The view of this tensor; where the value will be written
            Output(const std::shared_ptr<Node>& node,
                   size_t index,
                   const std::shared_ptr<TensorView>& tensor_view);

            std::shared_ptr<Node> get_node() const;
            size_t get_index() const { return m_index; }
            std::shared_ptr<TensorView> get_tensor_view() const { return m_tensor_view; }
            void add_input(Input* input);
            const std::set<Input*>& get_inputs() const { return m_inputs; }
            const Tensor& get_tensor() const;
            Tensor& get_tensor();

        protected:
            std::weak_ptr<Node> m_node;
            size_t m_index;
            std::shared_ptr<TensorView> m_tensor_view;
            std::set<Input*> m_inputs;
        };
    }
}
