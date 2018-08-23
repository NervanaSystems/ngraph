/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <memory>
#include <set>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/tensor_view.hpp"

namespace ngraph
{
    // The forward declaration of Node is needed here because Node has a deque of
    // Outputs, and Output is an incomplete type at this point. STL containers of
    // incomplete type have undefined behavior according to the C++11 standard, and
    // in practice including node.hpp here was causing compilation errors on some
    // systems (namely macOS).
    class Node;

    namespace descriptor
    {
        // Describes an output tensor of an op
        class Output
        {
        public:
            /// @param node Node that owns this output.
            /// @param index Position of the output tensor in all output tensors
            /// @param tensor_view The view of this tensor; where the value will be written
            Output(Node* node, size_t index, const std::shared_ptr<TensorView>& tensor_view);

            std::shared_ptr<Node> get_node() const;
            size_t get_index() const { return m_index; }
            std::shared_ptr<TensorView> get_tensor_view() const { return m_tensor_view; }
            void set_tensor_view(const std::shared_ptr<TensorView>& tensor_view)
            {
                m_tensor_view = tensor_view;
            }
            void add_input(Input* input);
            void remove_input(Input* input);
            const std::set<Input*>& get_inputs() const { return m_inputs; }
            Tensor& get_tensor() const;

        protected:
            /// @return the tensor view type for the output
            std::shared_ptr<const TensorViewType> get_tensor_view_type() const;

        public:
            /// @return the shape of the output
            const Shape& get_shape() const;
            /// @return the element type of the output
            const element::Type& get_element_type() const;

        protected:
            Node* m_node;
            size_t m_index;
            std::shared_ptr<TensorView> m_tensor_view;
            std::set<Input*> m_inputs;

        private:
            Output(const Output&) = delete;
            Output(Output&&) = delete;
            Output& operator=(const Output&) = delete;
        };
    }
}
