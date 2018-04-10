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

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/type/type.hpp"

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
            /// @param node The node that owns this input
            /// @param index The position of this this tensor in all input tensors
            /// @param output The output that supplies a value for this input
            Input(Node* node, size_t index, Output& output);

            /// @return the node that this is an input of
            std::shared_ptr<Node> get_node();

            /// @return the position within all supplied tensors of this input
            size_t get_index() const { return m_index; }
            // @return the connected output
            const Output& get_output() const { return *m_output; }
            // @return the connected output
            Output& get_output() { return *m_output; }
            // @return the tensor of the connected output
            const Tensor& get_tensor() const;

            // @return the tensor of the connected output
            Tensor& get_tensor();

            void replace_output(std::shared_ptr<Node> node, size_t i);
            void replace_output(Output& output);

        protected:
            /// @return the tensor view for the connected output
            std::shared_ptr<const TensorView> get_tensor_view() const;

            /// @return the tensor view for the connected output
            std::shared_ptr<TensorView> get_tensor_view();

            /// @return the tensor view type for the connected output
            std::shared_ptr<const TensorViewType> get_tensor_view_type() const;

        public:
            /// @return the shape of the connected output
            const Shape& get_shape() const;

            /// @return the element type of the connected output
            const element::Type& get_element_type() const;

        protected:
            //owner of an argument node (in lieu of m_arguments)
            std::shared_ptr<Node> m_src_node;
            Node* m_node;   // The node we are an input for
            size_t m_index; // Index into all input tensors
            Output* m_output;

        private:
            Input(const Input&) = delete;
            Input(Input&&) = delete;
            Input& operator=(const Input&) = delete;
        };
    }
}
