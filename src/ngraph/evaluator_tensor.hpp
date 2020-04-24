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

#pragma once

#include <memory>
#include <vector>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"

namespace ngraph
{
    class Node;
    template <typename T>
    class Output;
    class EvaluatorTensor;
    // This is the name we want to use; a handle to the API implementation
    using EvaluatorTensorPtr = std::shared_ptr<EvaluatorTensor>;
    using EvaluatorTensorVector = std::vector<EvaluatorTensorPtr>;

    NGRAPH_API
    std::string node_evaluation_failure_loc_string(const Node* node);

    class NGRAPH_API NodeEvaluationFailure : public CheckFailure
    {
    public:
        NodeEvaluationFailure(const CheckLocInfo& check_loc_info,
                              const Node* node,
                              const std::string& explanation)
            : CheckFailure(check_loc_info, node_evaluation_failure_loc_string(node), explanation)
        {
        }
    };

#define NODE_EVALUATION_CHECK(node, ...)                                                           \
    NGRAPH_CHECK_HELPER(::ngraph::NodeValidationFailure, (node), __VA_ARGS__)

    namespace runtime
    {
        class HostTensor;
    }

    /// \brief A generic handle to (potential) storage with element type and shape information
    class NGRAPH_API EvaluatorTensor
    {
    protected:
        EvaluatorTensor(const element::Type& element_type,
                        const PartialShape& partial_shape,
                        bool is_allocated);
        /// \brief Get type/shape from value
        EvaluatorTensor(const Output<Node>& value, bool is_allocated);

        /// \brief Marks the tensor as allocated
        void set_is_allocated();

    public:
        virtual ~EvaluatorTensor();
        /// \return pointer to storage. The element type and shape must be static when this is
        /// called.
        virtual void* get_data_ptr() = 0;
        /// \return The current element type
        const element::Type& get_element_type() const;
        /// \return The current partial shape
        const PartialShape& get_partial_shape() const;
        /// \return the static shape
        const Shape get_shape() const;
        /// \return The number of elements in the shape
        size_t get_element_count() const;
        /// \return The number of bytes in the shape
        size_t get_size_in_bytes() const;
        /// \return True if storage has been allocated
        bool get_is_allocated() const;
        /// \brief Set the element type. Must be compatible with the current element type.
        /// \param node The node being evaluated, for context
        /// \param element_type The element type
        void set_element_type(Node* node, const element::Type& element_type);
        /// \brief Set the actual shape of the tensor compatibly with the partial shape.
        /// \param node The node being evaluated (for errors)
        /// \param shape The shape being set
        void set_shape(Node* node, const Shape& shape);
        /// \brief Set the shape of a node from an input
        /// \param node The node being evaluated (for errors)
        /// \param arg The input argument
        void set_unary(Node* node, const EvaluatorTensorPtr& arg);
        /// \brief Set the shape of the tensor using broadcast rules
        /// \param node The node being evaluated (for errors)
        /// \param autob The broadcast mode
        /// \param arg0 The first argument
        /// \param arg1 The second argument
        void set_broadcast(Node* node,
                           const op::AutoBroadcastSpec& autob,
                           const EvaluatorTensorPtr& arg0,
                           const EvaluatorTensorPtr& arg1);

        /// \brief Get a pointer of the appropriate type. Will allocate if necessary.
        /// \tparam The element type
        template <element::Type_t ET>
        typename element_type_traits<ET>::value_type* get_data_ptr()
        {
            return static_cast<typename element_type_traits<ET>::value_type*>(get_data_ptr());
        }

    protected:
        element::Type m_element_type;
        PartialShape m_partial_shape;
        bool m_is_allocated;
    };
}
