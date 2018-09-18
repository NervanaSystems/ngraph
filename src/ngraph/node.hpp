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
#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/assertion.hpp"
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/placement.hpp"

namespace ngraph
{
    namespace pass
    {
        class GetOutputElementElimination;
    }
    namespace op
    {
        class Parameter;
        class Result;
    }

    void replace_node_users_arguments(std::shared_ptr<Node> target,
                                      std::shared_ptr<Node> replacement);

    std::pair<std::shared_ptr<op::Result>, std::shared_ptr<op::Parameter>>
        insert_result_parameter_split(const std::shared_ptr<Node>& src_node,
                                      const std::shared_ptr<Node>& dst_node);

    void insert_new_node_between(const std::shared_ptr<Node>& src_node,
                                 const std::shared_ptr<Node>& dst_node,
                                 const std::shared_ptr<Node>& new_node);

    std::string node_validation_assertion_string(const Node* node);

    const std::shared_ptr<Node>& check_single_output_arg(const std::shared_ptr<Node>& node,
                                                         size_t i);
    const NodeVector& check_single_output_args(const NodeVector& args);

    const std::shared_ptr<Node>& check_single_output_arg(const std::shared_ptr<Node>& node,
                                                         size_t i);
    const NodeVector& check_single_output_args(const NodeVector& args);

    /// Nodes are the backbone of the graph of Value dataflow. Every node has
    /// zero or more nodes as arguments and one value, which is either a tensor
    /// view or a (possibly empty) tuple of values.
    class Node : public std::enable_shared_from_this<Node>
    {
        // So Adjoints can call generate_adjoints
        friend class autodiff::Adjoints;
        friend class descriptor::Input;
        friend void replace_node_users_arguments(std::shared_ptr<Node> target,
                                                 std::shared_ptr<Node> replacement);
        friend std::pair<std::shared_ptr<op::Result>, std::shared_ptr<op::Parameter>>
            insert_result_parameter_split(const std::shared_ptr<Node>& src_node,
                                          const std::shared_ptr<Node>& dst_node);
        friend void insert_new_node_between(const std::shared_ptr<Node>& src_node,
                                            const std::shared_ptr<Node>& dst_node,
                                            const std::shared_ptr<Node>& new_node);

        friend class ngraph::pass::GetOutputElementElimination;

    protected:
        /// Throws if the node is invalid.
        virtual void validate_and_infer_types();

        // Called in constructors during transition
        void constructor_validate_and_infer_types();

        void validate_and_infer_elementwise(element::Type result_type);
        void validate_and_infer_elementwise()
        {
            validate_and_infer_elementwise(get_input_element_type(0));
        }
        void validate_and_infer_elementwise_arithmetic();
        void validate_and_infer_elementwise_logical();

        Node(const std::string& node_type, const NodeVector& arguments, size_t output_size = 1);
        virtual ~Node();

        virtual void generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas) {}
    public:
        // Called after transition
        void delayed_validate_and_infer_types();

        /// The class name, must not contain spaces
        std::string description() const { return m_node_type; }
        const std::string& get_friendly_name() const;
        const std::string& get_name() const;
        void set_name(const std::string& name);
        /// Return true if this has the same implementing class as node. This
        /// will be used by the pattern matcher when comparing a pattern
        /// graph against the graph.
        bool is_same_op_type(const std::shared_ptr<Node>& node) const
        {
            Node* n = node.get();
            return std::type_index(typeid(*this)) == std::type_index(typeid(*n));
        }

        void set_output_type(size_t i, const element::Type& element_type, const Shape& shape);

        bool is_parameter() const;
        virtual bool is_output() const;
        virtual bool is_constant() const;
        virtual bool is_commutative() { return false; }
        size_t get_instance_id() const { return m_instance_id; }
        friend std::ostream& operator<<(std::ostream&, const Node&);
        virtual std::ostream& write_short_description(std::ostream&) const;
        virtual std::ostream& write_long_description(std::ostream&) const;

        // TODO: Deprecate
        std::deque<descriptor::Input>& get_inputs() { return m_inputs; }
        // TODO: Deprecate
        const std::deque<descriptor::Input>& get_inputs() const { return m_inputs; }
        // Deprecated
        // TODO: Remove from unit tests.
        std::deque<descriptor::Output>& get_outputs();
        // Deprecated
        // TODO: Remove from unit tests.
        const std::deque<descriptor::Output>& get_outputs() const;

        /// Get control dependencies registered on the node
        const std::set<std::shared_ptr<Node>>& get_control_dependencies() const;

        void add_control_dependency(std::shared_ptr<Node> node);

        void remove_control_dependency(std::shared_ptr<Node> node)
        {
            m_control_dependencies.erase(node);
        }

        /// Returns the number of outputs on the for the node.
        size_t get_output_size() const;

        /// Returns the element type for output i
        const element::Type& get_output_element_type(size_t i) const;

        /// Checks that there is exactly one output and returns its element type
        const element::Type& get_element_type() const;

        /// Returns the shape for output i
        const Shape& get_output_shape(size_t i) const;

        /// Checks that there is exactly one output and returns its shape
        const Shape& get_shape() const;

        /// Returns the static value for output i
        const StaticValue& get_output_static_value(size_t i) const;

        /// Checks that there is exactly one output and returns its static value
        const Shape& get_static_value() const;

        /// Returns the tensor for output i
        descriptor::Tensor& get_output_tensor(size_t i) const;

        /// Checks that there is exactly one output and returns its tensor.
        descriptor::Tensor& get_output_tensor() const;

        /// Returns the tensor view of output i
        std::shared_ptr<descriptor::Tensor> get_output_tensor_ptr(size_t i) const;

        /// Checks that there is exactly one output and returns its tensor view.
        std::shared_ptr<descriptor::Tensor> get_output_tensor_ptr() const;

        /// Returns the set of inputs using output i
        const std::set<descriptor::Input*>& get_output_inputs(size_t i) const;

        /// Returns the number of inputs for the op
        size_t get_input_size() const;

        /// Returns the element type of input i
        const element::Type& get_input_element_type(size_t i) const;

        /// Returns the shape of input i
        const Shape& get_input_shape(size_t i) const;

        std::unordered_set<descriptor::Tensor*> liveness_new_list;
        std::unordered_set<descriptor::Tensor*> liveness_free_list;

        virtual NodeVector get_arguments() const;

        std::shared_ptr<Node> get_argument(size_t index) const;

        virtual std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const = 0;

        virtual std::vector<std::shared_ptr<Function>> get_functions() const;

        /// True if this and node have one output with same element type and shape
        bool has_same_type(std::shared_ptr<const Node> node) const;

        /// Get device placement
        Placement get_placement() const;

        /// Set device placement
        void set_placement(Placement placement);

        /// Get input descriptor that is connected to src
        descriptor::Input* get_input_from(const std::shared_ptr<Node>& src);

        /// Get ouput descriptor that outputs to dst
        descriptor::Output* get_output_to(const std::shared_ptr<Node>& dst);

        /// Get all the nodes that uses the current node
        NodeVector get_users() const;

        virtual std::shared_ptr<Node> get_default_value() const { return nullptr; }
        /// Use instance ids for comparison instead of memory addresses to improve determinism
        bool operator<(const Node& other) const { return m_instance_id < other.m_instance_id; }
    protected:
        std::set<std::shared_ptr<Node>> m_control_dependencies;
        void set_output_size(size_t n);
        void set_output_static_value(size_t n, const StaticValue& static_value);
        void set_static_value(const StaticValue& static_value);
        void clear_output_static_value(size_t n);
        void clear_static_value();

        std::string m_node_type;
        size_t m_instance_id;
        std::string m_name;
        const std::string m_unique_name;
        static std::atomic<size_t> m_next_instance_id;
        std::deque<descriptor::Input> m_inputs;
        std::deque<descriptor::Output> m_outputs;
        std::unordered_map<Node*, autodiff::Adjoints> m_adjoint_map;
        Placement m_placement = Placement::DEFAULT;
    };

    class NodeValidationError : public AssertionFailure
    {
    public:
        NodeValidationError(std::string what)
            : AssertionFailure(what)
        {
        }
        NodeValidationError(const char* what)
            : AssertionFailure(what)
        {
        }
    };

    class NodeDescription
    {
    public:
        NodeDescription(const Node& node, bool is_short)
            : m_node(node)
            , m_is_short(is_short)
        {
        }

        friend std::ostream& operator<<(std::ostream& out, const NodeDescription node_description)
        {
            return node_description.m_is_short
                       ? node_description.m_node.write_short_description(out)
                       : node_description.m_node.write_long_description(out);
        }
        const Node& m_node;
        bool m_is_short;
    };

    void check_new_args_count(const Node* node, const NodeVector& new_args);
}

#define NODE_VALIDATION_ASSERT(node, cond)                                                         \
    NGRAPH_ASSERT_STREAM_WITH_LOC(                                                                 \
        ::ngraph::NodeValidationError, cond, ::ngraph::node_validation_assertion_string(node))
#define NODE_VALIDATION_FAIL(node)                                                                 \
    NGRAPH_FAIL_STREAM_WITH_LOC(::ngraph::NodeValidationError,                                     \
                                ::ngraph::node_validation_assertion_string(node))
