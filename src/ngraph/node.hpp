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

#include <atomic>
#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <tuple>
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
    class NodeOutput;
    class NodeInput;

    /// Nodes are the backbone of the graph of Value dataflow. Every node has
    /// zero or more nodes as arguments and one value, which is either a tensor
    /// or a (possibly empty) tuple of values.
    class Node : public std::enable_shared_from_this<Node>
    {
        // So Adjoints can call generate_adjoints
        friend class autodiff::Adjoints;

    public:
        virtual ~Node();

        //===
        // NAMES/IDS
        //===

        /// \brief Get the unique name of the node.
        /// \returns A const reference to the node's unique name.
        const std::string& get_name() const;

        /// \brief Sets a friendly name for a node. This does not overwrite the unique name
        ///        of the node and is retrieved via get_friendly_name(). Used mainly for debugging.
        ///        The friendly name may be set exactly once.
        /// \param name is the friendly name to set
        void set_friendly_name(const std::string& name);

        /// \brief Gets the friendly name for a node. If no friendly name has been set via
        ///        set_friendly_name then the node's unique name is returned.
        /// \returns A const reference to the node's friendly name.
        const std::string& get_friendly_name() const;

        size_t get_instance_id() const { return m_instance_id; }
        /// Use instance ids for comparison instead of memory addresses to improve determinism
        bool operator<(const Node& other) const { return m_instance_id < other.m_instance_id; }
        //===
        // VALIDATION
        //===
        void revalidate_and_infer_types() { validate_and_infer_types(); }
        // Called after transition
        void delayed_validate_and_infer_types();

        //===
        // NODE TYPES
        //===

        /// \brief Get the string name for the type of the node, such as `Add` or `Multiply`.
        ///        The class name, must not contain spaces as it is used for codegen.
        /// \returns A const reference to the node's type name
        const std::string& description() const;

        bool is_parameter() const;
        // Should be is_result():
        virtual bool is_output() const;
        virtual bool is_constant() const;
        virtual bool is_null() const { return false; }
        virtual bool is_op() const { return false; }
        virtual bool is_commutative() { return false; }
        //===
        // PRETTY PRINTING
        //===
        friend std::ostream& operator<<(std::ostream&, const Node&);
        // TODO: change the virtuals to just something to pretty-print attributes
        virtual std::ostream& write_short_description(std::ostream&) const;
        virtual std::ostream& write_long_description(std::ostream&) const;
        // TODO: method to return long_desc, short_desc?

        //===
        // OUTPUTS
        //===
        // Deprecated
        // Bopped outside backends.
        // TODO: Remove from unit tests.
        std::deque<descriptor::Output>& get_outputs(); // TRIAGED (to be deprecated)
        // Deprecated
        // Bopped outside backends.
        // TODO: Remove from unit tests.
        const std::deque<descriptor::Output>& get_outputs() const; // TRIAGED (to be deprecated)

        /// Returns the number of outputs from this node.
        size_t get_output_size() const;

        /// Returns the element type for output i.
        const element::Type& get_output_element_type(size_t i) const;

        /// Returns the shape for output i
        const Shape& get_output_shape(size_t i) const;

        /// Returns the partial shape for output i
        const PartialShape& get_output_partial_shape(size_t i) const;

        /// Returns the tensor for output i
        descriptor::Tensor& get_output_tensor(size_t i) const;

        /// Returns the tensor of output i
        std::shared_ptr<descriptor::Tensor> get_output_tensor_ptr(size_t i) const;

        // DEPRECATED: Assumes only one output
        /// Checks that there is exactly one output and returns its element type
        const element::Type& get_element_type() const;

        // DEPRECATED: Assumes only one output
        /// Checks that there is exactly one output and returns its shape
        const Shape& get_shape() const;

        //===
        // INPUTS
        //===
        // TODO: Deprecate
        // Bopped outside backends
        std::deque<descriptor::Input>& get_inputs() { return m_inputs; } // TRIAGED (to be deprecated)
        // TODO: Deprecate
        // Bopped outside backends
        const std::deque<descriptor::Input>& get_inputs() const { return m_inputs; } // TRIAGED (to be deprecated)
        /// Returns the number of inputs for the op
        size_t get_input_size() const;

        /// Returns the element type of input i
        const element::Type& get_input_element_type(size_t i) const;

        /// Returns the shape of input i
        const Shape& get_input_shape(size_t i) const;

        /// Returns the partial shape of input i
        const PartialShape& get_input_partial_shape(size_t i) const;

        descriptor::Tensor& get_input_tensor(size_t i);
        const descriptor::Tensor& get_input_tensor(size_t i) const;
        descriptor::Tensor* get_input_tensor_ptr(size_t i) const;

        // virtual just because GetOutputElement wants to override...
        // TODO: make non-virtual
        // TODO: rename to something like "get_input_source_nodes", or deprecate.
        virtual NodeVector get_arguments() const;

        // TODO: rename to something like "get_input_source_node".
        std::shared_ptr<Node> get_argument(size_t index) const;

        NodeOutput get_input_source_output(size_t input_index) const;

        //===
        // RELATING OUTPUTS AND INPUTS
        //===

        // NEW
        void replace_input_source(size_t input_index, const NodeOutput& src_output);
        void replace_input_source(size_t input_index, const std::shared_ptr<Node>& source_node, size_t output_index);

        // NEW
        // Could be made to return const& at some point
        // TODO: vector for now but should it be unordered_set? set?
        std::vector<NodeInput> get_output_targets(size_t output_index) const;

        // TO DEPRECATE
        // Bopped outside of backends
        /// Returns the set of inputs using output i
        const std::set<descriptor::Input*>& get_output_inputs(size_t i) const;

        // TO DEPRECATE
        // Bopped outside of backends EXCEPT insert_node_between
        /// Get input descriptors that are connected from src to this
        std::vector<descriptor::Input*> get_inputs_from(const std::shared_ptr<Node>& src);

        // TO DEPRECATE
        // Bopped outside of backends EXCEPT insert_node_between
        /// Get output descriptors that are connected from this to dst
        std::vector<descriptor::Output*> get_outputs_to(const std::shared_ptr<Node>& dst);

        // TO DEPRECATE???
        /// Get all the nodes that use the current node
        // TODO: decide if this should also include control dependents.
        NodeVector get_users(bool check_is_used = false) const;

        //===
        // CONTROL DEPS
        //===

        /// Get control dependencies registered on the node
        const std::set<std::shared_ptr<Node>>& get_control_dependencies() const;

        void add_control_dependency(std::shared_ptr<Node> node);

        void remove_control_dependency(std::shared_ptr<Node> node)
        {
            m_control_dependencies.erase(node);
        }

        //===
        // EMBEDDED FUNCTIONS
        //===
        virtual std::vector<std::shared_ptr<Function>> get_functions() const;

        //===
        // PLACEMENT
        //===

        /// Get device placement
        Placement get_placement() const;

        /// Set device placement
        void set_placement(Placement placement);

        /// Get device placement
        size_t get_placement_index() const;

        /// Set device placement
        void set_placement_index(size_t placement);

        static const size_t placement_invalid = -1;

        //===
        // LIVENESS
        //===
        const std::unordered_set<descriptor::Tensor*>& get_liveness_new_list() const
        {
            return m_liveness_new_list;
        }
        void set_liveness_new_list(const std::unordered_set<descriptor::Tensor*>& list)
        {
            m_liveness_new_list = list;
        }
        const std::unordered_set<descriptor::Tensor*>& get_liveness_free_list() const
        {
            return m_liveness_free_list;
        }
        void set_liveness_free_list(const std::unordered_set<descriptor::Tensor*>& list)
        {
            m_liveness_free_list = list;
        }

        //===
        // COPYING
        //===
        // TODO(amprocte): make this a pure virtual once copy_with_new_args is sunsetted
        virtual std::shared_ptr<Node> copy_with_new_source_outputs(const std::vector<NodeOutput>& new_source_outputs) const;

        // DEPRECATED
        virtual std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const = 0;

        //===
        // DEFAULT VALUES FOR ZERODIMELIMINATION
        //===
        virtual std::shared_ptr<Node> get_default_value() const { return nullptr; }
    protected:
        Node(const std::string& node_type, const NodeVector& arguments, size_t output_size = 1);
        Node(const std::string& node_type,
             const std::vector<NodeOutput>& arguments,
             size_t output_size = 1);

        /// Throws if the node is invalid.
        virtual void validate_and_infer_types();

        // Called in constructors during transition
        void constructor_validate_and_infer_types();

        std::tuple<element::Type, PartialShape> validate_and_infer_elementwise_args();
        void validate_and_infer_elementwise_arithmetic();
        void validate_and_infer_elementwise_logical();

        void set_output_size(size_t n);

        void set_output_type(size_t i,
                             const element::Type& element_type,
                             const PartialShape& pshape);

    private:
        virtual void generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas) {}
        std::unordered_set<descriptor::Tensor*> m_liveness_new_list;
        std::unordered_set<descriptor::Tensor*> m_liveness_free_list;

        std::set<std::shared_ptr<Node>> m_control_dependencies;

        const std::string m_node_type;
        size_t m_instance_id;
        std::string m_friendly_name;
        const std::string m_unique_name;
        static std::atomic<size_t> s_next_instance_id;
        std::deque<descriptor::Input> m_inputs;
        std::deque<descriptor::Output> m_outputs;
        std::unordered_map<Node*, autodiff::Adjoints> m_adjoint_map;
        Placement m_placement = Placement::DEFAULT;
        size_t m_placement_index = placement_invalid;

        Node(const Node&) = delete;
        Node(Node&&) = delete;
        Node& operator=(const Node&) = delete;
    };

    //===
    // UTIL FOR PRETTY PRINTING
    //===
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

    //===
    // UTIL FOR VALIDATION
    //===
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

    std::string node_validation_assertion_string(const Node* node);

    const std::shared_ptr<Node>& check_single_output_arg(const std::shared_ptr<Node>& node,
                                                         size_t i);
    void check_new_args_count(const Node* node, const NodeVector& new_args);

} // namespace ngraph

#define NODE_VALIDATION_ASSERT(node, cond)                                                         \
    NGRAPH_ASSERT_STREAM_WITH_LOC(                                                                 \
        ::ngraph::NodeValidationError, cond, ::ngraph::node_validation_assertion_string(node))
#define NODE_VALIDATION_FAIL(node)                                                                 \
    NGRAPH_FAIL_STREAM_WITH_LOC(::ngraph::NodeValidationError,                                     \
                                ::ngraph::node_validation_assertion_string(node))
