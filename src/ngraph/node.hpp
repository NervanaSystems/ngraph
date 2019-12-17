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
#include <cstring>
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

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/check.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/deprecated.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/op_annotations.hpp"
#include "ngraph/placement.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    template <typename NodeType>
    class Input;

    template <typename NodeType>
    class Output;

    class AttributeVisitor;
    class Variant;
    class Node;
    using NodeVector = std::vector<std::shared_ptr<Node>>;
    using OutputVector = std::vector<Output<Node>>;

    class Function;

    namespace op
    {
        struct AutoBroadcastSpec;
        class Constant;
        class Result;
    } // namespace op

    using ResultVector = std::vector<std::shared_ptr<op::Result>>;

    namespace autodiff
    {
        class Adjoints;
    }

    NGRAPH_API
    std::string node_validation_failure_loc_string(const Node* node);

    const std::shared_ptr<Node>& check_single_output_arg(const std::shared_ptr<Node>& node,
                                                         size_t i);
    NGRAPH_API
    const NodeVector& check_single_output_args(const NodeVector& args);

    const std::shared_ptr<Node>& check_single_output_arg(const std::shared_ptr<Node>& node,
                                                         size_t i);

    NGRAPH_API
    OutputVector as_output_vector(const NodeVector& args);
    NodeVector as_node_vector(const OutputVector& values);
    /// Returns a ResultVector referencing values.
    ResultVector as_result_vector(const OutputVector& values);

    /// Alias useful for cloning
    using NodeMap = std::unordered_map<ngraph::Node*, std::shared_ptr<ngraph::Node>>;

    /// Nodes are the backbone of the graph of Value dataflow. Every node has
    /// zero or more nodes as arguments and one value, which is either a tensor
    /// or a (possibly empty) tuple of values.
    class NGRAPH_API Node : public std::enable_shared_from_this<Node>
    {
        // For access to generate_adjoints.
        friend class autodiff::Adjoints;

        // For access to m_outputs.
        friend class descriptor::Input;

        // For access to m_inputs and m_outputs.
        template <typename NodeType>
        friend class Input;

        // For access to m_outputs.
        template <typename NodeType>
        friend class Output;

    public:
        /// Throws if the node is invalid.
        virtual void validate_and_infer_types();

        // Called in constructors during transition
        void constructor_validate_and_infer_types();

        using type_info_t = DiscreteTypeInfo;

    protected:
        std::tuple<element::Type, PartialShape> validate_and_infer_elementwise_args(
            const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec());
        void validate_and_infer_elementwise_arithmetic(
            const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec());
        void validate_and_infer_elementwise_logical(
            const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec());

        /// \brief Construct an unitialized Node
        Node() {}
        /// \brief Construct an unitialized Node
        /// \param output_size Number of outputs for this node
        Node(size_t output_size);

        /// \brief Constructor for Node subclasses that have metaclasses.
        /// \param arguments Output i will connect to input i
        /// \param output_size Number of outputs for this node
        Node(const OutputVector& arguments, size_t output_size = 1);

        /// \brief Construct a node with arguments. Will be deprecated.
        Node(const std::string& node_type, const NodeVector& arguments, size_t output_size = 1);

        /// \brief Constructor for Node subclasses that have metaclasses. Will be deprecated.
        /// \param arguments The 0th output of node i will connect to input i
        /// \param output_size Number of outputs for this node
        Node(const NodeVector& arguments, size_t output_size = 1);

        // For back-compatibility
        virtual void generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas) {}
        virtual void generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
        {
            generate_adjoints(adjoints, as_node_vector(deltas));
        }
        /// \brief Moves nodes that would be deleted from inputs to nodes to avoid stack overflows
        /// on deep networks.
        void safe_delete(NodeVector& nodes, bool recurse);

    public:
        virtual ~Node();

        virtual bool visit_attributes(AttributeVisitor& visitor) { return false; }
        virtual bool is_unary_elementwise_arithmetic() const { return false; }
        virtual bool is_binary_elementwise_arithmetic() const { return false; }
        virtual bool is_binary_elementwise_comparison() const { return false; }
        virtual bool is_binary_elementwise_logical() const { return false; }
        /// \returns true if node supports autobroadcast operations
        virtual bool supports_auto_broadcast() const { return false; }
        /// \returns the autobroadcasr spec
        virtual const op::AutoBroadcastSpec& get_autob() const;
        /// \returns true if the node can decompose
        virtual bool supports_decompose() const { return false; }
        /// \brief Decomposes the FusedOp into a sub-graph consisting of core ngraph ops
        ///
        /// \return A vector of nodes comprising the sub-graph. The order of output
        ///         tensors must match the match output tensors of the FusedOp
        virtual NodeVector decompose_op() const { return NodeVector(); }
        /// Returns the NodeTypeInfo for the node's class.
        /// During transition to type_info, returns a dummy type_info for Node if the class
        /// has not been updated yet.
        virtual const type_info_t& get_type_info() const = 0;
        const char* get_type_name() const { return get_type_info().name; }
        /// Sets/replaces the arguments with new arguments.
        void set_arguments(const NodeVector& arguments);
        /// Sets/replaces the arguments with new arguments.
        void set_arguments(const OutputVector& arguments);
        /// Sets/replaces the arguments with new arguments.
        void set_argument(size_t position, const Output<Node>& argument);

        /// Sets the number of outputs
        void set_output_size(size_t output_size);

        void revalidate_and_infer_types() { validate_and_infer_types(); }
        // Called after transition
        void delayed_validate_and_infer_types();

        /// \brief Get the string name for the type of the node, such as `Add` or `Multiply`.
        ///        The class name, must not contain spaces as it is used for codegen.
        /// \returns A const reference to the node's type name
        virtual const std::string& description() const;
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

        /// Return true if this has the same implementing class as node. This
        /// will be used by the pattern matcher when comparing a pattern
        /// graph against the graph.
        bool is_same_op_type(const std::shared_ptr<Node>& node) const
        {
            return get_type_info() == node->get_type_info();
        }

        /// \brief Marks an input as being relevant or irrelevant to the output shapes of this
        ///        node.
        /// \param i The index of the input to mark as relevant or irrelevant.
        /// \param relevant true if the input is relevant to output shapes, false otherwise.
        ///
        /// This is used by the shape specialization pass to know which nodes must be statically
        /// evaluated in order to complete shape specialization. (For example, the shape input of
        /// DynReshape must be evaluated statically in order for the output shape to be
        /// determined.) By default, all inputs are marked as shape-irrelevant. Overrides of
        /// validate_and_infer_types should call this function to mark shape-relevant inputs.
        // TODO(amprocte): should be protected
        void set_input_is_relevant_to_shape(size_t i, bool relevant = true);

        /// \brief Marks an input as being relevant or irrelevant to the output values of this
        ///        node.
        /// \param i The index of the input to mark as relevant or irrelevant.
        /// \param relevant true if the input is relevant to output values, false otherwise.
        ///
        /// This is used by the shape specialization pass to cut short evaluation in cases where
        /// an input value does not actually have any effect on the output value of the node. (As
        /// of this writing, the only example of this is ShapeOf.) By default, all inputs are
        /// marked as value-relevant. Overrides of validate_and_infer_types should call this
        /// function to mark value-irrelevant inputs.
        // TODO(amprocte): should be protected
        void set_input_is_relevant_to_value(size_t i, bool relevant = true);

        // TODO(amprocte): should this be protected?
        void set_output_type(size_t i,
                             const element::Type& element_type,
                             const PartialShape& pshape);

        virtual bool is_parameter() const { return false; }
        virtual bool is_output() const;
        virtual bool is_constant() const;
        virtual bool is_null() const { return false; }
        virtual bool is_op() const { return false; }
        virtual bool is_commutative() const { return false; }
        virtual bool is_dynamic() const;
        virtual bool has_state() const { return false; }
        size_t get_instance_id() const { return m_instance_id; }
        friend NGRAPH_API std::ostream& operator<<(std::ostream&, const Node&);
        virtual std::ostream& write_short_description(std::ostream&) const;
        virtual std::ostream& write_long_description(std::ostream&) const;

        std::deque<descriptor::Input>& get_inputs() NGRAPH_DEPRECATED("use inputs() instead")
        {
            return m_inputs;
        }
        const std::deque<descriptor::Input>& get_inputs() const
            NGRAPH_DEPRECATED("use inputs() instead")
        {
            return m_inputs;
        }
        std::deque<descriptor::Output>& get_outputs() NGRAPH_DEPRECATED("use outputs() instead");
        const std::deque<descriptor::Output>& get_outputs() const
            NGRAPH_DEPRECATED("use outputs() instead");

        /// Get control dependencies registered on the node
        const std::vector<std::shared_ptr<Node>>& get_control_dependencies() const;

        /// Get nodes dependent on this node
        const std::vector<Node*>& get_control_dependents() const;

        /// This node cannot execute until node executes
        void add_control_dependency(std::shared_ptr<Node> node);

        /// Remove the dependency of this node on node
        void remove_control_dependency(std::shared_ptr<Node> node);

        /// Remove all dependencies from this node
        void clear_control_dependencies();

        /// Remove this node as a dependency from all dependent nodes
        void clear_control_dependents();

        /// This node absorbs the control dependencies of source_node
        void add_node_control_dependencies(std::shared_ptr<Node> source_node);

        /// This node becomes a dependent of every node dependent on source_node
        void add_node_control_dependents(std::shared_ptr<Node> source_node);

        /// Returns the number of outputs from the node.
        size_t get_output_size() const;

        /// Returns the element type for output i
        // TODO: deprecate in favor of node->output(i).get_element_type()
        const element::Type& get_output_element_type(size_t i) const;

        /// Checks that there is exactly one output and returns its element type
        // TODO: deprecate in favor of node->output(0).get_element_type() with a suitable check in
        // the calling code, or updates to the calling code if it is making an invalid assumption
        // of only one output.
        const element::Type& get_element_type() const;

        /// Returns the shape for output i
        // TODO: deprecate in favor of node->output(i).get_shape()
        const Shape& get_output_shape(size_t i) const;

        /// Returns the partial shape for output i
        const PartialShape& get_output_partial_shape(size_t i) const;

        std::shared_ptr<Node> get_output_as_single_output_node(size_t i,
                                                               bool for_get_output_element = true);

        /// Checks that there is exactly one output and returns its shape
        // TODO: deprecate in favor of node->output(0).get_shape() with a suitable check in the
        // calling code, or updates to the calling code if it is making an invalid assumption of
        // only one output.
        const Shape& get_shape() const;

        /// Returns the tensor for output i
        descriptor::Tensor& get_output_tensor(size_t i) const
            NGRAPH_DEPRECATED("use node->output(i).get_tensor() instead");

        /// Returns the tensor name for output i
        const std::string& get_output_tensor_name(size_t i) const;

        /// Checks that there is exactly one output and returns its tensor.
        descriptor::Tensor& get_output_tensor() const NGRAPH_DEPRECATED(
            "use node->output(0).get_tensor() instead; insert a check that the node has only one "
            "output, or update calling code not to assume only one output");

        /// Returns the tensor of output i
        // TODO: Investigate whether this really needs to be shared_ptr. If so, we'll need a
        // replacement in Output.
        std::shared_ptr<descriptor::Tensor> get_output_tensor_ptr(size_t i) const
            NGRAPH_DEPRECATED("use &node->output(i).get_tensor() instead");

        /// Checks that there is exactly one output and returns its tensor.
        std::shared_ptr<descriptor::Tensor> get_output_tensor_ptr() const NGRAPH_DEPRECATED(
            "use &node->output(i).get_tensor() instead; insert a check that the node has only one "
            "output, or update calling code not to assume only one output");

        /// Returns the set of inputs using output i
        const std::vector<descriptor::Input*>& get_output_inputs(size_t i) const
            NGRAPH_DEPRECATED("use node->output(i).get_target_inputs() instead");

        /// Returns the number of inputs for the op
        size_t get_input_size() const;

        /// Returns the element type of input i
        // TODO: deprecate in favor of node->input(i).get_element_type()
        const element::Type& get_input_element_type(size_t i) const;

        /// Returns the shape of input i
        // TODO: deprecate in favor of node->input(i).get_shape()
        const Shape& get_input_shape(size_t i) const;

        /// Returns the partial shape of input i
        // TODO: deprecate in favor of node->input(i).get_partial_shape()
        const PartialShape& get_input_partial_shape(size_t i) const;

        /// Returns the tensor name for input i
        const std::string& get_input_tensor_name(size_t i) const;

        std::unordered_set<descriptor::Tensor*> liveness_new_list;
        std::unordered_set<descriptor::Tensor*> liveness_free_list;

        // Will be deprecated
        virtual NodeVector get_arguments() const;
        // Will be deprecated
        std::shared_ptr<Node> get_argument(size_t index) const;

    protected:
        // Will be replaced with an OutputVector version
        virtual std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const = 0;

    public:
        std::shared_ptr<Node> copy_with_new_inputs(const OutputVector& new_args) const;

        std::shared_ptr<Node> copy_with_new_inputs(
            const OutputVector& inputs,
            const std::vector<std::shared_ptr<Node>>& control_dependencies) const;

        /// True if this and node have one output with same element type and shape
        bool has_same_type(std::shared_ptr<const Node> node) const;

        /// Get device placement
        Placement get_placement() const;

        /// Set device placement
        void set_placement(Placement placement);

        /// Get device placement
        size_t get_placement_index() const;

        /// Set device placement
        void set_placement_index(size_t placement);

        using RTMap = std::map<std::string, std::shared_ptr<Variant>>;

        RTMap& get_rt_info() { return m_rt_info; }
        const RTMap& get_rt_info() const { return m_rt_info; }
        const std::unordered_set<std::string>& get_provenance_tags() const;
        void add_provenance_tag(const std::string& tag);
        template <typename T>
        void add_provenance_tags(T tag_set)
        {
            for (auto tag : tag_set)
            {
                add_provenance_tag(tag);
            }
        }
        /// \brief Adds tag_set to this node and all intermediate nodes above base
        void add_provenance_tags_above(const OutputVector& base,
                                       const std::unordered_set<std::string>& tag_set);
        void remove_provenance_tag(const std::string& tag);
        /// \brief Add node to additional nodes that receive tags
        void add_provenance_group_member(const std::shared_ptr<Node>& node);
        /// \brief Remove node to additional nodes that receive tags
        void remove_provenance_group_member(const std::shared_ptr<Node>& node);
        /// \brief Replace current_node with replacement_node and transfer tags
        void replace_provenance_group_member(const std::shared_ptr<Node>& current_node,
                                             const std::shared_ptr<Node>& replacement_node);
        /// \return Provenance group nodes
        const std::set<std::shared_ptr<Node>>& get_provenance_group_members() const;

        /// \brief Add all nodes between this node and nodes in base as additional nodes to receive
        /// provenance tags.
        std::shared_ptr<Node> add_provenance_group_members_above(const OutputVector& base);

        // to be used when nodes are replaced
        void merge_provenance_tags_from(const std::shared_ptr<const Node>& source);

        /// Get all the nodes that uses the current node
        NodeVector get_users(bool check_is_used = false) const;

        /// \return Version of this node
        virtual size_t get_version() const { return get_type_info().version; }
        virtual std::shared_ptr<Node> get_default_value() const { return nullptr; }
        /// Use instance ids for comparison instead of memory addresses to improve determinism
        bool operator<(const Node& other) const { return m_instance_id < other.m_instance_id; }
        static const size_t placement_invalid = -1;

        /// \return A vector containing a handle for each of this node's inputs, in order.
        // TODO: Rename to get_inputs()?
        std::vector<Input<Node>> inputs();

        /// \return A vector containing a handle for each of this node's inputs, in order.
        std::vector<Input<const Node>> inputs() const;

        /// \return A vector containing the values for each input
        std::vector<Output<Node>> input_values() const;

        /// \return A vector containing a handle for each of this node's outputs, in order.
        // TODO: Rename to get_outputs()?
        std::vector<Output<Node>> outputs();

        /// \return A vector containing a handle for each of this node's outputs, in order.
        std::vector<Output<const Node>> outputs() const;

        /// \return A handle to the `input_index`th input of this node.
        /// \throw std::out_of_range if the node does not have at least `input_index+1` inputs.
        Input<Node> input(size_t input_index);

        /// \return A handle to the `input_index`th input of this node.
        /// \throw std::out_of_range if the node does not have at least `input_index+1` inputs.
        Input<const Node> input(size_t input_index) const;

        Output<Node> input_value(size_t input_index) const;

        /// \return A handle to the `output_index`th output of this node.
        /// \throw std::out_of_range if the node does not have at least `output_index+1` outputs.
        Output<Node> output(size_t output_index);

        /// \return A handle to the `output_index`th output of this node.
        /// \throw std::out_of_range if the node does not have at least `output_index+1` outputs.
        Output<const Node> output(size_t output_index) const;

        void set_op_annotations(std::shared_ptr<ngraph::op::util::OpAnnotations> op_annotations)
        {
            m_op_annotations = op_annotations;
        }
        std::shared_ptr<ngraph::op::util::OpAnnotations> get_op_annotations() const
        {
            return m_op_annotations;
        }

    private:
        descriptor::Input& get_input_descriptor(size_t position);
        descriptor::Output& get_output_descriptor(size_t position);

        std::vector<Node*> m_control_dependents;
        std::vector<std::shared_ptr<Node>> m_control_dependencies;
        std::string m_node_type;
        size_t m_instance_id{m_next_instance_id.fetch_add(1)};
        std::string m_friendly_name;
        std::string m_unique_name;
        static std::atomic<size_t> m_next_instance_id;
        std::unordered_set<std::string> m_provenance_tags;
        std::set<std::shared_ptr<Node>> m_provenance_group;
        std::deque<descriptor::Input> m_inputs;
        std::deque<descriptor::Output> m_outputs;
        std::unordered_map<Node*, autodiff::Adjoints> m_adjoint_map;
        Placement m_placement = Placement::DEFAULT;
        size_t m_placement_index = placement_invalid;
        std::shared_ptr<ngraph::op::util::OpAnnotations> m_op_annotations;
        std::map<std::string, std::shared_ptr<Variant>> m_rt_info;
    };

    using NodeTypeInfo = Node::type_info_t;

    template <typename NodeType>
    class Input
    {
    };

    template <typename NodeType>
    class Output
    {
    };

    /// \brief A handle for one of a node's inputs.
    template <>
    class Input<Node>
    {
    public:
        /// \brief Constructs a Input.
        /// \param node Pointer to the node for the input handle.
        /// \param index The index of the input.
        Input(Node* node, size_t index)
            : m_node(node)
            , m_index(index)
        {
        }

        /// \return A pointer to the node referenced by this input handle.
        Node* get_node() const { return m_node; }
        /// \return The index of the input referred to by this input handle.
        size_t get_index() const { return m_index; }
        /// \return The element type of the input referred to by this input handle.
        const element::Type& get_element_type() const
        {
            return m_node->get_input_element_type(m_index);
        }
        /// \return The shape of the input referred to by this input handle.
        const Shape& get_shape() const { return m_node->get_input_shape(m_index); }
        /// \return The partial shape of the input referred to by this input handle.
        const PartialShape& get_partial_shape() const
        {
            return m_node->get_input_partial_shape(m_index);
        }
        /// \return A handle to the output that is connected to this input.
        Output<Node> get_source_output() const;
        /// \return A reference to the tensor descriptor for this input.
        descriptor::Tensor& get_tensor() const
        {
            return m_node->m_inputs.at(m_index).get_output().get_tensor();
        }
        /// \return A shared pointer to the tensor descriptor for this input.
        std::shared_ptr<descriptor::Tensor> get_tensor_ptr() const
        {
            return m_node->m_inputs.at(m_index).get_output().get_tensor_ptr();
        }
        /// \return true if this input is relevant to its node's output shapes; else false.
        bool get_is_relevant_to_shapes() const
        {
            return m_node->m_inputs.at(m_index).get_is_relevant_to_shape();
        }
        /// \return true if this input is relevant to its node's output values; else false.
        bool get_is_relevant_to_values() const
        {
            return m_node->m_inputs.at(m_index).get_is_relevant_to_value();
        }

        /// \brief Replaces the source output of this input.
        /// \param new_source_output A handle for the output that will replace this input's source.
        void replace_source_output(const Output<Node>& new_source_output) const;

        bool operator==(const Input& other) const
        {
            return m_node == other.m_node && m_index == other.m_index;
        }
        bool operator!=(const Input& other) const { return !(*this == other); }
        bool operator<(const Input& other) const
        {
            return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
        }
        bool operator>(const Input& other) const
        {
            return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
        }
        bool operator<=(const Input& other) const { return !(*this > other); }
        bool operator>=(const Input& other) const { return !(*this < other); }
    private:
        Node* const m_node;
        const size_t m_index;
    };

    /// \brief A handle for one of a node's inputs.
    template <>
    class NGRAPH_API Input<const Node>
    {
    public:
        /// \brief Constructs a Input.
        /// \param node Pointer to the node for the input handle.
        /// \param index The index of the input.
        Input(const Node* node, size_t index)
            : m_node(node)
            , m_index(index)
        {
        }

        /// \return A pointer to the node referenced by this input handle.
        const Node* get_node() const { return m_node; }
        /// \return The index of the input referred to by this input handle.
        size_t get_index() const { return m_index; }
        /// \return The element type of the input referred to by this input handle.
        const element::Type& get_element_type() const
        {
            return m_node->get_input_element_type(m_index);
        }
        /// \return The shape of the input referred to by this input handle.
        const Shape& get_shape() const { return m_node->get_input_shape(m_index); }
        /// \return The partial shape of the input referred to by this input handle.
        const PartialShape& get_partial_shape() const
        {
            return m_node->get_input_partial_shape(m_index);
        }
        /// \return A handle to the output that is connected to this input.
        Output<Node> get_source_output() const;
        /// \return A reference to the tensor descriptor for this input.
        descriptor::Tensor& get_tensor() const
        {
            return m_node->m_inputs.at(m_index).get_output().get_tensor();
        }
        /// \return A shared pointer to the tensor descriptor for this input.
        std::shared_ptr<descriptor::Tensor> get_tensor_ptr() const
        {
            return m_node->m_inputs.at(m_index).get_output().get_tensor_ptr();
        }
        /// \return true if this input is relevant to its node's output shapes; else false.
        bool get_is_relevant_to_shapes() const
        {
            return m_node->m_inputs.at(m_index).get_is_relevant_to_shape();
        }
        /// \return true if this input is relevant to its node's output values; else false.
        bool get_is_relevant_to_values() const
        {
            return m_node->m_inputs.at(m_index).get_is_relevant_to_value();
        }

        bool operator==(const Input& other) const
        {
            return m_node == other.m_node && m_index == other.m_index;
        }
        bool operator!=(const Input& other) const { return !(*this == other); }
        bool operator<(const Input& other) const
        {
            return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
        }
        bool operator>(const Input& other) const
        {
            return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
        }
        bool operator<=(const Input& other) const { return !(*this > other); }
        bool operator>=(const Input& other) const { return !(*this < other); }
    private:
        const Node* const m_node;
        const size_t m_index;
    };

    /// \brief A handle for one of a node's outputs.
    template <>
    class NGRAPH_API Output<Node>
    {
    public:
        /// \brief Constructs a Output.
        /// \param node A pointer to the node for the output handle.
        /// \param index The index of the output.
        Output(Node* node, size_t index)
            : m_node(node->shared_from_this())
            , m_index(index)
        {
        }

        /// \brief Constructs a Output.
        /// \param node A `shared_ptr` to the node for the output handle.
        /// \param index The index of the output.
        ///
        /// TODO: Make a plan to deprecate this.
        Output(const std::shared_ptr<Node>& node, size_t index)
            : m_node(node)
            , m_index(index)
        {
        }

        /// \brief Constructs a Output, referencing the zeroth output of the node.
        /// \param node A `shared_ptr` to the node for the output handle.
        template <typename T>
        Output(const std::shared_ptr<T>& node)
            : Output(node, 0)
        {
        }

        /// A null output
        Output() = default;

        /// This output position for a different node
        Output<Node> for_node(const std::shared_ptr<Node>& node) { return Output(node, m_index); }
        /// \return A pointer to the node referred to by this output handle.
        Node* get_node() const { return m_node.get(); }
        /// \return A `shared_ptr` to the node referred to by this output handle.
        ///
        /// TODO: Make a plan to deprecate this.
        std::shared_ptr<Node> get_node_shared_ptr() const { return m_node; }
        /// \return A useable shared pointer to this output. If index 0, the node,
        /// otherwise find or create a GOE.
        std::shared_ptr<Node> as_single_output_node(bool for_get_output_element = true) const
            NGRAPH_DEPRECATED("Transitional.")
        {
            return m_node->get_output_as_single_output_node(m_index, for_get_output_element);
        }

        /// \return The index of the output referred to by this output handle.
        size_t get_index() const { return m_index; }
        /// \return A reference to the tensor descriptor for this output.
        descriptor::Tensor& get_tensor() const
        {
            return m_node->m_outputs.at(m_index).get_tensor();
        }
        /// \return A shared point to the tensor ptr for this output.
        std::shared_ptr<descriptor::Tensor> get_tensor_ptr() const
        {
            return m_node->m_outputs.at(m_index).get_tensor_ptr();
        }
        /// \return The element type of the output referred to by this output handle.
        const element::Type& get_element_type() const
        {
            return m_node->get_output_element_type(m_index);
        }
        /// \return The shape of the output referred to by this output handle.
        const Shape& get_shape() const { return m_node->get_output_shape(m_index); }
        /// \return The partial shape of the output referred to by this output handle.
        const PartialShape& get_partial_shape() const
        {
            return m_node->get_output_partial_shape(m_index);
        }

        /// \return A set containing handles for all inputs targeted by the output referenced by
        ///        this output handle.
        std::set<Input<Node>> get_target_inputs() const;

        /// \brief Removes a target input from the output referenced by this output handle.
        /// \param target_input The target input to remove.
        ///
        // TODO(amprocte): Investigate whether this really ought to be public.
        void remove_target_input(const Input<Node>& target_input) const;

        bool operator==(const Output& other) const
        {
            return m_node == other.m_node && m_index == other.m_index;
        }
        bool operator!=(const Output& other) const { return !(*this == other); }
        bool operator<(const Output& other) const
        {
            return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
        }
        bool operator>(const Output& other) const
        {
            return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
        }
        bool operator<=(const Output& other) const { return !(*this > other); }
        bool operator>=(const Output& other) const { return !(*this < other); }
    private:
        std::shared_ptr<Node> m_node;
        size_t m_index{0};
    };

    template <>
    class NGRAPH_API Output<const Node>
    {
    public:
        /// \brief Constructs a Output.
        /// \param node A pointer to the node for the output handle.
        /// \param index The index of the output.
        Output(const Node* node, size_t index)
            : m_node(node->shared_from_this())
            , m_index(index)
        {
        }

        /// \brief Constructs a Output.
        /// \param node A `shared_ptr` to the node for the output handle.
        /// \param index The index of the output.
        ///
        /// TODO: Make a plan to deprecate this.
        Output(const std::shared_ptr<const Node>& node, size_t index)
            : m_node(node)
            , m_index(index)
        {
        }

        /// \brief Constructs a Output, referencing the zeroth output of the node.
        /// \param node A `shared_ptr` to the node for the output handle.
        template <typename T>
        Output(const std::shared_ptr<T>& node)
            : Output(node, 0)
        {
        }

        /// A null output
        Output() = default;

        /// This output position for a different node
        Output<const Node> for_node(const std::shared_ptr<const Node>& node)
        {
            return Output(node, m_index);
        }

        /// \return A pointer to the node referred to by this output handle.
        const Node* get_node() const { return m_node.get(); }
        /// \return A `shared_ptr` to the node referred to by this output handle.
        ///
        /// TODO: Make a plan to deprecate this.
        std::shared_ptr<const Node> get_node_shared_ptr() const { return m_node; }
        /// \return The index of the output referred to by this output handle.
        size_t get_index() const { return m_index; }
        /// \return A reference to the tensor descriptor for this output.
        descriptor::Tensor& get_tensor() const
        {
            return m_node->m_outputs.at(m_index).get_tensor();
        }
        /// \return A shared point to the tensor ptr for this output.
        std::shared_ptr<descriptor::Tensor> get_tensor_ptr() const
        {
            return m_node->m_outputs.at(m_index).get_tensor_ptr();
        }
        /// \return The element type of the output referred to by this output handle.
        const element::Type& get_element_type() const
        {
            return m_node->get_output_element_type(m_index);
        }
        /// \return The shape of the output referred to by this output handle.
        const Shape& get_shape() const { return m_node->get_output_shape(m_index); }
        /// \return The partial shape of the output referred to by this output handle.
        const PartialShape& get_partial_shape() const
        {
            return m_node->get_output_partial_shape(m_index);
        }

        /// \return A set containing handles for all inputs targeted by the output referenced by
        ///        this output handle.
        std::set<Input<Node>> get_target_inputs() const;

        bool operator==(const Output& other) const
        {
            return m_node == other.m_node && m_index == other.m_index;
        }
        bool operator!=(const Output& other) const { return !(*this == other); }
        bool operator<(const Output& other) const
        {
            return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
        }
        bool operator>(const Output& other) const
        {
            return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
        }
        bool operator<=(const Output& other) const { return !(*this > other); }
        bool operator>=(const Output& other) const { return !(*this < other); }
    private:
        std::shared_ptr<const Node> m_node;
        size_t m_index{0};
    };

    inline Input<Node> Node::input(size_t input_index)
    {
        if (input_index >= m_inputs.size())
        {
            throw std::out_of_range("node input index is out of range");
        }

        return Input<Node>(this, input_index);
    }

    inline Output<Node> Node::input_value(size_t input_index) const
    {
        return input(input_index).get_source_output();
    }

    inline Input<const Node> Node::input(size_t input_index) const
    {
        if (input_index >= m_inputs.size())
        {
            throw std::out_of_range("node input index is out of range");
        }

        return Input<const Node>(this, input_index);
    }

    inline Output<Node> Node::output(size_t output_index)
    {
        if (output_index >= m_outputs.size())
        {
            throw std::out_of_range("node output index is out of range");
        }

        return Output<Node>(this, output_index);
    }

    inline Output<const Node> Node::output(size_t output_index) const
    {
        if (output_index >= m_outputs.size())
        {
            throw std::out_of_range("node output index is out of range");
        }

        return Output<const Node>(this, output_index);
    }

    inline Output<Node> Input<Node>::get_source_output() const
    {
        auto& output_descriptor = m_node->m_inputs.at(m_index).get_output();
        return Output<Node>(output_descriptor.get_node(), output_descriptor.get_index());
    }

    inline Output<Node> Input<const Node>::get_source_output() const
    {
        auto& output_descriptor = m_node->m_inputs.at(m_index).get_output();
        return Output<Node>(output_descriptor.get_node(), output_descriptor.get_index());
    }

    inline void Input<Node>::replace_source_output(const Output<Node>& new_source_output) const
    {
        m_node->m_inputs.at(m_index).replace_output(new_source_output.get_node_shared_ptr(),
                                                    new_source_output.get_index());
    }

    inline std::set<Input<Node>> Output<Node>::get_target_inputs() const
    {
        std::set<Input<Node>> result;

        for (auto& input : m_node->m_outputs.at(m_index).get_inputs())
        {
            result.emplace(input->get_raw_pointer_node(), input->get_index());
        }

        return result;
    }

    inline std::set<Input<Node>> Output<const Node>::get_target_inputs() const
    {
        std::set<Input<Node>> result;

        for (auto& input : m_node->m_outputs.at(m_index).get_inputs())
        {
            result.emplace(input->get_raw_pointer_node(), input->get_index());
        }

        return result;
    }

    inline void Output<Node>::remove_target_input(const Input<Node>& target_input) const
    {
        m_node->m_outputs.at(m_index).remove_input(
            &(target_input.get_node()->m_inputs.at(target_input.get_index())));
    }

    inline std::vector<Input<Node>> Node::inputs()
    {
        std::vector<Input<Node>> result;

        for (size_t i = 0; i < get_input_size(); i++)
        {
            result.emplace_back(this, i);
        }

        return result;
    }

    inline std::vector<Output<Node>> Node::input_values() const
    {
        std::vector<Output<Node>> result;

        for (size_t i = 0; i < get_input_size(); i++)
        {
            result.emplace_back(input(i).get_source_output());
        }

        return result;
    }

    inline std::vector<Input<const Node>> Node::inputs() const
    {
        std::vector<Input<const Node>> result;

        for (size_t i = 0; i < get_input_size(); i++)
        {
            result.emplace_back(this, i);
        }

        return result;
    }

    inline std::vector<Output<Node>> Node::outputs()
    {
        std::vector<Output<Node>> result;

        for (size_t i = 0; i < get_output_size(); i++)
        {
            result.emplace_back(shared_from_this(), i);
        }

        return result;
    }

    inline std::vector<Output<const Node>> Node::outputs() const
    {
        std::vector<Output<const Node>> result;

        for (size_t i = 0; i < get_output_size(); i++)
        {
            result.emplace_back(shared_from_this(), i);
        }

        return result;
    }

    class NodeValidationFailure : public CheckFailure
    {
    public:
        NodeValidationFailure(const CheckLocInfo& check_loc_info,
                              const Node* node,
                              const std::string& explanation)
            : CheckFailure(check_loc_info, node_validation_failure_loc_string(node), explanation)
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
}
#define NODE_VALIDATION_CHECK(node, ...)                                                           \
    NGRAPH_CHECK_HELPER(::ngraph::NodeValidationFailure, (node), __VA_ARGS__)

namespace ngraph
{
    template <typename T>
    void check_new_args_count(const Node* node, T new_args)
    {
        NODE_VALIDATION_CHECK(node,
                              new_args.size() == node->get_arguments().size(),
                              "copy_with_new_args() expected ",
                              node->get_arguments().size(),
                              " argument",
                              (node->get_arguments().size() == 1 ? "" : "s"),
                              " but got ",
                              new_args.size());
    }

} // namespace ngraph
