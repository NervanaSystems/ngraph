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

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/check.hpp"
#include "ngraph/deprecated.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/placement.hpp"

namespace ngraph
{
    template <typename NodeType>
    class Input;

    template <typename NodeType>
    class Output;

    namespace op
    {
        class Constant;
    } // namespace op

    std::string node_validation_failure_loc_string(const Node* node);

    const std::shared_ptr<Node>& check_single_output_arg(const std::shared_ptr<Node>& node,
                                                         size_t i);
    const NodeVector& check_single_output_args(const NodeVector& args);

    const std::shared_ptr<Node>& check_single_output_arg(const std::shared_ptr<Node>& node,
                                                         size_t i);
    const NodeVector& check_single_output_args(const NodeVector& args);

    /// Nodes are the backbone of the graph of Value dataflow. Every node has
    /// zero or more nodes as arguments and one value, which is either a tensor
    /// or a (possibly empty) tuple of values.
    class Node : public std::enable_shared_from_this<Node>
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

    protected:
        /// Throws if the node is invalid.
        virtual void validate_and_infer_types();

        // Called in constructors during transition
        void constructor_validate_and_infer_types();

        std::tuple<element::Type, PartialShape> validate_and_infer_elementwise_args();
        void validate_and_infer_elementwise_arithmetic();
        void validate_and_infer_elementwise_logical();

        Node(const std::string& node_type, const NodeVector& arguments, size_t output_size = 1);

        virtual void generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas) {}
    public:
        virtual ~Node();
        void revalidate_and_infer_types() { validate_and_infer_types(); }
        // Called after transition
        void delayed_validate_and_infer_types();

        /// \brief Produce a vector of constant nodes (one for each of this node's outputs) that
        ///        can replace this node's outputs. May return an empty vector to signal that
        ///        conversion to constants is not possible or not supported.
        /// \returns If conversion is successful, a vector of op::Constant nodes, corresponding
        ///          to this node's outputs in order. If unsuccessful, an empty vector.
        ///
        /// Conversion does not have to be complete. That means that subclasses *may* override
        /// as_constants, but do not have to. It is allowed for as_constants to return an empty
        /// vector even in cases where the output values are statically computable. Thus, any user
        /// of as_constants must allow for the possibility that conversion will fail (i.e.,
        /// as_constants will return {}).
        ///
        /// Conversion must be sound. That means that if as_constants returns a non-empty vector,
        /// the value of each constant in the vector must be exactly the value that would have
        /// been returned for the corresponding output at runtime.
        virtual std::vector<std::shared_ptr<op::Constant>> as_constants() const { return {}; }
        /// \brief Get the string name for the type of the node, such as `Add` or `Multiply`.
        ///        The class name, must not contain spaces as it is used for codegen.
        /// \returns A const reference to the node's type name
        const std::string& description() const;

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
            Node* n = node.get();
            return std::type_index(typeid(*this)) == std::type_index(typeid(*n));
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

        bool is_parameter() const;
        virtual bool is_output() const;
        virtual bool is_constant() const;
        virtual bool is_null() const { return false; }
        virtual bool is_op() const { return false; }
        virtual bool is_commutative() { return false; }
        virtual bool is_dynamic() const;
        size_t get_instance_id() const { return m_instance_id; }
        friend std::ostream& operator<<(std::ostream&, const Node&);
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
        const std::set<std::shared_ptr<Node>>& get_control_dependencies() const;

        void add_control_dependency(std::shared_ptr<Node> node);

        void remove_control_dependency(std::shared_ptr<Node> node)
        {
            m_control_dependencies.erase(node);
        }

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
        const std::set<descriptor::Input*>& get_output_inputs(size_t i) const
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

        /// Get device placement
        size_t get_placement_index() const;

        /// Set device placement
        void set_placement_index(size_t placement);

        const std::unordered_set<std::string>& get_provenance_tags() const;
        void add_provenance_tag(const std::string& tag);
        void remove_provenance_tag(const std::string& tag);

        // to be used when nodes are replaced
        void merge_provenance_tags_from(const std::shared_ptr<const Node>& source);

        /// Get all the nodes that uses the current node
        NodeVector get_users(bool check_is_used = false) const;

        virtual std::shared_ptr<Node> get_default_value() const { return nullptr; }
        /// Use instance ids for comparison instead of memory addresses to improve determinism
        bool operator<(const Node& other) const { return m_instance_id < other.m_instance_id; }
        static const size_t placement_invalid = -1;

        /// \return A vector containing a handle for each of this node's inputs, in order.
        // TODO: Rename to get_inputs()?
        std::vector<Input<Node>> inputs();

        /// \return A vector containing a handle for each of this node's inputs, in order.
        std::vector<Input<const Node>> inputs() const;

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

        /// \return A handle to the `output_index`th output of this node.
        /// \throw std::out_of_range if the node does not have at least `output_index+1` outputs.
        Output<Node> output(size_t output_index);

        /// \return A handle to the `output_index`th output of this node.
        /// \throw std::out_of_range if the node does not have at least `output_index+1` outputs.
        Output<const Node> output(size_t output_index) const;

    protected:
        void set_output_size(size_t n);

    private:
        std::set<std::shared_ptr<Node>> m_control_dependencies;

        const std::string m_node_type;
        size_t m_instance_id;
        std::string m_friendly_name;
        const std::string m_unique_name;
        static std::atomic<size_t> m_next_instance_id;
        std::unordered_set<std::string> m_provenance_tags;
        std::deque<descriptor::Input> m_inputs;
        std::deque<descriptor::Output> m_outputs;
        std::unordered_map<Node*, autodiff::Adjoints> m_adjoint_map;
        Placement m_placement = Placement::DEFAULT;
        size_t m_placement_index = placement_invalid;
    };

    /// \brief A handle for one of a node's inputs.
    template <typename NodeType>
    class Input
    {
    public:
        /// \brief Constructs a Input.
        /// \param node Pointer to the node for the input handle.
        /// \param index The index of the input.
        Input(NodeType* node, size_t index)
            : m_node(node)
            , m_index(index)
        {
        }

        /// \return A pointer to the node referenced by this input handle.
        NodeType* get_node() const { return m_node; }
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
        NodeType* const m_node;
        const size_t m_index;
    };

    /// \brief A handle for one of a node's outputs.
    template <typename NodeType>
    class Output
    {
    public:
        /// \brief Constructs a Output.
        /// \param node A pointer to the node for the output handle.
        /// \param index The index of the output.
        Output(NodeType* node, size_t index)
            : m_node(node)
            , m_index(index)
        {
        }

        /// \brief Constructs a Output.
        /// \param node A `shared_ptr` to the node for the output handle.
        /// \param index The index of the output.
        ///
        /// TODO: Make a plan to deprecate this.
        Output(const std::shared_ptr<NodeType>& node, size_t index)
            : m_node(node.get())
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

        /// \return A pointer to the node referred to by this output handle.
        NodeType* get_node() const { return m_node; }
        /// \return A `shared_ptr` to the node referred to by this output handle.
        ///
        /// TODO: Make a plan to deprecate this.
        std::shared_ptr<NodeType> get_node_shared_ptr() const { return m_node->shared_from_this(); }
        /// \return The index of the output referred to by this output handle.
        size_t get_index() const { return m_index; }
        /// \return A reference to the tensor descriptor for this output.
        descriptor::Tensor& get_tensor() const
        {
            return m_node->m_outputs.at(m_index).get_tensor();
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
        NodeType* const m_node;
        const size_t m_index;
    };

    inline Input<Node> Node::input(size_t input_index)
    {
        if (input_index >= m_inputs.size())
        {
            throw std::out_of_range("node input index is out of range");
        }

        return Input<Node>(this, input_index);
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

    template <typename NodeType>
    Output<Node> Input<NodeType>::get_source_output() const
    {
        auto& output_descriptor = m_node->m_inputs.at(m_index).get_output();
        return Output<Node>(output_descriptor.get_node(), output_descriptor.get_index());
    }

    template <typename NodeType>
    void Input<NodeType>::replace_source_output(const Output<Node>& new_source_output) const
    {
        m_node->m_inputs.at(m_index).replace_output(new_source_output.get_node_shared_ptr(),
                                                    new_source_output.get_index());
    }

    template <typename NodeType>
    std::set<Input<Node>> Output<NodeType>::get_target_inputs() const
    {
        std::set<Input<Node>> result;

        for (auto& input : m_node->m_outputs.at(m_index).get_inputs())
        {
            result.emplace(input->get_raw_pointer_node(), input->get_index());
        }

        return result;
    }

    template <typename NodeType>
    void Output<NodeType>::remove_target_input(const Input<Node>& target_input) const
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

    void check_new_args_count(const Node* node, const NodeVector& new_args);
} // namespace ngraph

#define NODE_VALIDATION_CHECK(node, cond, ...)                                                     \
    NGRAPH_CHECK_HELPER(::ngraph::NodeValidationFailure, (node), (cond), ##__VA_ARGS__)
