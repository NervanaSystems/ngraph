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
#include "ngraph/check.hpp"
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
        class Constant;
        class Parameter;
        class Result;
    } // namespace op

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
    /// or a (possibly empty) tuple of values.
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
        void set_input_is_relevant_to_value(size_t i, bool relevant = true);

        void set_output_type(size_t i,
                             const element::Type& element_type,
                             const PartialShape& pshape);

        bool is_parameter() const;
        virtual bool is_output() const;
        virtual bool is_constant() const;
        virtual bool is_null() const { return false; }
        virtual bool is_op() const { return false; }
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

        /// Returns the partial shape for output i
        const PartialShape& get_output_partial_shape(size_t i) const;

        /// Checks that there is exactly one output and returns its shape
        const Shape& get_shape() const;

        /// Returns the tensor for output i
        descriptor::Tensor& get_output_tensor(size_t i) const;

        /// Checks that there is exactly one output and returns its tensor.
        descriptor::Tensor& get_output_tensor() const;

        /// Returns the tensor of output i
        std::shared_ptr<descriptor::Tensor> get_output_tensor_ptr(size_t i) const;

        /// Checks that there is exactly one output and returns its tensor.
        std::shared_ptr<descriptor::Tensor> get_output_tensor_ptr() const;

        /// Returns the set of inputs using output i
        const std::set<descriptor::Input*>& get_output_inputs(size_t i) const;

        /// Returns the number of inputs for the op
        size_t get_input_size() const;

        /// Returns the element type of input i
        const element::Type& get_input_element_type(size_t i) const;

        /// Returns the shape of input i
        const Shape& get_input_shape(size_t i) const;

        /// Returns the partial shape of input i
        const PartialShape& get_input_partial_shape(size_t i) const;

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

        /// Get input descriptor that is connected to src
        descriptor::Input* get_input_from(const std::shared_ptr<Node>& src);

        /// Get ouput descriptor that outputs to dst
        descriptor::Output* get_output_to(const std::shared_ptr<Node>& dst);

        /// Get all the nodes that uses the current node
        NodeVector get_users(bool check_is_used = false) const;

        virtual std::shared_ptr<Node> get_default_value() const { return nullptr; }
        /// Use instance ids for comparison instead of memory addresses to improve determinism
        bool operator<(const Node& other) const { return m_instance_id < other.m_instance_id; }
        static const size_t placement_invalid = -1;

    protected:
        std::set<std::shared_ptr<Node>> m_control_dependencies;
        void set_output_size(size_t n);

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

    class NodeValidationFailure : public CheckFailure
    {
    public:
        NodeValidationFailure(const CheckLocInfo& check_loc_info,
                              const Node* node,
                              const std::string& explanation)
            : CheckFailure(check_loc_info, node_validation_assertion_string(node), explanation)
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
    NGRAPH_CHECK(::ngraph::NodeValidationFailure, (node), (cond), __VA_ARGS__)
