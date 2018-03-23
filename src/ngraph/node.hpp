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

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/placement.hpp"
#include "ngraph/type/type.hpp"

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
        Node(const std::string& node_type, const NodeVector& arguments);
        virtual ~Node()
        {
            for (auto arg : m_arguments)
            {
                arg->m_users.erase(this);
            }
            for (auto& input : m_inputs)
            {
                input.get_output().remove_input(&input);
            }
        }
        virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const std::shared_ptr<Node>& delta)
        {
        }

    public:
        /// The class name, must not contain spaces
        std::string description() const { return m_node_type; }
        const std::string& get_friendly_name() const;
        const std::string& get_name() const;
        void set_name(const std::string& name);
        void clear_arguments() { m_arguments.clear(); }
        const std::multiset<Node*>& users() const { return m_users; }
        /// Return true if this has the same implementing class as node. This
        /// will be used by the pattern matcher when comparing a pattern
        /// graph against the graph.
        bool is_same_op_type(const std::shared_ptr<Node>& node) const
        {
            Node* n = node.get();
            return std::type_index(typeid(*this)) == std::type_index(typeid(*n));
        }

        // Set the value type if it has not already been set; otherwise, ensure that
        // value_type agrees with the value type that was set.
        // This is used when the framework specifies a value type for the value, and we
        // independently compute what we thing the value type should be from the arguments.
        void set_value_type_checked(const std::shared_ptr<const TensorViewType>& value_type);
        void set_value_type_checked(const element::Type& element_type, const Shape& shape);

        bool is_parameter() const;
        virtual bool is_output() const;
        virtual bool is_constant() const;
        virtual bool is_commutative() { return false; }
        size_t get_instance_id() const { return m_instance_id; }
        friend std::ostream& operator<<(std::ostream&, const Node&);

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

        /// Returns the tensor for output i
        descriptor::Tensor& get_output_tensor(size_t i) const;

        /// Checks that there is exactly one output and returns its tensor.
        descriptor::Tensor& get_output_tensor() const;

        /// Returns the tensor view of output i
        std::shared_ptr<descriptor::TensorView> get_output_tensor_view(size_t i) const;

        /// Checks that there is exactly one output and returns its tensor view.
        std::shared_ptr<descriptor::TensorView> get_output_tensor_view() const;

        /// Returns the set of inputs using output i
        const std::set<descriptor::Input*>& get_output_inputs(size_t i) const;

        /// Returns the number of inputs for the op
        size_t get_input_size() const;

        /// Returns the element type of input i
        const element::Type& get_input_element_type(size_t i) const;

        /// Returns the shape of input i
        const Shape& get_input_shape(size_t i) const;

        std::unordered_set<descriptor::Tensor*> liveness_live_list;
        std::unordered_set<descriptor::Tensor*> liveness_new_list;
        std::unordered_set<descriptor::Tensor*> liveness_free_list;

        std::shared_ptr<Node> backprop_node(const std::shared_ptr<Node>& x,
                                            const std::shared_ptr<Node>& c);

        virtual NodeVector get_input_ops(); //const;

        std::shared_ptr<Node> get_input_op(size_t index);

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

    protected:
        void add_output(const element::Type& element_type, const Shape& shape);

        std::string m_node_type;
        std::multiset<Node*> m_users;
        size_t m_instance_id;
        std::string m_name;
        const std::string m_unique_name;
        static std::atomic<size_t> m_next_instance_id;
        std::deque<descriptor::Input> m_inputs;
        std::deque<descriptor::Output> m_outputs;
        std::unordered_map<Node*, autodiff::Adjoints> m_adjoint_map;
        Placement m_placement = Placement::DEFAULT;

    private:
        NodeVector m_arguments;
        //m_arguments still needs to be kept in sync with i/o since get_input_ops
        //is pretty ubiquitous and might be called after the original graph was modified.
        //get_input_ops uses m_arguments to check if a node view reconstruction from i/o
        //is correct.
        NodeVector& get_arguments_FOR_GRAPH_REWRITE_ONLY() { return m_arguments; }
    };
}
