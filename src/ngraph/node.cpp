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

#include <memory>
#include <sstream>
#include <typeindex>
#include <typeinfo>

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/placement.hpp"

using namespace std;
using namespace ngraph;

atomic<size_t> Node::m_next_instance_id(0);

Node::Node(const std::string& node_type, const NodeVector& arguments, size_t output_size)
    : m_node_type(node_type)
    , m_instance_id(m_next_instance_id.fetch_add(1))
    , m_unique_name(description() + "_" + to_string(m_instance_id))
{
    // Add this node as a user of each argument.
    size_t i = 0;
    for (auto arg : arguments)
    {
        for (descriptor::Output& output : arg->m_outputs)
        {
            m_inputs.emplace_back(this, i++, output);
        }
    }
    set_output_size(output_size);
}

// While we are still doing validation and type inference in the constructor, this is true
// The #define can be commented out to debug doing validation/inference after construction.
// When that is working, these two functions will be removed.
#define IN_TRANSITION

void Node::constructor_validate_and_infer_types()
{
#ifdef IN_TRANSITION
    validate_and_infer_types();
#endif
}

void Node::delayed_validate_and_infer_types()
{
#ifndef IN_TRANSITION
    validate_and_infer_types();
#endif
}
#undef IN_TRANSITION

void Node::set_output_size(size_t n)
{
    NGRAPH_ASSERT(n >= m_outputs.size()) << "shrinking " << m_outputs.size() << " to " << n;
    for (size_t i = m_outputs.size(); i < n; ++i)
    {
        auto tensor_descriptor = make_shared<descriptor::Tensor>(
            element::dynamic, PartialShape::dynamic(), get_name() + "_" + to_string(i));
        m_outputs.emplace_back(this, i, tensor_descriptor);
    }
}

void Node::validate_and_infer_types()
{
}

void Node::set_input_is_relevant_to_shape(size_t i, bool relevant)
{
    m_inputs.at(i).m_is_relevant_to_shape = relevant;
}

void Node::set_input_is_relevant_to_value(size_t i, bool relevant)
{
    m_inputs.at(i).m_is_relevant_to_value = relevant;
}

void Node::set_output_type(size_t i, const element::Type& element_type, const PartialShape& pshape)
{
    m_outputs.at(i).get_tensor_ptr()->set_tensor_type(element_type, pshape);
}

std::deque<descriptor::Output>& Node::get_outputs()
{
    return m_outputs;
}

const std::deque<descriptor::Output>& Node::get_outputs() const
{
    return m_outputs;
}

bool Node::is_parameter() const
{
    return dynamic_cast<const op::Parameter*>(this) != nullptr;
}

bool Node::is_output() const
{
    return false;
}

bool Node::is_constant() const
{
    return false;
}

const std::string& Node::description() const
{
    return m_node_type;
}

const std::string& Node::get_friendly_name() const
{
    if (m_friendly_name.empty())
    {
        return m_unique_name;
    }
    return m_friendly_name;
}

const std::string& Node::get_name() const
{
    return m_unique_name;
}

void Node::set_friendly_name(const string& name)
{
    m_friendly_name = name;
}

Placement Node::get_placement() const
{
    return m_placement;
}

void Node::set_placement(Placement placement)
{
    m_placement = placement;
}

size_t Node::get_placement_index() const
{
    return m_placement_index;
}

void Node::set_placement_index(size_t placement)
{
    m_placement_index = placement;
}

const std::unordered_set<std::string>& Node::get_provenance_tags() const
{
    return m_provenance_tags;
}

void Node::add_provenance_tag(const std::string& tag)
{
    m_provenance_tags.insert(tag);
}

void Node::remove_provenance_tag(const std::string& tag)
{
    m_provenance_tags.erase(tag);
}

void Node::merge_provenance_tags_from(const std::shared_ptr<const Node>& source)
{
    for (auto& tag : source->get_provenance_tags())
    {
        add_provenance_tag(tag);
    }
}

std::shared_ptr<Node> Node::get_argument(size_t index) const
{
    for (auto& i : m_inputs)
    {
        NGRAPH_ASSERT(i.get_output().get_node()->get_output_size() == 1)
            << "child " << i.get_output().get_node()->get_name() << " has multiple outputs";
    }
    return m_inputs.at(index).get_output().get_node();
}

Node::~Node()
{
    for (auto& input : m_inputs)
    {
        input.get_output().remove_input(&input);
    }
}

NodeVector Node::get_arguments() const
{
    NodeVector result;
    for (auto& i : m_inputs)
    {
        {
            result.push_back(i.get_output().get_node());
        }
    }
    return result;
}

const std::set<std::shared_ptr<Node>>& Node::get_control_dependencies() const
{
    return m_control_dependencies;
}

void Node::add_control_dependency(std::shared_ptr<Node> node)
{
    m_control_dependencies.insert(node);
}

std::vector<std::shared_ptr<Function>> Node::get_functions() const
{
    return std::vector<std::shared_ptr<Function>>{};
}

namespace ngraph
{
    ostream& operator<<(ostream& out, const Node& node)
    {
        return out << NodeDescription(node, false);
    }
}

std::ostream& Node::write_short_description(std::ostream& out) const
{
    return out << get_name();
}

static std::string pretty_element_type(const element::Type& et)
{
    if (et.is_dynamic())
    {
        return "?";
    }
    else
    {
        return et.c_type_string();
    }
}

std::ostream& Node::write_long_description(std::ostream& out) const
{
    out << description() << '[' << get_name() << "](";
    string sep = "";
    for (auto arg : get_arguments())
    {
        out << sep << NodeDescription(*arg, true) << ": "
            << pretty_element_type(arg->get_output_element_type(0))
            << arg->get_output_partial_shape(0);
        sep = ", ";
    }
    out << ") -> (";
    sep = "";
    for (size_t i = 0; i < get_output_size(); i++)
    {
        out << sep << pretty_element_type(get_output_element_type(i))
            << get_output_partial_shape(i);
        sep = ", ";
    }
    out << ")";

    return out;
}

size_t Node::get_output_size() const
{
    return m_outputs.size();
}

const element::Type& Node::get_output_element_type(size_t i) const
{
    return m_outputs.at(i).get_element_type();
}

const element::Type& Node::get_element_type() const
{
    if (get_output_size() != 1)
    {
        throw ngraph_error("get_element_type() must be called on a node with exactly one output.");
    }
    return get_output_element_type(0);
}

const Shape& Node::get_output_shape(size_t i) const
{
    return m_outputs.at(i).get_shape();
}

const PartialShape& Node::get_output_partial_shape(size_t i) const
{
    return m_outputs.at(i).get_partial_shape();
}

const Shape& Node::get_shape() const
{
    if (get_output_size() != 1)
    {
        stringstream es;
        es << "get_shape() must be called on a node with exactly one output (" << description()
           << ")";
        throw ngraph_error(es);
    }
    return get_output_shape(0);
}

shared_ptr<descriptor::Tensor> Node::get_output_tensor_ptr(size_t i) const
{
    return m_outputs.at(i).get_tensor_ptr();
}

shared_ptr<descriptor::Tensor> Node::get_output_tensor_ptr() const
{
    if (get_output_size() != 1)
    {
        throw ngraph_error(
            "get_output_tensor_ptr() must be called on a node with exactly one output.");
    }
    return m_outputs.at(0).get_tensor_ptr();
}

const std::set<descriptor::Input*>& Node::get_output_inputs(size_t i) const
{
    return m_outputs.at(i).get_inputs();
}

descriptor::Tensor& Node::get_output_tensor(size_t i) const
{
    return m_outputs.at(i).get_tensor();
}

const string& Node::get_output_tensor_name(size_t i) const
{
    return m_outputs.at(i).get_tensor().get_name();
}

descriptor::Tensor& Node::get_output_tensor() const
{
    if (get_output_size() != 1)
    {
        throw ngraph_error("get_output_tensor() must be called on a node with exactly one output.");
    }
    return get_output_tensor(0);
}

size_t Node::get_input_size() const
{
    return m_inputs.size();
}

const element::Type& Node::get_input_element_type(size_t i) const
{
    return m_inputs.at(i).get_element_type();
}

const Shape& Node::get_input_shape(size_t i) const
{
    return m_inputs.at(i).get_shape();
}

const PartialShape& Node::get_input_partial_shape(size_t i) const
{
    return m_inputs.at(i).get_partial_shape();
}

const string& Node::get_input_tensor_name(size_t i) const
{
    return m_inputs.at(i).get_tensor().get_name();
}

bool Node::has_same_type(std::shared_ptr<const Node> node) const
{
    if (get_output_size() != node->get_output_size())
    {
        return false;
    }
    for (size_t i = 0; i < get_output_size(); ++i)
    {
        if (get_output_element_type(i) != node->get_output_element_type(i) ||
            get_output_shape(i) != node->get_output_shape(i))
        {
            return false;
        }
    }
    return true;
}

NodeVector Node::get_users(bool check_is_used) const
{
    NodeVector result;

    for (size_t i = 0; i < get_output_size(); ++i)
    {
        for (auto input : m_outputs.at(i).get_inputs())
        {
            if (check_is_used)
            {
                if (is_used(input->get_node().get()))
                {
                    result.push_back(input->get_node());
                }
            }
            else
            {
                result.push_back(input->get_node());
            }
        }
    }

    return result;
}

std::string ngraph::node_validation_assertion_string(const Node* node)
{
    std::stringstream ss;
    ss << "While validating node '" << *node << "'";
    return ss.str();
}

void ngraph::check_new_args_count(const Node* node, const NodeVector& new_args)
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

const std::shared_ptr<Node>& ngraph::check_single_output_arg(const std::shared_ptr<Node>& node,
                                                             size_t i)
{
    NGRAPH_ASSERT(node->get_output_size() == 1) << "Argument " << i << node
                                                << " must produce exactly one value.";
    return node;
}

const NodeVector& ngraph::check_single_output_args(const NodeVector& args)
{
    for (size_t i = 0; i < args.size(); ++i)
    {
        ngraph::check_single_output_arg(args.at(i), i);
    }
    return args;
}

std::tuple<element::Type, PartialShape> Node::validate_and_infer_elementwise_args()
{
    element::Type element_type = get_input_element_type(0);
    PartialShape pshape = get_input_partial_shape(0);

    if (get_input_size() > 1)
    {
        for (size_t i = 1; i < get_input_size(); ++i)
        {
            NODE_VALIDATION_CHECK(
                this,
                element::Type::merge(element_type, element_type, get_input_element_type(i)),
                "Argument element types are inconsistent.");

            NODE_VALIDATION_CHECK(this,
                                  PartialShape::merge_into(pshape, get_input_partial_shape(i)),
                                  "Argument shapes are inconsistent.");
        }
    }

    return std::make_tuple(element_type, pshape);
}

void Node::validate_and_infer_elementwise_arithmetic()
{
    auto args_et_pshape = validate_and_infer_elementwise_args();
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_CHECK(this,
                          args_et.is_dynamic() || args_et != element::boolean,
                          "Arguments cannot have boolean element type (argument element type: ",
                          args_et,
                          ").");

    set_output_type(0, args_et, args_pshape);
}

void Node::validate_and_infer_elementwise_logical()
{
    auto args_et_pshape = validate_and_infer_elementwise_args();
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_CHECK(
        this,
        args_et.is_dynamic() || args_et == element::boolean,
        "Operands for logical operators must have boolean element type but have element type ",
        args_et,
        ".");

    set_output_type(0, element::boolean, args_pshape);
}
