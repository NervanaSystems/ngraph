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

#include <memory>
#include <sstream>
#include <typeindex>
#include <typeinfo>

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
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
        for (descriptor::Output& output : arg->get_outputs())
        {
            m_inputs.emplace_back(this, i++, output);
        }
    }
    set_output_size(output_size);
}

// While we are still doing validation and type inference in the constructor, this is true
// It can be set to false to debug doing validation/inference after construction. When that
// is working, these two functions will be removed.
static bool in_transition = true;

void Node::constructor_validate_and_infer_types()
{
    if (in_transition)
    {
        validate_and_infer_types();
    }
}

void Node::delayed_validate_and_infer_types()
{
    if (!in_transition)
    {
        validate_and_infer_types();
    }
}

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

const std::string& Node::get_friendly_name() const
{
    if (m_name.empty())
    {
        return m_unique_name;
    }
    return m_name;
}

const std::string& Node::get_name() const
{
    return m_unique_name;
}

void Node::set_name(const string& name)
{
    if (m_name.empty())
    {
        m_name = name;
    }
    else
    {
        throw ngraph_error("Node name may be set exactly once");
    }
}

Placement Node::get_placement() const
{
    return m_placement;
}

void Node::set_placement(Placement placement)
{
    m_placement = placement;
}

size_t Node::get_placement_size() const
{
    return m_placement_size;
}

void Node::set_placement(size_t placement)
{
    m_placement_size = placement;
}

std::shared_ptr<Node> Node::get_argument(size_t index) const
{
    for (auto& i : get_inputs())
    {
        NGRAPH_ASSERT(i.get_output().get_node()->get_outputs().size() == 1)
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
    for (auto& i : get_inputs())
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
            << arg->get_output_partial_shape(0) << "";
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
    return get_output_tensor_ptr(0);
}

const std::set<descriptor::Input*>& Node::get_output_inputs(size_t i) const
{
    return m_outputs.at(i).get_inputs();
}

descriptor::Tensor& Node::get_output_tensor(size_t i) const
{
    return m_outputs.at(i).get_tensor();
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

descriptor::Input* Node::get_input_from(const shared_ptr<Node>& src)
{
    for (size_t i = 0; i < this->get_input_size(); ++i)
    {
        if (this->get_argument(i) == src)
        {
            return &(this->get_inputs().at(i));
        }
    }
    throw ngraph_error("Error: src is not one of self's input Node");
}

descriptor::Output* Node::get_output_to(const shared_ptr<Node>& dst)
{
    for (size_t i = 0; i < dst->get_input_size(); ++i)
    {
        if (dst->get_argument(i).get() == this)
        {
            return &(dst->get_inputs().at(i).get_output());
        }
    }
    throw ngraph_error("Error: dst is not one of self's output Node");
}

NodeVector Node::get_users() const
{
    NodeVector result;

    for (size_t i = 0; i < get_output_size(); ++i)
    {
        for (auto input : get_output_inputs(i))
        {
            result.push_back(input->get_node());
        }
    }

    return result;
}

std::string ngraph::node_validation_assertion_string(const Node* node)
{
    std::stringstream ss;
    ss << "While validating node '" << *node << "' of type '" << node->description() << "'";
    return ss.str();
}

void ngraph::check_new_args_count(const Node* node, const NodeVector& new_args)
{
    NODE_VALIDATION_ASSERT(node, new_args.size() == node->get_arguments().size())
        << "copy_with_new_args() expected " << node->get_arguments().size() << " argument"
        << (node->get_arguments().size() == 1 ? "" : "s") << " but got " << new_args.size();
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
            NODE_VALIDATION_ASSERT(
                this, element::Type::merge(element_type, element_type, get_input_element_type(i)))
                << "Argument element types are inconsistent.";

            NODE_VALIDATION_ASSERT(this,
                                   PartialShape::merge_into(pshape, get_input_partial_shape(i)))
                << "Argument shapes are inconsistent.";
        }
    }

    return std::make_tuple(element_type, pshape);
}

void Node::validate_and_infer_elementwise_arithmetic()
{
    auto args_et_pshape = validate_and_infer_elementwise_args();
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_ASSERT(this, args_et.is_dynamic() || args_et != element::boolean)
        << "Arguments cannot have boolean element type (argument element type: " << args_et << ").";

    set_output_type(0, args_et, args_pshape);
}

void Node::validate_and_infer_elementwise_logical()
{
    auto args_et_pshape = validate_and_infer_elementwise_args();
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_ASSERT(this, args_et.is_dynamic() || args_et == element::boolean)
        << "Operands for logical operators must have boolean element type but have element type "
        << args_et << ".";

    set_output_type(0, element::boolean, args_pshape);
}

bool Node::validate_punt_if_dynamic()
{
    bool any_dynamic = false;

    for (auto& input : m_inputs)
    {
        any_dynamic |= input.get_partial_shape().is_dynamic();
        any_dynamic |= input.get_element_type().is_dynamic();
    }

    if (any_dynamic)
    {
        for (size_t i = 0; i < get_output_size(); i++)
        {
            set_output_type(i, element::dynamic, PartialShape::dynamic());
        }
        return true;
    }
    else
    {
        return false;
    }
}
