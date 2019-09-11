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

#include "ngraph/op/tensor_iterator.hpp"
#include "ngraph/graph_util.hpp"

using namespace std;
using namespace ngraph;

const string op::TensorIterator::type_name{"TensorIterator"};

op::TensorIterator::TensorIterator(const OutputVector& values)
    : op::util::FusedOp(values)
{
}

op::TensorIterator::InputDescription::InputDescription(
    uint64_t input_index, const std::shared_ptr<Parameter>& body_parameter)
    : m_input_index(input_index)
    , m_body_parameter(body_parameter)
{
}

const op::TensorIterator::SliceInputDescription*
    op::TensorIterator::InputDescription::as_slice() const
{
    return nullptr;
}

const op::TensorIterator::BodyConnectionInputDescription*
    op::TensorIterator::InputDescription::as_body_connection() const
{
    return nullptr;
}

op::TensorIterator::SliceInputDescription::SliceInputDescription(
    uint64_t input_index,
    const std::shared_ptr<Parameter>& body_parameter,
    int64_t start,
    int64_t stride,
    uint64_t part_size,
    int64_t end,
    int64_t axis)
    : InputDescription(input_index, body_parameter)
    , m_start(start)
    , m_stride(stride)
    , m_part_size(part_size)
    , m_end(end)
    , m_axis(axis)
{
}

shared_ptr<op::TensorIterator::InputDescription>
    op::TensorIterator::SliceInputDescription::copy() const
{
    return make_shared<SliceInputDescription>(
        m_input_index, m_body_parameter, m_start, m_stride, m_part_size, m_end, m_axis);
}

const op::TensorIterator::SliceInputDescription*
    op::TensorIterator::SliceInputDescription::as_slice() const
{
    return this;
}

op::TensorIterator::BodyConnectionInputDescription::BodyConnectionInputDescription(
    uint64_t input_index,
    const std::shared_ptr<Parameter>& body_parameter,
    const Output<Node>& body_value)
    : InputDescription(input_index, body_parameter)
    , m_body_value(body_value)
{
}

shared_ptr<op::TensorIterator::InputDescription>
    op::TensorIterator::BodyConnectionInputDescription::copy() const
{
    return make_shared<BodyConnectionInputDescription>(
        m_input_index, m_body_parameter, m_body_value);
}

const op::TensorIterator::BodyConnectionInputDescription*
    op::TensorIterator::BodyConnectionInputDescription::as_body_connection() const
{
    return this;
}

op::TensorIterator::OutputDescription::OutputDescription(const Output<Node>& body_value,
                                                         uint64_t output_index)
    : m_body_value(body_value)
    , m_output_index(output_index)
{
}

const op::TensorIterator::ConcatOutputDescription*
    op::TensorIterator::OutputDescription::as_concat_output_description() const
{
    return nullptr;
}
const op::TensorIterator::BodyOutputDescription*
    op::TensorIterator::OutputDescription::as_body_output_description() const
{
    return nullptr;
}

op::TensorIterator::ConcatOutputDescription::ConcatOutputDescription(const Output<Node>& body_value,
                                                                     uint64_t output_index,
                                                                     int64_t start,
                                                                     int64_t stride,
                                                                     uint64_t part_size,
                                                                     int64_t end,
                                                                     int64_t axis)
    : OutputDescription(body_value, output_index)
    , m_start(start)
    , m_stride(stride)
    , m_part_size(part_size)
    , m_end(end)
    , m_axis(axis)
{
}

shared_ptr<op::TensorIterator::OutputDescription>
    op::TensorIterator::ConcatOutputDescription::copy() const
{
    return make_shared<ConcatOutputDescription>(
        m_body_value, m_output_index, m_start, m_stride, m_part_size, m_end, m_axis);
}

const op::TensorIterator::ConcatOutputDescription*
    op::TensorIterator::ConcatOutputDescription::as_concat_output_description() const
{
    return this;
}

op::TensorIterator::BodyOutputDescription::BodyOutputDescription(const Output<Node>& body_value,
                                                                 uint64_t output_index,
                                                                 int64_t iteration)
    : OutputDescription(body_value, output_index)
    , m_iteration(iteration)
{
}

shared_ptr<op::TensorIterator::OutputDescription>
    op::TensorIterator::BodyOutputDescription::copy() const
{
    return make_shared<BodyOutputDescription>(m_body_value, m_output_index, m_iteration);
}

const op::TensorIterator::BodyOutputDescription*
    op::TensorIterator::BodyOutputDescription::as_body_output_description() const
{
    return this;
}

Input<Node> op::TensorIterator::input_for_value(const Output<Node>& value)
{
    for (auto input : inputs())
    {
        if (input.get_source_output() == value)
        {
            return input;
        }
    }
    auto input_index = get_input_size();
    set_argument(input_index, value);
    return Input<Node>(this, input_index);
}

void op::TensorIterator::set_sliced_input(const std::shared_ptr<op::Parameter>& body_parameter,
                                          const Output<Node>& value,
                                          int64_t start,
                                          int64_t stride,
                                          int64_t part_size,
                                          int64_t end,
                                          int64_t axis)
{
    m_input_descriptions.push_back(make_shared<SliceInputDescription>(
        input_for_value(value).get_index(), body_parameter, start, stride, part_size, end, axis));
}

void op::TensorIterator::set_initialized_input(const std::shared_ptr<Parameter>& body_parameter,
                                               const Output<Node>& initial_value,
                                               const Output<Node>& successive_value)
{
    m_input_descriptions.push_back(make_shared<BodyConnectionInputDescription>(
        input_for_value(initial_value).get_index(), body_parameter, successive_value));
}

Output<Node> op::TensorIterator::get_iter_value(const Output<Node>& body_value, int64_t iteration)
{
    auto output_index = get_output_size();
    m_output_descriptions.push_back(
        make_shared<BodyOutputDescription>(body_value, output_index, iteration));
    return Output<Node>(shared_from_this(), output_index);
}

Output<Node> op::TensorIterator::get_concatenated_slices(const Output<Node>& body_value,
                                                         int64_t start,
                                                         int64_t stride,
                                                         int64_t part_size,
                                                         int64_t end,
                                                         int64_t axis)
{
    auto output_index = get_output_size();
    m_output_descriptions.push_back(make_shared<ConcatOutputDescription>(
        body_value, output_index, start, stride, part_size, end, axis));
    return Output<Node>(shared_from_this(), output_index);
}

NodeVector op::TensorIterator::decompose_op() const
{
    // Stub
    return NodeVector{};
}

#if 0
void op::TensorIterator::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == m_body_parameters.size(),
                          "Number of inputs must be the same as number of body parameters");

    // The number of iterations is determined by the shortest sequence input
    size_t iteration_count{0};
    // If true, iteration count is dynamic
    bool iteration_count_dynamic{false};
    // true when we know something about the count
    bool iteration_count_valid{false};
    for (auto input : inputs())
    {
        size_t input_index = input.get_index();
        Output<Node> value = input.get_source_output();
        PartialShape sequence_shape = value.get_partial_shape();
        PartialShape iterator_shape = sequence_shape;
        Rank sequence_rank = sequence_shape.rank();

        if (sequence_rank.is_dynamic())
        {
            // Can't determine the sequence length
            iteration_count_dynamic = true;
        }
        else
        {
            NODE_VALIDATION_CHECK(this,
                                  static_cast<size_t>(sequence_shape.rank()) != 0,
                                  "Input ",
                                  input_index,
                                  " is specified to be a sequence but is scalar.");
            Dimension sequence_dim = sequence_shape[0];
            vector<Dimension> dimensions = static_cast<vector<Dimension>>(sequence_shape);
            dimensions.erase(dimensions.begin());
            iterator_shape = PartialShape(dimensions);

            if (sequence_dim.is_dynamic())
            {
                // Can't determine the sequence length
                iteration_count_dynamic = true;
            }
            else
            {
                size_t sequence_length = static_cast<size_t>(sequence_dim);
                if (!iteration_count_valid || (sequence_length < iteration_count))
                {
                    iteration_count = sequence_length;
                    iteration_count_valid = true;
                }
            }
        }

        NODE_VALIDATION_CHECK(
            this,
            iterator_shape.compatible(m_body_parameters.at(input_index)->get_partial_shape()),
            "Iterator body param is not compatible with value");
    }
    // The body may depend on the body parameters as well as values from outside the body
    // Body parameters depend on the loop initialization
    NodeVector body_result_nodes;
    for (auto& body_output : m_body_outputs)
    {
        body_result_nodes.push_back(body_output.get_node_shared_ptr());
    }
    std::list<std::shared_ptr<Node>> body_node_closure(topological_sort(body_result_nodes, true));
    std::set<Node*> bound_nodes;
    std::vector<Node*> free_nodes;
    for (auto& parameter : m_body_parameters)
    {
        std::cerr << *this << " Bound: " << *parameter << std::endl;
        bound_nodes.insert(parameter.get());
    }
    for (auto& node : body_node_closure)
    {
        if (bound_nodes.find(node.get()) == bound_nodes.end())
        {
            bool is_free = true;
            for (auto input : node->inputs())
            {
                auto input_node = input.get_source_output().get_node();
                if (bound_nodes.find(input_node) != bound_nodes.end())
                {
                    bound_nodes.insert(node.get());
                    is_free = false;
                    std::cerr << *this << " Bound: "
                              << " : " << *node << std::endl;
                    break;
                }
            }
            if (is_free)
            {
                free_nodes.push_back(node.get());
                std::cout << *this << " Free: " << *node << std::endl;
            }
        }
    }
}
#endif

std::shared_ptr<Node> op::TensorIterator::copy_with_new_args(const NodeVector& new_args) const
{
    auto op = make_shared<op::TensorIterator>(as_output_vector(new_args));
    for (auto& input_description : m_input_descriptions)
    {
        op->m_input_descriptions.push_back(input_description->copy());
    }
    for (auto& output_description : m_output_descriptions)
    {
        op->m_output_descriptions.push_back(output_description->copy());
    }
    return move(op);
}
