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

using namespace std;
using namespace ngraph;

const string op::TensorIterator::type_name{"TensorIterator"};

op::TensorIterator::TensorIterator(const OutputVector& body_inputs,
                                   const ParameterVector& body_parameters,
                                   const OutputVector& body_outputs,
                                   const OutputVector& tensor_iterator_outputs,
                                   const std::vector<bool>& sequence_inputs,
                                   const std::vector<bool>& sequence_outputs)
    : Op(body_inputs)
    , m_body_parameters(body_parameters)
    , m_body_outputs(body_outputs)
    , m_tensor_iterator_outputs(tensor_iterator_outputs)
    , m_sequence_inputs(sequence_inputs)
    , m_sequence_outputs(sequence_outputs)
{
    constructor_validate_and_infer_types();
}

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

        if (m_sequence_inputs.at(input_index))
        {
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
        }
        NODE_VALIDATION_CHECK(
            this,
            iterator_shape.compatible(m_body_parameters.at(input_index)->get_partial_shape()),
            "Iterator body param is not compatible with value");
    }
    // The body may depend on the body parameters as well as values from outside the body
    // Body parameters depend on the loop initialization
}

const ParameterVector& op::TensorIterator::get_body_parameters() const
{
    return m_body_parameters;
}

ParameterVector& op::TensorIterator::get_body_parameters()
{
    return m_body_parameters;
}

void op::TensorIterator::set_body_parameters(const ParameterVector& body_parameters)
{
    m_body_parameters = body_parameters;
}

const OutputVector& op::TensorIterator::get_body_outputs() const
{
    return m_body_outputs;
}

OutputVector& op::TensorIterator::get_body_outputs()
{
    return m_body_outputs;
}

void op::TensorIterator::set_body_outputs(const OutputVector& body_outputs)
{
    m_body_outputs = body_outputs;
}

const OutputVector& op::TensorIterator::get_tensor_iterator_outputs() const
{
    return m_tensor_iterator_outputs;
}

OutputVector& op::TensorIterator::get_tensor_iterator_outputs()
{
    return m_tensor_iterator_outputs;
}

void op::TensorIterator::set_tensor_iterator_outputs(const OutputVector& tensor_iterator_outputs)
{
    m_tensor_iterator_outputs = tensor_iterator_outputs;
}

const vector<bool>& op::TensorIterator::get_sequence_inputs() const
{
    return m_sequence_inputs;
}

vector<bool>& op::TensorIterator::get_sequence_inputs()
{
    return m_sequence_inputs;
}

void op::TensorIterator::set_sequence_inputs(const vector<bool>& sequence_inputs)
{
    m_sequence_inputs = sequence_inputs;
}

const vector<bool>& op::TensorIterator::get_sequence_outputs() const
{
    return m_sequence_outputs;
}

vector<bool>& op::TensorIterator::get_sequence_outputs()
{
    return m_sequence_outputs;
}

void op::TensorIterator::set_sequence_outputs(const vector<bool>& sequence_outputs)
{
    m_sequence_outputs = sequence_outputs;
}
std::shared_ptr<Node> op::TensorIterator::copy_with_new_args(const NodeVector& new_args) const
{
    OutputVector output_vector;
    for (auto arg : new_args)
    {
        output_vector.push_back(arg);
    }
    return make_shared<TensorIterator>(output_vector,
                                       m_body_parameters,
                                       m_body_outputs,
                                       m_tensor_iterator_outputs,
                                       m_sequence_inputs,
                                       m_sequence_outputs);
}
