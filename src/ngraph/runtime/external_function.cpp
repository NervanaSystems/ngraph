// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/abs.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/equal.hpp"
#include "ngraph/ops/get_tuple_element.hpp"
#include "ngraph/ops/less.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/not_equal.hpp"
#include "ngraph/ops/select.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/tuple.hpp"
#include "ngraph/pass/assign_tensors.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/propagate_types.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "ngraph/runtime/eigen/abs.hpp"
#include "ngraph/runtime/eigen/add.hpp"
#include "ngraph/runtime/eigen/concat_matrix.hpp"
#include "ngraph/runtime/eigen/concat_vector.hpp"
#include "ngraph/runtime/eigen/constant.hpp"
#include "ngraph/runtime/eigen/copy.hpp"
#include "ngraph/runtime/eigen/divide.hpp"
#include "ngraph/runtime/eigen/dot.hpp"
#include "ngraph/runtime/eigen/equal.hpp"
#include "ngraph/runtime/eigen/less_than.hpp"
#include "ngraph/runtime/eigen/log.hpp"
#include "ngraph/runtime/eigen/matrix_mult.hpp"
#include "ngraph/runtime/eigen/matrix_vector_product.hpp"
#include "ngraph/runtime/eigen/maximum.hpp"
#include "ngraph/runtime/eigen/multiply.hpp"
#include "ngraph/runtime/eigen/negate.hpp"
#include "ngraph/runtime/eigen/not_equal.hpp"
#include "ngraph/runtime/eigen/return.hpp"
#include "ngraph/runtime/eigen/scalar_tensor_product.hpp"
#include "ngraph/runtime/eigen/select.hpp"
#include "ngraph/runtime/eigen/subtract.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/utils.hpp"

using namespace std;
using namespace ngraph::runtime;

ExternalFunction::ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                   bool release_function)
    : m_function(function)
    , m_release_function(release_function)
    , m_is_compiled(false)
    , m_instructions(make_shared<std::vector<std::shared_ptr<ngraph::runtime::Instruction>>>())
{
}

#define REGISTER_INSTRUCTION(op_class, instr_class, ...)                                           \
    op_map[type_index(typeid(op_class))] = [](const Node* n,                                       \
                                              ExternalFunction* ef,                                \
                                              const std::vector<size_t>& in,                       \
                                              const std::vector<size_t>& out) {                    \
        ef->get_instructions()->push_back(make_shared<instr_class>(__VA_ARGS__));                  \
    }

#define REGISTER_UNOP(op_class, instr_class)                                                       \
    REGISTER_INSTRUCTION(op_class, instr_class, in[0], out[0])
#define REGISTER_BINOP(op_class, instr_class)                                                      \
    REGISTER_INSTRUCTION(op_class, instr_class, in[0], in[1], out[0])
#define REGISTER_TERNOP(op_class, instr_class)                                                     \
    REGISTER_INSTRUCTION(op_class, instr_class, in[0], in[1], in[2], out[0])

// Define code generators for handled ops.
ExternalFunction::OpMap& ExternalFunction::get_op_map()
{
    static bool initialized = false;
    static OpMap op_map;
    if (!initialized)
    {
        REGISTER_UNOP(op::Abs, runtime::eigen::AbsInstruction<element::Float32>);
        REGISTER_BINOP(op::Add, runtime::eigen::AddInstruction<element::Float32>);
        REGISTER_BINOP(op::Divide, runtime::eigen::DivideInstruction<element::Float32>);
        REGISTER_BINOP(op::Equal, runtime::eigen::EqualInstruction<element::Float32>);
        REGISTER_BINOP(op::Less, runtime::eigen::LessThanInstruction<element::Float32>);
        REGISTER_UNOP(op::Log, runtime::eigen::LogInstruction<element::Float32>);
        REGISTER_BINOP(op::Maximum, runtime::eigen::MaximumInstruction<element::Float32>);
        REGISTER_BINOP(op::Multiply, runtime::eigen::MultiplyInstruction<element::Float32>);
        REGISTER_UNOP(op::Negative, runtime::eigen::NegateInstruction<element::Float32>);
        REGISTER_BINOP(op::NotEqual, runtime::eigen::NotEqualInstruction<element::Float32>);
        REGISTER_TERNOP(op::Select, runtime::eigen::SelectInstruction<element::Float32>);
        REGISTER_BINOP(op::Subtract, runtime::eigen::SubtractInstruction<element::Float32>);

        REGISTER_INSTRUCTION(
            op::ScalarConstant<element::Float32>,
            runtime::eigen::ConstantInstruction<element::Float32>,
            std::vector<element::Float32::type>{
                dynamic_cast<const op::ScalarConstant<element::Float32>*>(n)->get_value()},
            out[0]);

        REGISTER_INSTRUCTION(
            op::TensorConstant<element::Float32>,
            runtime::eigen::ConstantInstruction<element::Float32>,
            dynamic_cast<const op::TensorConstant<element::Float32>*>(n)->get_value()->get_vector(),
            out[0]);

        op_map[type_index(typeid(op::Concat))] = [](const Node* n,
                                                    ExternalFunction* ef,
                                                    const std::vector<size_t>& in,
                                                    const std::vector<size_t>& out) {
            auto result_tensor_type =
                dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
            assert(nullptr != result_tensor_type);

            auto result_shape = result_tensor_type->get_shape();

            if (result_shape.size() == 1)
            {
                ef->get_instructions()->push_back(
                    make_shared<runtime::eigen::ConcatVectorInstruction<element::Float32>>(in,
                                                                                           out[0]));
            }
            else if (result_shape.size() == 2)
            {
                ef->get_instructions()->push_back(
                    make_shared<runtime::eigen::ConcatMatrixInstruction<element::Float32>>(
                        in,
                        (dynamic_cast<const op::Concat*>(n))->get_concatenation_axis(),
                        out[0]));
            }
            else
            {
                throw ngraph_error("Concat not implemented for rank>2 in VM yet");
            }
        };

        op_map[type_index(typeid(op::Dot))] = [](const Node* n,
                                                 ExternalFunction* ef,
                                                 const std::vector<size_t>& in,
                                                 const std::vector<size_t>& out) {
            auto& arg_nodes = n->get_arguments();

            assert(arg_nodes.size() == 2);

            auto arg0_tensor_type =
                dynamic_pointer_cast<const TensorViewType>(arg_nodes.at(0)->get_value_type());
            assert(nullptr != arg0_tensor_type);

            auto arg1_tensor_type =
                dynamic_pointer_cast<const TensorViewType>(arg_nodes.at(1)->get_value_type());
            assert(nullptr != arg1_tensor_type);

            auto arg0_shape = arg0_tensor_type->get_shape();
            auto arg1_shape = arg1_tensor_type->get_shape();

            // If arg0 or arg1 is a scalar, emit a scalar-tensor product.
            if (arg0_shape.size() == 0)
            {
                ef->get_instructions()->push_back(
                    make_shared<runtime::eigen::ScalarTensorProductInstruction<element::Float32>>(
                        in[0], in[1], out[0]));
            }
            else if (arg1_shape.size() == 0)
            {
                // If arg1 is the scalar, do the same thing but switch the order of operands.
                ef->get_instructions()->push_back(
                    make_shared<runtime::eigen::ScalarTensorProductInstruction<element::Float32>>(
                        in[1], in[0], out[0]));
            }

            // If arg0 and arg1 are both vectors, emit a dot product.
            else if (arg0_shape.size() == 1 && arg1_shape.size() == 1)
            {
                ef->get_instructions()->push_back(
                    make_shared<runtime::eigen::DotInstruction<element::Float32>>(
                        in[0], in[1], out[0]));
            }

            // If arg0 is a matrix and arg1 is a vector, emit a matrix-vector product.
            else if (arg0_shape.size() == 2 && arg1_shape.size() == 1)
            {
                ef->get_instructions()->push_back(
                    make_shared<runtime::eigen::MatrixVectorProductInstruction<element::Float32>>(
                        in[0], in[1], out[0]));
            }

            // If arg0 and arg1 are both matrices, emit a matrix product.
            else if (arg0_shape.size() == 2 && arg1_shape.size() == 2)
            {
                ef->get_instructions()->push_back(
                    make_shared<runtime::eigen::MatrixMultInstruction<element::Float32>>(
                        in[0], in[1], out[0]));
            }

            else
            {
                throw ngraph_error("Dot product for tensors with rank>2 not implemented yet.");
            }
        };

        // Parameter is a "runtime no-op" because the output tensor has already been filled.
        op_map[type_index(typeid(op::Parameter))] = [](const Node* n,
                                                       ExternalFunction* ef,
                                                       const std::vector<size_t>& in,
                                                       const std::vector<size_t>& out) {};

        // GetTupleElement will be spliced out, with the users of out redirected to in's source, but, for now, we need to copy.
        op_map[type_index(typeid(op::GetTupleElement))] = [](const Node* n,
                                                             ExternalFunction* ef,
                                                             const std::vector<size_t>& in,
                                                             const std::vector<size_t>& out) {
            auto get_tuple_element = static_cast<const op::GetTupleElement*>(n);
            ef->get_instructions()->push_back(
                make_shared<runtime::eigen::CopyInstruction<element::Float32>>(
                    in.at(get_tuple_element->get_n()), out.at(0)));
        };

        // Tuple will be spliced out, with the users of out connected to the corresponding in's source, but, for now, we need to copy.
        op_map[type_index(typeid(op::Tuple))] = [](const Node* n,
                                                   ExternalFunction* ef,
                                                   const std::vector<size_t>& in,
                                                   const std::vector<size_t>& out) {
            for (size_t i = 0; i < in.size(); ++i)
            {
                ef->get_instructions()->push_back(
                    make_shared<runtime::eigen::CopyInstruction<element::Float32>>(in.at(i),
                                                                                   out.at(i)));
            }
        };

        initialized = true;
    }
    return op_map;
}

void ExternalFunction::compile()
{
    if (m_is_compiled)
    {
        return;
    }

    // This will be replaced with the pass manager
    // Get the ordered list of ops in execution order
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::TopologicalSort>();
    pass_manager.register_pass<pass::PropagateTypes>();
    pass_manager.register_pass<pass::AssignTensors>();
    pass_manager.run_passes(m_function);

    // Determine tensor requirements for  the call frame
    unordered_map<shared_ptr<ngraph::descriptor::TensorView>, size_t> tensor_index;
    // First come the function inputs
    for (auto param : m_function->get_parameters())
    {
        for (const descriptor::Output& output : param->get_outputs())
        {
            auto tv = output.get_tensor_view();
            size_t index = tensor_index.size();
            tensor_index[tv] = index;
        }
    }
    m_n_inputs = tensor_index.size();

    // Next are the function outputs
    for (const descriptor::Output& output : m_function->get_result()->get_outputs())
    {
        auto tv = output.get_tensor_view();
        size_t index = tensor_index.size();
        tensor_index[tv] = index;
    }
    m_n_outputs = tensor_index.size() - m_n_inputs;

    // All remaining tensor views
    for (const Node* node : pass_manager.get_call_graph())
    {
        for (const descriptor::Output& output : node->get_outputs())
        {
            auto tv = output.get_tensor_view();
            if (0 == tensor_index.count(tv))
            {
                size_t index = tensor_index.size();
                tensor_index[tv] = index;
                m_temp_views.push_back(tv);
            }
        }
    }

    // Now we build the eigen-VM instructions
    auto op_map = get_op_map();
    for (const Node* node : pass_manager.get_call_graph())
    {
        auto handler_it = op_map.find(type_index(typeid(*node)));
        if (handler_it == op_map.end())
        {
            throw ngraph_error("Unhandled op during code generation");
        }
        std::vector<size_t> in;
        for (const descriptor::Input& input : node->get_inputs())
        {
            const descriptor::Output& output = input.get_output();
            auto tv = output.get_tensor_view();
            in.push_back(tensor_index.at(tv));
        }
        std::vector<size_t> out;
        for (const descriptor::Output& output : node->get_outputs())
        {
            auto tv = output.get_tensor_view();
            out.push_back(tensor_index.at(tv));
        }
        handler_it->second(node, this, in, out);
    }
    m_instructions->push_back(make_shared<runtime::eigen::ReturnInstruction>());
    m_is_compiled = true;
    if (m_release_function)
    {
        release_function();
    }
}

shared_ptr<ngraph::runtime::CallFrame> ExternalFunction::make_call_frame()
{
    if (!m_is_compiled)
    {
        compile();
    }
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> temps;
    for (auto tv : m_temp_views)
    {
        temps.push_back(ngraph::runtime::make_tensor<ngraph::element::Float32>(
            tv->get_tensor_view_type()->get_shape()));
    }
    return make_shared<ngraph::runtime::CallFrame>(
        m_n_inputs, m_n_outputs, temps, 0, m_instructions);
}
