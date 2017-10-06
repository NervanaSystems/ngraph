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
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/abs.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convert.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/equal.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/get_tuple_element.hpp"
#include "ngraph/ops/greater.hpp"
#include "ngraph/ops/greater_eq.hpp"
#include "ngraph/ops/less.hpp"
#include "ngraph/ops/less_eq.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/not_equal.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/select.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/tuple.hpp"
#include "ngraph/pass/assign_tensors.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/propagate_types.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "ngraph/runtime/eigen/abs.hpp"
#include "ngraph/runtime/eigen/add.hpp"
#include "ngraph/runtime/eigen/broadcast_scalar.hpp"
#include "ngraph/runtime/eigen/broadcast_vector_colwise.hpp"
#include "ngraph/runtime/eigen/broadcast_vector_rowwise.hpp"
#include "ngraph/runtime/eigen/call.hpp"
#include "ngraph/runtime/eigen/concat_matrix.hpp"
#include "ngraph/runtime/eigen/concat_vector.hpp"
#include "ngraph/runtime/eigen/constant.hpp"
#include "ngraph/runtime/eigen/convert.hpp"
#include "ngraph/runtime/eigen/copy.hpp"
#include "ngraph/runtime/eigen/divide.hpp"
#include "ngraph/runtime/eigen/dot.hpp"
#include "ngraph/runtime/eigen/equal.hpp"
#include "ngraph/runtime/eigen/greater_eq.hpp"
#include "ngraph/runtime/eigen/greater_than.hpp"
#include "ngraph/runtime/eigen/less_eq.hpp"
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

using ngraph::descriptor::layout::DenseTensorViewLayout;

ExternalFunction::ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                   bool release_function)
    : m_function(function)
    , m_release_function(release_function)
    , m_is_compiled(false)
    , m_instructions(make_shared<std::vector<std::shared_ptr<ngraph::runtime::Instruction>>>())
{
}

#define REGISTER_TO_OP_MAP(op_class)                                                               \
    op_map[type_index(typeid(op_class))] = [](const Node* n,                                       \
                                              ExternalFunction* ef,                                \
                                              FunctionMap& function_map,                           \
                                              const std::vector<TensorViewInfo>& in,               \
                                              const std::vector<TensorViewInfo>& out)

// Suppress Clang's complaints about the ,##__VA_ARGS__ token-pasting hack, which is a GNU extension
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"

#define DO_ON_ELEMENT_TYPE(et, err_msg, macro, ...)                                                \
    {                                                                                              \
        if (et == element::Bool::element_type())                                                   \
        {                                                                                          \
            macro(element::Bool, ##__VA_ARGS__);                                                   \
        }                                                                                          \
        else if (et == element::Float32::element_type())                                           \
        {                                                                                          \
            macro(element::Float32, ##__VA_ARGS__);                                                \
        }                                                                                          \
        else if (et == element::Int8::element_type())                                              \
        {                                                                                          \
            macro(element::Int8, ##__VA_ARGS__);                                                   \
        }                                                                                          \
        else if (et == element::Int32::element_type())                                             \
        {                                                                                          \
            macro(element::Int32, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::Int64::element_type())                                             \
        {                                                                                          \
            macro(element::Int64, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::UInt8::element_type())                                             \
        {                                                                                          \
            macro(element::UInt8, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::UInt32::element_type())                                            \
        {                                                                                          \
            macro(element::UInt32, ##__VA_ARGS__);                                                 \
        }                                                                                          \
        else if (et == element::UInt64::element_type())                                            \
        {                                                                                          \
            macro(element::UInt64, ##__VA_ARGS__);                                                 \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            throw ngraph_error(err_msg);                                                           \
        }                                                                                          \
    }

#define DO_ON_NUMERIC_TYPE(et, err_msg, macro, ...)                                                \
    {                                                                                              \
        if (et == element::Float32::element_type())                                                \
        {                                                                                          \
            macro(element::Float32, ##__VA_ARGS__);                                                \
        }                                                                                          \
        else if (et == element::Int8::element_type())                                              \
        {                                                                                          \
            macro(element::Int8, ##__VA_ARGS__);                                                   \
        }                                                                                          \
        else if (et == element::Int32::element_type())                                             \
        {                                                                                          \
            macro(element::Int32, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::Int64::element_type())                                             \
        {                                                                                          \
            macro(element::Int64, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::UInt8::element_type())                                             \
        {                                                                                          \
            macro(element::UInt8, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::UInt32::element_type())                                            \
        {                                                                                          \
            macro(element::UInt32, ##__VA_ARGS__);                                                 \
        }                                                                                          \
        else if (et == element::UInt64::element_type())                                            \
        {                                                                                          \
            macro(element::UInt64, ##__VA_ARGS__);                                                 \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            throw ngraph_error(err_msg);                                                           \
        }                                                                                          \
    }

#define DO_ON_SIGNED_NUMERIC_TYPE(et, err_msg, macro, ...)                                         \
    {                                                                                              \
        if (et == element::Float32::element_type())                                                \
        {                                                                                          \
            macro(element::Float32, ##__VA_ARGS__);                                                \
        }                                                                                          \
        else if (et == element::Int8::element_type())                                              \
        {                                                                                          \
            macro(element::Int8, ##__VA_ARGS__);                                                   \
        }                                                                                          \
        else if (et == element::Int32::element_type())                                             \
        {                                                                                          \
            macro(element::Int32, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::Int64::element_type())                                             \
        {                                                                                          \
            macro(element::Int64, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            throw ngraph_error(err_msg);                                                           \
        }                                                                                          \
    }

#define REGISTER_INSTRUCTION(op_class, instr_class, ...)                                           \
    REGISTER_TO_OP_MAP(op_class)                                                                   \
    {                                                                                              \
        ef->get_instructions()->push_back(make_shared<instr_class>(__VA_ARGS__));                  \
    }

#define M_REGISTER_SIGNED_NUMERIC_UNOP(T, instr_class)                                             \
    ef->get_instructions()->push_back(make_shared<instr_class<T>>(in[0], out[0]));
#define REGISTER_SIGNED_NUMERIC_UNOP(op_class, instr_class)                                        \
    REGISTER_TO_OP_MAP(op_class)                                                                   \
    {                                                                                              \
        const element::Type& et = (dynamic_pointer_cast<const TensorViewType>(                     \
                                       n->get_arguments().at(0)->get_value_type()))                \
                                      ->get_element_type();                                        \
        DO_ON_SIGNED_NUMERIC_TYPE(                                                                 \
            et,                                                                                    \
            "Internal error: signed numeric unop has unhandled element type",                      \
            M_REGISTER_SIGNED_NUMERIC_UNOP,                                                        \
            instr_class);                                                                          \
    }

#define M_REGISTER_NUMERIC_UNOP(T, instr_class)                                                    \
    ef->get_instructions()->push_back(make_shared<instr_class<T>>(in[0], out[0]));
#define REGISTER_NUMERIC_UNOP(op_class, instr_class)                                               \
    REGISTER_TO_OP_MAP(op_class)                                                                   \
    {                                                                                              \
        const element::Type& et = (dynamic_pointer_cast<const TensorViewType>(                     \
                                       n->get_arguments().at(0)->get_value_type()))                \
                                      ->get_element_type();                                        \
        DO_ON_NUMERIC_TYPE(et,                                                                     \
                           "Internal error: numeric unop has unhandled element type",              \
                           M_REGISTER_NUMERIC_UNOP,                                                \
                           instr_class);                                                           \
    }

#define M_REGISTER_NUMERIC_BINOP(T, instr_class)                                                   \
    ef->get_instructions()->push_back(make_shared<instr_class<T>>(in[0], in[1], out[0]));
#define REGISTER_NUMERIC_BINOP(op_class, instr_class)                                              \
    REGISTER_TO_OP_MAP(op_class)                                                                   \
    {                                                                                              \
        const element::Type& et = (dynamic_pointer_cast<const TensorViewType>(                     \
                                       n->get_arguments().at(0)->get_value_type()))                \
                                      ->get_element_type();                                        \
        DO_ON_NUMERIC_TYPE(et,                                                                     \
                           "Internal error: numeric binop has unhandled element type",             \
                           M_REGISTER_NUMERIC_BINOP,                                               \
                           instr_class);                                                           \
    }

#define M_REGISTER_POLYMORPHIC_BINOP(T, instr_class)                                               \
    ef->get_instructions()->push_back(make_shared<instr_class<T>>(in[0], in[1], out[0]));
#define REGISTER_POLYMORPHIC_BINOP(op_class, instr_class)                                          \
    REGISTER_TO_OP_MAP(op_class)                                                                   \
    {                                                                                              \
        const element::Type& et = (dynamic_pointer_cast<const TensorViewType>(                     \
                                       n->get_arguments().at(0)->get_value_type()))                \
                                      ->get_element_type();                                        \
        DO_ON_ELEMENT_TYPE(et,                                                                     \
                           "Internal error: polymorphic binop has unhandled element type",         \
                           M_REGISTER_POLYMORPHIC_BINOP,                                           \
                           instr_class);                                                           \
    }

// Something sneaky here: note the at(1) instead of at(0).
#define M_REGISTER_POLYMORPHIC_TERNOP(T, instr_class)                                              \
    ef->get_instructions()->push_back(make_shared<instr_class<T>>(in[0], in[1], in[2], out[0]));
#define REGISTER_POLYMORPHIC_TERNOP(op_class, instr_class)                                         \
    REGISTER_TO_OP_MAP(op_class)                                                                   \
    {                                                                                              \
        const element::Type& et = (dynamic_pointer_cast<const TensorViewType>(                     \
                                       n->get_arguments().at(1)->get_value_type()))                \
                                      ->get_element_type();                                        \
        DO_ON_ELEMENT_TYPE(et,                                                                     \
                           "Internal error: polymorphic ternop has unhandled element type",        \
                           M_REGISTER_POLYMORPHIC_TERNOP,                                          \
                           instr_class);                                                           \
    }

#define REGISTER_CONSTANT_INSTRUCTIONS(T)                                                          \
    {                                                                                              \
        REGISTER_INSTRUCTION(                                                                      \
            op::ScalarConstant<T>,                                                                 \
            runtime::eigen::ConstantInstruction<T>,                                                \
            std::vector<T::type>{dynamic_cast<const op::ScalarConstant<T>*>(n)->get_value()},      \
            out[0]);                                                                               \
        REGISTER_INSTRUCTION(                                                                      \
            op::TensorConstant<T>,                                                                 \
            runtime::eigen::ConstantInstruction<T>,                                                \
            std::vector<T::type>{                                                                  \
                dynamic_cast<const op::TensorConstant<T>*>(n)->get_value()->get_vector()},         \
            out[0]);                                                                               \
    }

#define PUSH_INSTRUCTION(T, instr, ...)                                                            \
    {                                                                                              \
        ef->get_instructions()->push_back(make_shared<instr<T>>(__VA_ARGS__));                     \
    }
#define PUSH_POLYMORPHIC_INSTRUCTION(et, err_msg, instr, ...)                                      \
    DO_ON_ELEMENT_TYPE(et, err_msg, PUSH_INSTRUCTION, instr, __VA_ARGS__)
#define PUSH_NUMERIC_POLYMORPHIC_INSTRUCTION(et, err_msg, instr, ...)                              \
    DO_ON_NUMERIC_TYPE(et, err_msg, PUSH_INSTRUCTION, instr, __VA_ARGS__)

// Turn off complaint suppression (see above)
#pragma clang diagnostic pop

// Define code generators for handled ops.
ExternalFunction::OpMap& ExternalFunction::get_op_map()
{
    static bool initialized = false;
    static OpMap op_map;
    if (!initialized)
    {
        REGISTER_NUMERIC_UNOP(op::Log, runtime::eigen::LogInstruction);
        REGISTER_NUMERIC_UNOP(op::Negative, runtime::eigen::NegateInstruction);

        REGISTER_SIGNED_NUMERIC_UNOP(op::Abs, runtime::eigen::AbsInstruction);

        REGISTER_NUMERIC_BINOP(op::Add, runtime::eigen::AddInstruction);
        REGISTER_NUMERIC_BINOP(op::Divide, runtime::eigen::DivideInstruction);
        REGISTER_NUMERIC_BINOP(op::Greater, runtime::eigen::GreaterThanInstruction);
        REGISTER_NUMERIC_BINOP(op::GreaterEq, runtime::eigen::GreaterEqInstruction);
        REGISTER_NUMERIC_BINOP(op::Less, runtime::eigen::LessThanInstruction);
        REGISTER_NUMERIC_BINOP(op::LessEq, runtime::eigen::LessEqInstruction);
        REGISTER_NUMERIC_BINOP(op::Maximum, runtime::eigen::MaximumInstruction);
        REGISTER_NUMERIC_BINOP(op::Multiply, runtime::eigen::MultiplyInstruction);
        REGISTER_NUMERIC_BINOP(op::Subtract, runtime::eigen::SubtractInstruction);

        REGISTER_POLYMORPHIC_BINOP(op::Equal, runtime::eigen::EqualInstruction);
        REGISTER_POLYMORPHIC_BINOP(op::NotEqual, runtime::eigen::NotEqualInstruction);

        REGISTER_POLYMORPHIC_TERNOP(op::Select, runtime::eigen::SelectInstruction);

        REGISTER_CONSTANT_INSTRUCTIONS(element::Bool);
        REGISTER_CONSTANT_INSTRUCTIONS(element::Float32);
        REGISTER_CONSTANT_INSTRUCTIONS(element::Int8);
        REGISTER_CONSTANT_INSTRUCTIONS(element::Int32);
        REGISTER_CONSTANT_INSTRUCTIONS(element::Int64);
        REGISTER_CONSTANT_INSTRUCTIONS(element::UInt8);
        REGISTER_CONSTANT_INSTRUCTIONS(element::UInt32);
        REGISTER_CONSTANT_INSTRUCTIONS(element::UInt64);

        REGISTER_TO_OP_MAP(op::Broadcast)
        {
            auto broadcast = static_cast<const op::Broadcast*>(n);

            auto arg_tensor_type = dynamic_pointer_cast<const TensorViewType>(
                n->get_arguments().at(0)->get_value_type());
            assert(nullptr != arg_tensor_type);

            auto result_tensor_type =
                dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
            assert(nullptr != result_tensor_type);

            auto arg_shape = arg_tensor_type->get_shape();
            auto result_shape = result_tensor_type->get_shape();
            auto& result_element_type = result_tensor_type->get_element_type();

            if (broadcast->get_broadcast_axes().empty())
            {
                PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                             "Broadcast has unhandled element type",
                                             runtime::eigen::CopyInstruction,
                                             in[0].get_index(),
                                             out[0].get_index());
            }
            else if (arg_shape.size() == 0)
            {
                PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                             "Broadcast has unhandled element type",
                                             runtime::eigen::BroadcastScalarInstruction,
                                             in[0],
                                             out[0]);
            }
            else if (arg_shape.size() == 1 && result_shape.size() == 2)
            {
                if (broadcast->get_broadcast_axes() == AxisSet{1})
                {
                    PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                                 "Broadcast has unhandled element type",
                                                 runtime::eigen::BroadcastVectorColwiseInstruction,
                                                 in[0],
                                                 out[0]);
                }
                else if (broadcast->get_broadcast_axes() == AxisSet{0})
                {
                    PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                                 "Broadcast has unhandled element type",
                                                 runtime::eigen::BroadcastVectorRowwiseInstruction,
                                                 in[0],
                                                 out[0]);
                }
                else
                {
                    throw ngraph_error(
                        "Internal error: axis set for vector-matrix broadcast is neither {0} nor "
                        "{1}");
                }
            }
            else
            {
                throw ngraph_error("Broadcast not implemented for rank>2 in VM yet");
            }
        };

        REGISTER_TO_OP_MAP(op::Concat)
        {
            auto result_tensor_type =
                dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
            assert(nullptr != result_tensor_type);

            auto result_shape = result_tensor_type->get_shape();
            auto& result_element_type = result_tensor_type->get_element_type();

            if (result_shape.size() == 1)
            {
                PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                             "Concat has unhandled element type",
                                             runtime::eigen::ConcatVectorInstruction,
                                             in,
                                             out[0]);
            }
            else if (result_shape.size() == 2)
            {
                PUSH_POLYMORPHIC_INSTRUCTION(
                    result_element_type,
                    "Concat has unhandled element type",
                    runtime::eigen::ConcatMatrixInstruction,
                    in,
                    (dynamic_cast<const op::Concat*>(n))->get_concatenation_axis(),
                    out[0]);
            }
            else
            {
                throw ngraph_error("Concat not implemented for rank>2 in VM yet");
            }
        };

        REGISTER_TO_OP_MAP(op::Convert)
        {
            auto arg = n->get_arguments().at(0);

            auto arg_tensor_type =
                dynamic_pointer_cast<const TensorViewType>(arg->get_value_type());
            assert(nullptr != arg_tensor_type);

            auto& arg_element_type = arg_tensor_type->get_element_type();

            auto result_tensor_type =
                dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
            assert(nullptr != result_tensor_type);

            auto& result_element_type = result_tensor_type->get_element_type();

// Hacky macro: we are going to be building up a series of else-ifs for each possible
// pair of element types.
#define REGISTER_CONVERT(TI, TO)                                                                   \
    else if (arg_element_type == (TI::element_type()) &&                                           \
             result_element_type == (TO::element_type()))                                          \
    {                                                                                              \
        ef->get_instructions()->push_back(                                                         \
            make_shared<runtime::eigen::ConvertInstruction<TI, TO>>(in[0], out[0]));               \
    }
// End hacky macro

// Hacky macro: Given some type TI, generate the else-ifs for TI to every other element
// type.
#define REGISTER_CONVERTS(TI)                                                                      \
    REGISTER_CONVERT(TI, element::Bool)                                                            \
    REGISTER_CONVERT(TI, element::Float32)                                                         \
    REGISTER_CONVERT(TI, element::Int8)                                                            \
    REGISTER_CONVERT(TI, element::Int32)                                                           \
    REGISTER_CONVERT(TI, element::Int64)                                                           \
    REGISTER_CONVERT(TI, element::UInt8)                                                           \
    REGISTER_CONVERT(TI, element::UInt32)                                                          \
    REGISTER_CONVERT(TI, element::UInt64)
            // End hacky macro

            if (false)
            {
            }
            REGISTER_CONVERTS(element::Bool)
            REGISTER_CONVERTS(element::Float32)
            REGISTER_CONVERTS(element::Int8)
            REGISTER_CONVERTS(element::Int32)
            REGISTER_CONVERTS(element::Int64)
            REGISTER_CONVERTS(element::UInt8)
            REGISTER_CONVERTS(element::UInt32)
            REGISTER_CONVERTS(element::UInt64)
            else { throw ngraph_error("Internal error: cannot convert between element types"); }
#undef REGISTER_CONVERTS
#undef REGISTER_CONVERT
        };

        REGISTER_TO_OP_MAP(op::Dot)
        {
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
            auto& arg0_element_type = arg0_tensor_type->get_element_type();

            // If arg0 or arg1 is a scalar, emit a scalar-tensor product.
            if (arg0_shape.size() == 0)
            {
                PUSH_NUMERIC_POLYMORPHIC_INSTRUCTION(arg0_element_type,
                                                     "Dot has unhandled element type",
                                                     runtime::eigen::ScalarTensorProductInstruction,
                                                     in[0],
                                                     in[1],
                                                     out[0]);
            }
            else if (arg1_shape.size() == 0)
            {
                PUSH_NUMERIC_POLYMORPHIC_INSTRUCTION(arg0_element_type,
                                                     "Dot has unhandled element type",
                                                     runtime::eigen::ScalarTensorProductInstruction,
                                                     in[1],
                                                     in[0],
                                                     out[0]);
            }

            // If arg0 and arg1 are both vectors, emit a dot product.
            else if (arg0_shape.size() == 1 && arg1_shape.size() == 1)
            {
                PUSH_NUMERIC_POLYMORPHIC_INSTRUCTION(arg0_element_type,
                                                     "Dot has unhandled element type",
                                                     runtime::eigen::DotInstruction,
                                                     in[0],
                                                     in[1],
                                                     out[0]);
            }

            // If arg0 is a matrix and arg1 is a vector, emit a matrix-vector product.
            else if (arg0_shape.size() == 2 && arg1_shape.size() == 1)
            {
                PUSH_NUMERIC_POLYMORPHIC_INSTRUCTION(arg0_element_type,
                                                     "Dot has unhandled element type",
                                                     runtime::eigen::MatrixVectorProductInstruction,
                                                     in[0],
                                                     in[1],
                                                     out[0]);
            }

            // If arg0 and arg1 are both matrices, emit a matrix product.
            else if (arg0_shape.size() == 2 && arg1_shape.size() == 2)
            {
                PUSH_NUMERIC_POLYMORPHIC_INSTRUCTION(arg0_element_type,
                                                     "Dot has unhandled element type",
                                                     runtime::eigen::MatrixMultInstruction,
                                                     in[0],
                                                     in[1],
                                                     out[0]);
            }

            else
            {
                throw ngraph_error("Dot product for tensors with rank>2 not implemented yet.");
            }
        };

        // Parameter is a "runtime no-op" because the output tensor has already been filled.
        REGISTER_TO_OP_MAP(op::Parameter){};

        // GetTupleElement will be spliced out, with the users of out redirected to in's source, but, for now, we need to copy.
        REGISTER_TO_OP_MAP(op::GetTupleElement)
        {
            auto get_tuple_element = static_cast<const op::GetTupleElement*>(n);

            auto result_tensor_type =
                dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
            assert(nullptr != result_tensor_type);

            auto& result_element_type = result_tensor_type->get_element_type();

            PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                         "GetTupleElement has unhandled element type",
                                         runtime::eigen::CopyInstruction,
                                         in.at(get_tuple_element->get_n()).get_index(),
                                         out.at(0).get_index());
        };

        // Tuple will be spliced out, with the users of out connected to the corresponding in's source, but, for now, we need to copy.
        REGISTER_TO_OP_MAP(op::Tuple)
        {
            for (size_t i = 0; i < in.size(); ++i)
            {
                auto& et = in.at(i).get_tensor_view_layout()->get_element_type();
                PUSH_POLYMORPHIC_INSTRUCTION(et,
                                             "Tuple has unhandled element type",
                                             runtime::eigen::CopyInstruction,
                                             in.at(i).get_index(),
                                             out.at(i).get_index());
            }
        };

        REGISTER_TO_OP_MAP(op::FunctionCall)
        {
            auto function_call = static_cast<const op::FunctionCall*>(n);
            auto function = function_call->get_function();

            std::shared_ptr<ExternalFunction> external;

            try
            {
                external = function_map.at(function);
            }
            catch (const std::out_of_range)
            {
                external =
                    make_shared<ngraph::runtime::ExternalFunction>(function_call->get_function());
                function_map.insert({function, external});
            }

            ef->get_instructions()->push_back(
                make_shared<runtime::eigen::CallInstruction>(external, in, out));
        };

        REGISTER_TO_OP_MAP(op::Reduce) { throw ngraph_error("op::Reduce not implemented yet"); };
        initialized = true;
    }
    return op_map;
}

void ExternalFunction::compile(FunctionMap& function_map)
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

    // Turn this into a pass
    // Assign layouts
    // For now, just make everyone row-major.
    for (const Node* node : m_function->get_ordered_ops())
    {
        for (const descriptor::Output& output : node->get_outputs())
        {
            auto tv = output.get_tensor_view();
            if (nullptr == tv->get_tensor_view_layout())
            {
                auto layout = std::make_shared<DenseTensorViewLayout>(*tv);
                tv->set_tensor_view_layout(layout);
            }
        }
    }

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
    for (const Node* node : m_function->get_ordered_ops())
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
    for (const Node* node : m_function->get_ordered_ops())
    {
        auto handler_it = op_map.find(type_index(typeid(*node)));
        if (handler_it == op_map.end())
        {
            throw ngraph_error("Unhandled op during code generation");
        }
        std::vector<TensorViewInfo> in;
        for (const descriptor::Input& input : node->get_inputs())
        {
            const descriptor::Output& output = input.get_output();
            auto tv = output.get_tensor_view();
            in.push_back({tensor_index.at(tv), tv});
        }
        std::vector<TensorViewInfo> out;
        for (const descriptor::Output& output : node->get_outputs())
        {
            auto tv = output.get_tensor_view();
            out.push_back({tensor_index.at(tv), tv});
        }
        handler_it->second(node, this, function_map, in, out);
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
    FunctionMap function_map;
    return make_call_frame(function_map);
}

shared_ptr<ngraph::runtime::CallFrame> ExternalFunction::make_call_frame(FunctionMap& function_map)
{
    if (!m_is_compiled)
    {
        compile(function_map);
    }
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> temps;
    for (auto tv : m_temp_views)
    {
        auto& et = tv->get_tensor_view_type()->get_element_type();
        auto shape = tv->get_tensor_view_type()->get_shape();

#define M(T) temps.push_back(ngraph::runtime::make_tensor<T>(shape));
        DO_ON_ELEMENT_TYPE(
            et, "Internal error: tried to create temporary for unhandled element type", M);
#undef M
    }
    return make_shared<ngraph::runtime::CallFrame>(
        m_n_inputs, m_n_outputs, temps, 0, m_instructions);
}
