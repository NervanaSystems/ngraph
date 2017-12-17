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
#include "ngraph/ops/acos.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/asin.hpp"
#include "ngraph/ops/atan.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/ceiling.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convert.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/cos.hpp"
#include "ngraph/ops/cosh.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/equal.hpp"
#include "ngraph/ops/exp.hpp"
#include "ngraph/ops/floor.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/greater.hpp"
#include "ngraph/ops/greater_eq.hpp"
#include "ngraph/ops/less.hpp"
#include "ngraph/ops/less_eq.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/minimum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/not.hpp"
#include "ngraph/ops/not_equal.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/power.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/select.hpp"
#include "ngraph/ops/sign.hpp"
#include "ngraph/ops/sin.hpp"
#include "ngraph/ops/sinh.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sqrt.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/ops/tan.hpp"
#include "ngraph/ops/tanh.hpp"
#include "ngraph/ops/xla_get_tuple_element.hpp"
#include "ngraph/ops/xla_tuple.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "ngraph/runtime/ngvm/external_function.hpp"
#include "ngraph/runtime/ngvm/instruction/abs.hpp"
#include "ngraph/runtime/ngvm/instruction/acos.hpp"
#include "ngraph/runtime/ngvm/instruction/add.hpp"
#include "ngraph/runtime/ngvm/instruction/asin.hpp"
#include "ngraph/runtime/ngvm/instruction/atan.hpp"
#include "ngraph/runtime/ngvm/instruction/broadcast.hpp"
#include "ngraph/runtime/ngvm/instruction/call.hpp"
#include "ngraph/runtime/ngvm/instruction/ceiling.hpp"
#include "ngraph/runtime/ngvm/instruction/concat.hpp"
#include "ngraph/runtime/ngvm/instruction/constant.hpp"
#include "ngraph/runtime/ngvm/instruction/convert.hpp"
#include "ngraph/runtime/ngvm/instruction/convolution.hpp"
#include "ngraph/runtime/ngvm/instruction/copy.hpp"
#include "ngraph/runtime/ngvm/instruction/copy_by_index.hpp"
#include "ngraph/runtime/ngvm/instruction/cos.hpp"
#include "ngraph/runtime/ngvm/instruction/cosh.hpp"
#include "ngraph/runtime/ngvm/instruction/divide.hpp"
#include "ngraph/runtime/ngvm/instruction/dot.hpp"
#include "ngraph/runtime/ngvm/instruction/equal.hpp"
#include "ngraph/runtime/ngvm/instruction/exp.hpp"
#include "ngraph/runtime/ngvm/instruction/floor.hpp"
#include "ngraph/runtime/ngvm/instruction/greater.hpp"
#include "ngraph/runtime/ngvm/instruction/greater_eq.hpp"
#include "ngraph/runtime/ngvm/instruction/less.hpp"
#include "ngraph/runtime/ngvm/instruction/less_eq.hpp"
#include "ngraph/runtime/ngvm/instruction/log.hpp"
#include "ngraph/runtime/ngvm/instruction/maximum.hpp"
#include "ngraph/runtime/ngvm/instruction/minimum.hpp"
#include "ngraph/runtime/ngvm/instruction/multiply.hpp"
#include "ngraph/runtime/ngvm/instruction/negate.hpp"
#include "ngraph/runtime/ngvm/instruction/not.hpp"
#include "ngraph/runtime/ngvm/instruction/not_equal.hpp"
#include "ngraph/runtime/ngvm/instruction/one_hot.hpp"
#include "ngraph/runtime/ngvm/instruction/power.hpp"
#include "ngraph/runtime/ngvm/instruction/reduce.hpp"
#include "ngraph/runtime/ngvm/instruction/replace_slice.hpp"
#include "ngraph/runtime/ngvm/instruction/reshape.hpp"
#include "ngraph/runtime/ngvm/instruction/return.hpp"
#include "ngraph/runtime/ngvm/instruction/select.hpp"
#include "ngraph/runtime/ngvm/instruction/sign.hpp"
#include "ngraph/runtime/ngvm/instruction/sin.hpp"
#include "ngraph/runtime/ngvm/instruction/sinh.hpp"
#include "ngraph/runtime/ngvm/instruction/slice.hpp"
#include "ngraph/runtime/ngvm/instruction/sqrt.hpp"
#include "ngraph/runtime/ngvm/instruction/subtract.hpp"
#include "ngraph/runtime/ngvm/instruction/sum.hpp"
#include "ngraph/runtime/ngvm/instruction/tan.hpp"
#include "ngraph/runtime/ngvm/instruction/tanh.hpp"
#include "ngraph/runtime/ngvm/types.hpp"
#include "ngraph/runtime/utils.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph::runtime::ngvm;

using ngraph::descriptor::layout::DenseTensorViewLayout;

ExternalFunction::ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                   bool release_function)
    : ngraph::runtime::ExternalFunction(function, release_function)
    , m_instructions(make_shared<std::vector<std::shared_ptr<Instruction>>>())
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
        if (et == element::boolean)                                                                \
        {                                                                                          \
            macro(element::Bool, ##__VA_ARGS__);                                                   \
        }                                                                                          \
        else if (et == element::f32)                                                               \
        {                                                                                          \
            macro(element::Float32, ##__VA_ARGS__);                                                \
        }                                                                                          \
        else if (et == element::f64)                                                               \
        {                                                                                          \
            macro(element::Float64, ##__VA_ARGS__);                                                \
        }                                                                                          \
        else if (et == element::i8)                                                                \
        {                                                                                          \
            macro(element::Int8, ##__VA_ARGS__);                                                   \
        }                                                                                          \
        else if (et == element::i16)                                                               \
        {                                                                                          \
            macro(element::Int16, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::i32)                                                               \
        {                                                                                          \
            macro(element::Int32, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::i64)                                                               \
        {                                                                                          \
            macro(element::Int64, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::u8)                                                                \
        {                                                                                          \
            macro(element::UInt8, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::u16)                                                               \
        {                                                                                          \
            macro(element::UInt16, ##__VA_ARGS__);                                                 \
        }                                                                                          \
        else if (et == element::u32)                                                               \
        {                                                                                          \
            macro(element::UInt32, ##__VA_ARGS__);                                                 \
        }                                                                                          \
        else if (et == element::u64)                                                               \
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
        if (et == element::f32)                                                                    \
        {                                                                                          \
            macro(element::Float32, ##__VA_ARGS__);                                                \
        }                                                                                          \
        else if (et == element::f64)                                                               \
        {                                                                                          \
            macro(element::Float64, ##__VA_ARGS__);                                                \
        }                                                                                          \
        else if (et == element::i8)                                                                \
        {                                                                                          \
            macro(element::Int8, ##__VA_ARGS__);                                                   \
        }                                                                                          \
        else if (et == element::i16)                                                               \
        {                                                                                          \
            macro(element::Int16, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::i32)                                                               \
        {                                                                                          \
            macro(element::Int32, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::i64)                                                               \
        {                                                                                          \
            macro(element::Int64, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::u8)                                                                \
        {                                                                                          \
            macro(element::UInt8, ##__VA_ARGS__);                                                  \
        }                                                                                          \
        else if (et == element::u16)                                                               \
        {                                                                                          \
            macro(element::UInt16, ##__VA_ARGS__);                                                 \
        }                                                                                          \
        else if (et == element::u32)                                                               \
        {                                                                                          \
            macro(element::UInt32, ##__VA_ARGS__);                                                 \
        }                                                                                          \
        else if (et == element::u64)                                                               \
        {                                                                                          \
            macro(element::UInt64, ##__VA_ARGS__);                                                 \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            throw ngraph_error(err_msg);                                                           \
        }                                                                                          \
    }

#define M_REGISTER_NUMERIC_UNOP(T, instr_class)                                                    \
    ef->get_instructions()->push_back(make_shared<instr_class<T>>(in[0], out[0]));
#define REGISTER_NUMERIC_UNOP(op_class, instr_class)                                               \
    REGISTER_TO_OP_MAP(op_class)                                                                   \
    {                                                                                              \
        const element::Type& et = n->get_inputs().at(0).get_element_type();                        \
        DO_ON_NUMERIC_TYPE(et,                                                                     \
                           "Internal error: numeric unop has unhandled element type",              \
                           M_REGISTER_NUMERIC_UNOP,                                                \
                           instr_class);                                                           \
    }

#define REGISTER_LOGICAL_UNOP(op_class, instr_class)                                               \
    REGISTER_TO_OP_MAP(op_class)                                                                   \
    {                                                                                              \
<<<<<<< HEAD
        const element::Type& et = n->get_inputs().at(0).get_element_type();                        \
        if (element::Bool::element_type() == et)                                                   \
=======
        const element::Type& et = (dynamic_pointer_cast<const TensorViewType>(                     \
                                       n->get_inputs().at(0).get_tensor_view_type()))              \
                                      ->get_element_type();                                        \
        if (element::boolean == et)                                                                \
>>>>>>> apply format, which keeps crashing
        {                                                                                          \
            ef->get_instructions()->push_back(make_shared<instr_class>(in[0], out[0]));            \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            throw ngraph_error("Internal error: logical unop has unhandled element type");         \
        }                                                                                          \
    }

#define M_REGISTER_NUMERIC_BINOP(T, instr_class)                                                   \
    ef->get_instructions()->push_back(make_shared<instr_class<T>>(in[0], in[1], out[0]));
#define REGISTER_NUMERIC_BINOP(op_class, instr_class)                                              \
    REGISTER_TO_OP_MAP(op_class)                                                                   \
    {                                                                                              \
        const element::Type& et = n->get_inputs().at(0).get_element_type();                        \
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
        const element::Type& et = n->get_inputs().at(0).get_element_type();                        \
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
        const element::Type& et = n->get_inputs().at(1).get_element_type();                        \
        DO_ON_ELEMENT_TYPE(et,                                                                     \
                           "Internal error: polymorphic ternop has unhandled element type",        \
                           M_REGISTER_POLYMORPHIC_TERNOP,                                          \
                           instr_class);                                                           \
    }

#define PUSH_INSTRUCTION(T, instr, ...)                                                            \
    {                                                                                              \
        ef->get_instructions()->push_back(make_shared<instr<T>>(__VA_ARGS__));                     \
    }
#define PUSH_POLYMORPHIC_INSTRUCTION(et, err_msg, instr, ...)                                      \
    DO_ON_ELEMENT_TYPE(et, err_msg, PUSH_INSTRUCTION, instr, __VA_ARGS__)

// Turn off complaint suppression (see above)
#pragma clang diagnostic pop

// Define code generators for handled ops.
ExternalFunction::OpMap& ExternalFunction::get_op_map()
{
    static bool initialized = false;
    static OpMap op_map;
    if (!initialized)
    {
        REGISTER_NUMERIC_UNOP(op::Abs, instruction::AbsInstruction);
        REGISTER_NUMERIC_UNOP(op::Acos, instruction::AcosInstruction);
        REGISTER_NUMERIC_UNOP(op::Asin, instruction::AsinInstruction);
        REGISTER_NUMERIC_UNOP(op::Atan, instruction::AtanInstruction);
        REGISTER_NUMERIC_UNOP(op::Ceiling, instruction::CeilingInstruction);
        REGISTER_NUMERIC_UNOP(op::Cos, instruction::CosInstruction);
        REGISTER_NUMERIC_UNOP(op::Cosh, instruction::CoshInstruction);
        REGISTER_NUMERIC_UNOP(op::Exp, instruction::ExpInstruction);
        REGISTER_NUMERIC_UNOP(op::Floor, instruction::FloorInstruction);
        REGISTER_NUMERIC_UNOP(op::Log, instruction::LogInstruction);
        REGISTER_NUMERIC_UNOP(op::Negative, instruction::NegateInstruction);
        REGISTER_NUMERIC_UNOP(op::Sign, instruction::SignInstruction);
        REGISTER_NUMERIC_UNOP(op::Sin, instruction::SinInstruction);
        REGISTER_NUMERIC_UNOP(op::Sinh, instruction::SinhInstruction);
        REGISTER_NUMERIC_UNOP(op::Sqrt, instruction::SqrtInstruction);
        REGISTER_NUMERIC_UNOP(op::Tan, instruction::TanInstruction);
        REGISTER_NUMERIC_UNOP(op::Tanh, instruction::TanhInstruction);

        REGISTER_NUMERIC_BINOP(op::Add, instruction::AddInstruction);
        REGISTER_NUMERIC_BINOP(op::Divide, instruction::DivideInstruction);
        REGISTER_NUMERIC_BINOP(op::Maximum, instruction::MaximumInstruction);
        REGISTER_NUMERIC_BINOP(op::Minimum, instruction::MinimumInstruction);
        REGISTER_NUMERIC_BINOP(op::Multiply, instruction::MultiplyInstruction);
        REGISTER_NUMERIC_BINOP(op::Power, instruction::PowerInstruction);
        REGISTER_NUMERIC_BINOP(op::Subtract, instruction::SubtractInstruction);

        REGISTER_TO_OP_MAP(op::Constant)
        {
            auto c = static_cast<const op::Constant*>(n);
            auto c_tensor_type = dynamic_pointer_cast<const TensorViewType>(c->get_value_type());
            assert(nullptr != c_tensor_type);
            auto& c_element_type = c_tensor_type->get_element_type();
            auto c_value_strings = c->get_value_strings();

#define M_REGISTER_POLYMORPHIC_CONSTANT(ET)                                                        \
    ef->get_instructions()->push_back(make_shared<instruction::ConstantInstruction<ET>>(           \
        parse_string<typename ET::type>(c_value_strings), out[0]));

            DO_ON_ELEMENT_TYPE(c_element_type,
                               "Constant has unhandled element type",
                               M_REGISTER_POLYMORPHIC_CONSTANT);
        };

        REGISTER_POLYMORPHIC_BINOP(op::Equal, instruction::EqualInstruction);
        REGISTER_POLYMORPHIC_BINOP(op::NotEqual, instruction::NotEqualInstruction);
        REGISTER_POLYMORPHIC_BINOP(op::Greater, instruction::GreaterInstruction);
        REGISTER_POLYMORPHIC_BINOP(op::GreaterEq, instruction::GreaterEqInstruction);
        REGISTER_POLYMORPHIC_BINOP(op::Less, instruction::LessInstruction);
        REGISTER_POLYMORPHIC_BINOP(op::LessEq, instruction::LessEqInstruction);

        REGISTER_LOGICAL_UNOP(op::Not, instruction::NotInstruction);

        REGISTER_POLYMORPHIC_TERNOP(op::Select, instruction::SelectInstruction);

        REGISTER_TO_OP_MAP(op::Broadcast)
        {
            auto broadcast = static_cast<const op::Broadcast*>(n);

            auto& input_shape = n->get_inputs().at(0).get_shape();
            auto& result = n->get_outputs().at(0);
            auto result_shape = result.get_shape();
            auto& result_element_type = result.get_element_type();

            PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                         "Broadcast has unhandled element type",
                                         instruction::BroadcastInstruction,
                                         in[0],
                                         out[0],
                                         input_shape,
                                         result_shape,
                                         broadcast->get_broadcast_axes());
        };

        REGISTER_TO_OP_MAP(op::Concat)
        {
            auto concat = static_cast<const op::Concat*>(n);

            std::vector<Shape> input_shapes;

            for (auto& input : n->get_inputs())
            {
                input_shapes.push_back(input.get_shape());
            }

            auto& result = n->get_outputs().at(0);
            auto result_shape = result.get_shape();
            auto& result_element_type = result.get_element_type();

            PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                         "Concat has unhandled element type",
                                         instruction::ConcatInstruction,
                                         in,
                                         out[0],
                                         input_shapes,
                                         result_shape,
                                         concat->get_concatenation_axis());
        };

        REGISTER_TO_OP_MAP(op::Convert)
        {
            auto& arg_element_type = n->get_inputs().at(0).get_element_type();
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
            make_shared<instruction::ConvertInstruction<TI, TO>>(in[0], out[0]));                  \
    }
// End hacky macro

// Hacky macro: Given some type TI, generate the else-ifs for TI to every other element
// type.
#define REGISTER_CONVERTS(TI)                                                                      \
    REGISTER_CONVERT(TI, element::Bool)                                                            \
    REGISTER_CONVERT(TI, element::Float32)                                                         \
    REGISTER_CONVERT(TI, element::Float64)                                                         \
    REGISTER_CONVERT(TI, element::Int8)                                                            \
    REGISTER_CONVERT(TI, element::Int16)                                                           \
    REGISTER_CONVERT(TI, element::Int32)                                                           \
    REGISTER_CONVERT(TI, element::Int64)                                                           \
    REGISTER_CONVERT(TI, element::UInt8)                                                           \
    REGISTER_CONVERT(TI, element::UInt16)                                                          \
    REGISTER_CONVERT(TI, element::UInt32)                                                          \
    REGISTER_CONVERT(TI, element::UInt64)
            // End hacky macro

            if (false)
            {
            }
            REGISTER_CONVERTS(element::Bool)
            REGISTER_CONVERTS(element::Float32)
            REGISTER_CONVERTS(element::Float64)
            REGISTER_CONVERTS(element::Int8)
            REGISTER_CONVERTS(element::Int16)
            REGISTER_CONVERTS(element::Int32)
            REGISTER_CONVERTS(element::Int64)
            REGISTER_CONVERTS(element::UInt8)
            REGISTER_CONVERTS(element::UInt16)
            REGISTER_CONVERTS(element::UInt32)
            REGISTER_CONVERTS(element::UInt64)
            else { throw ngraph_error("Internal error: cannot convert between element types"); }
#undef REGISTER_CONVERTS
#undef REGISTER_CONVERT
        };

        REGISTER_TO_OP_MAP(op::Convolution)
        {
            auto convolution = static_cast<const op::Convolution*>(n);

            auto input_0_shape = n->get_inputs().at(0).get_shape();
            auto input_1_shape = n->get_inputs().at(1).get_shape();

            auto& result = n->get_outputs().at(0);
            auto& result_shape = result.get_shape();
            auto& result_element_type = result.get_element_type();

            PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                         "Convolution has unhandled element type",
                                         instruction::ConvolutionInstruction,
                                         in[0],
                                         in[1],
                                         out[0],
                                         input_0_shape,
                                         input_1_shape,
                                         result_shape,
                                         convolution->get_window_movement_strides(),
                                         convolution->get_window_dilation_strides());
        };

        REGISTER_TO_OP_MAP(op::Dot)
        {
            auto dot = static_cast<const op::Dot*>(n);

            assert(n->get_inputs().size() == 2);

            auto& input_0 = n->get_inputs().at(0);
            auto& input_1 = n->get_inputs().at(1);

            auto input_0_shape = input_0.get_shape();
            auto input_1_shape = input_1.get_shape();
            auto& input_0_element_type = input_0.get_element_type();

            auto reduction_axes_count = dot->get_reduction_axes_count();

            auto result_shape = n->get_outputs().at(0).get_shape();

            PUSH_POLYMORPHIC_INSTRUCTION(input_0_element_type,
                                         "Dot has unhandled element type",
                                         instruction::DotInstruction,
                                         in[0],
                                         in[1],
                                         out[0],
                                         input_0_shape,
                                         input_1_shape,
                                         result_shape,
                                         reduction_axes_count);
        };

        // Parameter is a "runtime no-op" because the output tensor has already been filled.
        REGISTER_TO_OP_MAP(op::Parameter){};

        // GetOutputElement will be spliced out, with the users of out redirected to in's source, but, for now, we need to copy.
        REGISTER_TO_OP_MAP(op::XLAGetTupleElement)
        {
            auto get_tuple_element = static_cast<const op::XLAGetTupleElement*>(n);

            auto result_tensor_type =
                dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
            assert(nullptr != result_tensor_type);

            auto& result_element_type = result_tensor_type->get_element_type();

            PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                         "XLAGetTupleElement has unhandled element type",
                                         instruction::CopyInstruction,
                                         in[get_tuple_element->get_n()],
                                         out[0]);
        };

        // Tuple will be spliced out, with the users of out connected to the corresponding in's source, but, for now, we need to copy.
        REGISTER_TO_OP_MAP(op::XLATuple)
        {
            for (size_t i = 0; i < in.size(); ++i)
            {
                auto& et = in.at(i).get_tensor_view_layout()->get_element_type();
                PUSH_POLYMORPHIC_INSTRUCTION(et,
                                             "Tuple has unhandled element type",
                                             instruction::CopyInstruction,
                                             in[i],
                                             out[i]);
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
                external = make_shared<ExternalFunction>(function);
                function_map.insert({function, external});
            }

            ef->get_instructions()->push_back(
                make_shared<instruction::CallInstruction>(external, in, out));
        };

        REGISTER_TO_OP_MAP(op::Reduce)
        {
            auto reduce = static_cast<const op::Reduce*>(n);

            auto reduction_function = reduce->get_function();
            std::shared_ptr<ExternalFunction> external;

            try
            {
                external = function_map.at(reduction_function);
            }
            catch (const std::out_of_range)
            {
                external = make_shared<ExternalFunction>(reduction_function);
                function_map.insert({reduction_function, external});
            }

            auto input_shape = n->get_inputs().at(0).get_shape();
            auto& result = n->get_outputs().at(0);
            auto result_shape = result.get_shape();
            auto& result_element_type = result.get_element_type();

#define M(ET)                                                                                      \
    {                                                                                              \
        auto reduce_handler = [external](typename ET::type x, typename ET::type y) ->              \
            typename ET::type                                                                      \
        {                                                                                          \
            std::shared_ptr<CallFrame> cf =                                                        \
                std::dynamic_pointer_cast<CallFrame>(external->make_call_frame());                 \
                                                                                                   \
            auto tx = ngraph::runtime::ngvm::make_tensor<ET>(Shape{}, {x});                        \
            auto ty = ngraph::runtime::ngvm::make_tensor<ET>(Shape{}, {y});                        \
            auto tr = ngraph::runtime::ngvm::make_tensor<ET>(Shape{});                             \
                                                                                                   \
            cf->call({tx, ty}, {tr});                                                              \
            return tr->get_vector()[0];                                                            \
        };                                                                                         \
                                                                                                   \
        PUSH_INSTRUCTION(ET,                                                                       \
                         instruction::ReduceInstruction,                                           \
                         in[0],                                                                    \
                         in[1],                                                                    \
                         out[0],                                                                   \
                         input_shape,                                                              \
                         result_shape,                                                             \
                         reduce->get_reduction_axes(),                                             \
                         reduce_handler);                                                          \
    }

            DO_ON_ELEMENT_TYPE(
                result_element_type,
                "Internal error: tried to create reduction handler for unhandled element type",
                M);
#undef M
        };

        REGISTER_TO_OP_MAP(op::Sum)
        {
            auto sum = static_cast<const op::Sum*>(n);

            auto input_shape = n->get_inputs().at(0).get_shape();

            auto& result = n->get_outputs().at(0);
            auto result_shape = result.get_shape();
            auto& result_element_type = result.get_element_type();

            PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                         "Sum has unhandled element type",
                                         instruction::SumInstruction,
                                         in[0],
                                         out[0],
                                         input_shape,
                                         result_shape,
                                         sum->get_reduction_axes());
        };

        REGISTER_TO_OP_MAP(op::Reshape)
        {
            auto reshape = static_cast<const op::Reshape*>(n);

            auto input_shape = n->get_inputs().at(0).get_shape();

            auto& result = n->get_outputs().at(0);
            auto result_shape = result.get_shape();
            auto& result_element_type = result.get_element_type();

            PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                         "Reshape has unhandled element type",
                                         instruction::ReshapeInstruction,
                                         in[0],
                                         out[0],
                                         input_shape,
                                         reshape->get_input_order(),
                                         result_shape);
        };

        REGISTER_TO_OP_MAP(op::Slice)
        {
            auto slice = static_cast<const op::Slice*>(n);

            auto& input = slice->get_inputs().at(0);
            auto input_shape = input.get_shape();
            auto& input_element_type = input.get_element_type();

            auto result_shape = slice->get_outputs().at(0).get_shape();

            auto& lower_bounds = slice->get_lower_bounds();
            auto& upper_bounds = slice->get_upper_bounds();

            auto& strides = slice->get_strides();

            PUSH_POLYMORPHIC_INSTRUCTION(input_element_type,
                                         "Slice has unhandled element type",
                                         runtime::ngvm::instruction::SliceInstruction,
                                         in[0],
                                         out[0],
                                         input_shape,
                                         lower_bounds,
                                         upper_bounds,
                                         strides,
                                         result_shape);
        };

        REGISTER_TO_OP_MAP(op::ReplaceSlice)
        {
            auto replace_slice = static_cast<const op::ReplaceSlice*>(n);

            auto& input_0_element_type = replace_slice->get_inputs().at(0).get_element_type();
            auto input_1_shape = replace_slice->get_inputs().at(1).get_shape();

            auto result_shape = replace_slice->get_outputs().at(0).get_shape();

            auto& lower_bounds = replace_slice->get_lower_bounds();
            auto& upper_bounds = replace_slice->get_upper_bounds();

            auto& strides = replace_slice->get_strides();

            PUSH_POLYMORPHIC_INSTRUCTION(input_0_element_type,
                                         "Replace-slice has unhandled element type",
                                         runtime::ngvm::instruction::ReplaceSliceInstruction,
                                         in[0],
                                         in[1],
                                         out[0],
                                         input_1_shape,
                                         lower_bounds,
                                         upper_bounds,
                                         strides,
                                         result_shape);
        };

        REGISTER_TO_OP_MAP(op::OneHot)
        {
            auto one_hot = static_cast<const op::OneHot*>(n);

            auto input_shape = n->get_inputs().at(0).get_shape();

            auto& result = n->get_outputs().at(0);
            auto result_shape = result.get_shape();
            auto& result_element_type = result.get_element_type();

            PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                         "One-hot has unhandled element type",
                                         instruction::OneHotInstruction,
                                         in[0],
                                         out[0],
                                         input_shape,
                                         result_shape,
                                         one_hot->get_one_hot_axis());
        };

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
    pass_manager.run_passes(m_function);

    // Turn this into a pass
    // Assign layouts
    // For now, just make everyone row-major.
    for (shared_ptr<Node> node : m_function->get_ordered_ops())
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
            size_t index = m_frame_size++;
            tensor_index[tv] = index;
        }
    }
    m_n_inputs = m_frame_size;

    for (const descriptor::Output* output : m_function->get_outputs())
    {
        auto tv = output->get_tensor_view();
        size_t index = m_frame_size++;
        auto prev_index_it = tensor_index.find(tv);
        if (prev_index_it != tensor_index.end())
        {
            auto result_tensor_type =
                dynamic_pointer_cast<const TensorViewType>(tv->get_value_type());
            assert(nullptr != result_tensor_type);
            auto& result_element_type = result_tensor_type->get_element_type();
            auto ef = this;
            // TODO: This is the one case where we can't use the new CopyInstruction that takes in a TensorViewInfo. (At least, I can't figure out how to do it.)
            PUSH_POLYMORPHIC_INSTRUCTION(result_element_type,
                                         "Copy has unhandled element type",
                                         instruction::CopyByIndexInstruction,
                                         prev_index_it->second,
                                         index);
        }
        else
        {
            tensor_index[tv] = index;
        }
    }
    m_n_outputs = m_frame_size - m_n_inputs;
    vector<shared_ptr<Instruction>> input_output_copies;
    swap(*m_instructions, input_output_copies);

    // All remaining tensor views
    for (shared_ptr<Node> node : m_function->get_ordered_ops())
    {
        for (const descriptor::Output& output : node->get_outputs())
        {
            auto tv = output.get_tensor_view();
            if (0 == tensor_index.count(tv))
            {
                size_t index = m_frame_size++;
                tensor_index[tv] = index;
                m_temp_views.push_back(tv);
            }
        }
    }

    // Now we build the VM instructions
    auto op_map = get_op_map();
    for (shared_ptr<Node> node : m_function->get_ordered_ops())
    {
        auto& n = *node; // Work around a compiler warning (*node inside typeid may have effects
                         // with shared pointers, which is fine here but clang doesn't like it.)
        auto handler_it = op_map.find(type_index(typeid(n)));
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
        handler_it->second(node.get(), this, function_map, in, out);
    }
    m_instructions->insert(
        m_instructions->end(), input_output_copies.begin(), input_output_copies.end());
    m_instructions->push_back(make_shared<instruction::ReturnInstruction>());
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

#define M(T) temps.push_back(ngraph::runtime::ngvm::make_tensor<T>(shape));
        DO_ON_ELEMENT_TYPE(
            et, "Internal error: tried to create temporary for unhandled element type", M);
#undef M
    }
    return make_shared<ngraph::runtime::ngvm::CallFrame>(
        m_n_inputs, m_n_outputs, m_frame_size, temps, 0, m_instructions);
}
