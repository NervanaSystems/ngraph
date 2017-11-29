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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/get_tuple_element.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/runtime/cpu/call_frame.hpp"
#include "ngraph/runtime/cpu/emitter.hpp"
#include "ngraph/runtime/cpu/external_function.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::runtime::cpu;

using ngraph::descriptor::layout::DenseTensorViewLayout;

static string eigen_vector_format(const runtime::TensorViewInfo& tvi)
{
    return "fmt::V{" + to_string(tvi.get_layout<DenseTensorViewLayout>()->get_size()) + "}";
}

static std::string eigen_matrix_format(const ngraph::Shape& shape, const ngraph::Strides& strides)
{
    stringstream ss;
    ss << "fmt::M{{" << join(shape) << "}, {" << join(strides) << "}}";
    return ss.str();
}

void Emitter::EmitNop(const ngraph::Node* n,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{
}

void Emitter::EmitAdd(const ngraph::Node* n,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " = \n";
    TU.indent++;
    TU << emit_array1d(inputs[0]) << " +\n ";
    TU << emit_array1d(inputs[1]) << ";\n";
    TU.indent -= 2;
    TU << "}\n";
}

void Emitter::EmitDot(const ngraph::Node* n,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{
    auto& arg_nodes = n->get_arguments();
    assert(arg_nodes.size() == 2);

    auto arg0_tensor_type =
        dynamic_pointer_cast<const TensorViewType>(arg_nodes.at(0)->get_value_type());
    assert(arg0_tensor_type);

    auto arg1_tensor_type =
        dynamic_pointer_cast<const TensorViewType>(arg_nodes.at(1)->get_value_type());
    assert(arg1_tensor_type);

    auto arg0_shape = arg0_tensor_type->get_shape();
    auto arg1_shape = arg1_tensor_type->get_shape();
    auto& arg0_element_type = arg0_tensor_type->get_element_type();

    if (arg0_shape.empty() || arg1_shape.empty())
    {
        auto& first = (arg0_shape.empty() ? inputs[0] : inputs[1]);
        auto& second = (arg0_shape.empty() ? inputs[1] : inputs[0]);

        TU << "{   // " << n->get_name() << "\n";
        TU.indent++;
        TU << "" << emit_vector(outputs[0]) << "\n    = ";
        TU << first.get_tensor().get_name() << "[0]\n    * " << emit_vector(second) << ";\n";
        TU.indent--;
        TU << "}\n";
    }
    else if ((arg0_shape.size() == 1) && (arg1_shape.size() == 1))
    {
        TU << "{   // " << n->get_name() << "\n";
        TU.indent++;
        TU << "" << emit_vector(outputs[0]) << " << \n"
           << "    " << emit_vector(inputs[0]) << ".dot("
           << "" << emit_vector(inputs[1]) << ");\n";
        TU.indent--;
        TU << "}\n";
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1))
    {
        auto arg0_layout = inputs[0].get_layout<DenseTensorViewLayout>();

        TU << "{   // " << n->get_name() << "\n";
        TU.indent++;
        TU << "" << emit_vector(outputs[0]) << " = \n"
           << "    " << emit_matrix(inputs[0]) << " * "
           << "" << emit_vector(inputs[1]) << ";\n";
        TU.indent--;
        TU << "}\n";
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2))
    {
        auto arg0_layout = inputs[0].get_layout<DenseTensorViewLayout>();
        auto arg1_layout = inputs[1].get_layout<DenseTensorViewLayout>();
        auto out_layout = outputs[0].get_layout<DenseTensorViewLayout>();

        // Emit an MKL SGEMM call if possible
        // clang-format off
        if (arg0_element_type == ngraph::element::Float32::element_type())
        {
            TU << "{   // " << n->get_name() << "\n";
            TU.indent++;
            TU << "cblas::cblas_sgemm("
               << "cblas::Layout::RowMajor, "
               << "cblas::Transpose::None, "
               << "cblas::Transpose::None, "
               << arg0_shape[0] << ", " << arg1_shape[1] << ", " << arg0_shape[1] << ",\n" <<
                "        1.0f, " << inputs[0].get_tensor().get_name() << ", " << max(1UL, arg0_shape[1]) << ", " << inputs[1].get_tensor().get_name() << ", " << max(1UL, arg1_shape[1]) << ", 0.0f,\n" <<
                "        " << outputs[0].get_tensor().get_name() << ", " << max(1UL, arg1_shape[1]) << ");\n";
            TU.indent--;
            TU << "}\n";
        }
        // clang-format on
        else
        {
            TU << "{   // " << n->get_name() << "\n";
            TU.indent++;
            TU << "" << emit_matrix(outputs[0]) << " = \n"
               << "    " << emit_matrix(inputs[0]) << " * "
               << "" << emit_matrix(inputs[1]) << ";\n";
            TU.indent--;
            TU << "}\n";
        }
    }
    else
    {
        throw ngraph_error("Dot product not implemented for given inputs");
    }
}

void Emitter::EmitMultiply(const ngraph::Node* n,
                           const std::vector<TensorViewInfo>& inputs,
                           const std::vector<TensorViewInfo>& outputs)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();
    string type = et.c_type_string();

    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "   " << emit_array1d(inputs[0]) << " *\n"
       << "   " << emit_array1d(inputs[1]) << ";\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitGetTupleElement(const ngraph::Node* n,
                                  const std::vector<TensorViewInfo>& inputs,
                                  const std::vector<TensorViewInfo>& outputs)
{
    auto get_tuple_element = static_cast<const op::GetTupleElement*>(n);
    auto result_tensor_type = dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
    assert(result_tensor_type);
    auto& result_element_type = result_tensor_type->get_element_type();
    string type = result_element_type.c_type_string();

    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << "memcpy(" << outputs[0].get_tensor().get_name() << ", "
       << inputs[get_tuple_element->get_n()].get_tensor().get_name() << ", "
       << outputs[0].get_tensor_view_layout()->get_size() *
              outputs[0].get_tensor_view_layout()->get_element_type().size()
       << ");\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitTuple(const ngraph::Node* n,
                        const std::vector<TensorViewInfo>& inputs,
                        const std::vector<TensorViewInfo>& outputs)
{
    assert(inputs.size() == outputs.size());

    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto& et = inputs.at(i).get_tensor_view_layout()->get_element_type();
        TU << "// call_frame->get_parameterized_tensor_view<" << et.c_type_string() << ">("
           << outputs.at(i).get_index() << ")->get_vector() =\n"
           << "//     call_frame->get_parameterized_tensor_view<" << et.c_type_string() << ">("
           << inputs.at(i).get_index() << ")->get_vector();\n";
        TU << "memcpy(" << outputs.at(i).get_tensor().get_name() << ", "
           << inputs.at(i).get_tensor().get_name() << ", "
           << outputs[i].get_tensor_view_layout()->get_size() *
                  outputs[i].get_tensor_view_layout()->get_element_type().size()
           << ");\n";
    }
    TU.indent--;
    TU += "}\n";
}

void Emitter::EmitAbs(const ngraph::Node* n,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n";
    TU << "Eigen::abs(" << emit_array1d(inputs[0]) << ");\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitConcat(const ngraph::Node* n,
                         const std::vector<TensorViewInfo>& inputs,
                         const std::vector<TensorViewInfo>& outputs)
{
    auto result_tensor_type = dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
    assert(result_tensor_type);

    auto result_shape = result_tensor_type->get_shape();

    if (result_shape.size() == 1)
    {
        TU << "{   // " << n->get_name() << "\n";
        TU.indent++;
        TU << "" << emit_vector(outputs[0], "out_vector") << ";\n";

        size_t concat_pos = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            TU << "out_vector.segment(" << concat_pos << ", "
               << inputs[i].get_tensor_view_layout()->get_shape().at(0) << ") << "
               << "" << emit_vector(inputs[i]) << ";\n";
            concat_pos += inputs[i].get_tensor_view_layout()->get_shape().at(0);
        }
        TU.indent--;
        TU << "}\n";
    }
    else if (result_shape.size() == 2)
    {
        auto out_layout = outputs[0].get_layout<DenseTensorViewLayout>();
        auto axis = (dynamic_cast<const op::Concat*>(n))->get_concatenation_axis();

        TU << "{   // " << n->get_name() << "\n";
        TU.indent++;
        TU << "" << emit_matrix(outputs[0], "out_matrix") << ";\n";

        size_t concat_pos[2]{0, 0};
        for (size_t i = 0; i < inputs.size(); i++)
        {
            auto arg_layout = inputs[i].get_layout<DenseTensorViewLayout>();
            auto& arg_shape = inputs[i].get_tensor_view_layout()->get_shape();

            TU << "out_matrix.block(" << concat_pos[0] << ", " << concat_pos[1] << ", "
               << arg_shape.at(0) << ", " << arg_shape.at(1) << ") << "
               << "" << emit_matrix(inputs[i]) << ";\n";

            concat_pos[axis] += arg_shape.at(axis);
        }

        TU.indent--;
        TU << "}\n";
    }
}

void Emitter::EmitDivide(const ngraph::Node* n,
                         const std::vector<TensorViewInfo>& inputs,
                         const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << " /\n"
       << "    " << emit_array1d(inputs[1]) << ";\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitEqual(const ngraph::Node* n,
                        const std::vector<TensorViewInfo>& inputs,
                        const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    (" << emit_array1d(inputs[0]) << " ==\n"
       << "    " << emit_array1d(inputs[1]) << ").template cast<char>();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitGreater(const ngraph::Node* n,
                          const std::vector<TensorViewInfo>& inputs,
                          const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << " xxx\n";
    TU.indent++;
    TU << "" << emit_array1d(outputs[0]) << " =\n"
       << "    (" << emit_array1d(inputs[0]) << " >\n"
       << "    " << emit_array1d(inputs[1]) << ").template cast<char>();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitGreaterEq(const ngraph::Node* n,
                            const std::vector<TensorViewInfo>& inputs,
                            const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << "" << emit_array1d(outputs[0]) << " =\n"
       << "    (" << emit_array1d(inputs[0]) << " >=\n"
       << "    " << emit_array1d(inputs[1]) << ").template cast<char>();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitLess(const ngraph::Node* n,
                       const std::vector<TensorViewInfo>& inputs,
                       const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << "" << emit_array1d(outputs[0]) << " =\n"
       << "    (" << emit_array1d(inputs[0]) << " <\n"
       << "    " << emit_array1d(inputs[1]) << ").template cast<char>();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitLessEq(const ngraph::Node* n,
                         const std::vector<TensorViewInfo>& inputs,
                         const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << "" << emit_array1d(outputs[0]) << " =\n"
       << "    (" << emit_array1d(inputs[0]) << " <=\n"
       << "    " << emit_array1d(inputs[1]) << ").template cast<char>();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitLog(const ngraph::Node* n,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    Eigen::log(" << emit_array1d(inputs[0]) << ");\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitMaximum(const ngraph::Node* n,
                          const std::vector<TensorViewInfo>& inputs,
                          const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "        " << emit_array1d(inputs[0]) << ".max(\n"
       << "        " << emit_array1d(inputs[1]) << ");\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitMinimum(const ngraph::Node* n,
                          const std::vector<TensorViewInfo>& inputs,
                          const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << ".min(\n"
       << "    " << emit_array1d(inputs[1]) << ");\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitNegative(const ngraph::Node* n,
                           const std::vector<TensorViewInfo>& inputs,
                           const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    -" << emit_array1d(inputs[0]) << ";\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitNotEqual(const ngraph::Node* n,
                           const std::vector<TensorViewInfo>& inputs,
                           const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << "" << emit_array1d(outputs[0]) << " =\n"
       << "    (" << emit_array1d(inputs[0]) << " !=\n"
       << "    " << emit_array1d(inputs[1]) << ").template cast<char>();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitSelect(const ngraph::Node* n,
                         const std::vector<TensorViewInfo>& inputs,
                         const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "   " << emit_array1d(inputs[0]) << "\n"
       << "    .select(" << emit_array1d(inputs[1]) << ",\n"
       << "       " << emit_array1d(inputs[2]) << ");\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitSubtract(const ngraph::Node* n,
                           const std::vector<TensorViewInfo>& inputs,
                           const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << " -\n"
       << "    " << emit_array1d(inputs[1]) << ";\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitParameterizedConstantBool(const ngraph::Node* n,
                                            const std::vector<TensorViewInfo>& inputs,
                                            const std::vector<TensorViewInfo>& outputs)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::Bool>*>(n)
                     ->get_value()
                     ->get_vector();

    TU << "// " << n->get_name() << " EmitParameterizedConstantBool\n";
    if (outputs[0].get_tensor().is_output())
    {
        // Special case where constant is stored directly in the output
        for (size_t i = 0; i < value.size(); i++)
        {
            TU << outputs[0].get_tensor().get_name() << "[" << i << "] = static_cast<char>("
               << (value[i] ? "true" : "false") << ");\n";
        }
    }
    else
    {
        TU << "// this should be const but eigen hates const :(\n";
        TU << "char " << outputs[0].get_tensor().get_name() << "[] = {\n";
        for (size_t i = 0; i < value.size(); i++)
        {
            if (i != 0)
            {
                TU << ",\n";
            }
            TU << "    " << (value[i] ? "true" : "false");
        }
        TU << "\n};";
    }
    TU << "\n";
}

static string format_float_as_string(float value)
{
    if (isnan(value))
    {
        return "NAN";
    }
    else if (isinf(value))
    {
        if (value > 0)
        {
            return "INFINITY";
        }
        else
        {
            return "-INFINITY";
        }
    }
    else
    {
        return to_string(value);
    }
}

void Emitter::EmitParameterizedConstantFloat32(const ngraph::Node* n,
                                               const std::vector<TensorViewInfo>& inputs,
                                               const std::vector<TensorViewInfo>& outputs)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::Float32>*>(n)
                     ->get_value()
                     ->get_vector();
    const char* type = "float";

    TU << "// " << n->get_name() << " EmitParameterizedConstantFloat32\n";
    if (outputs[0].get_tensor().is_output())
    {
        // Special case where constant is stored directly in the output
        for (size_t i = 0; i < value.size(); i++)
        {
            TU << outputs[0].get_tensor().get_name() << "[" << i << "] = static_cast<" << type
               << ">(" << format_float_as_string(value[i]) << ");\n";
        }
    }
    else
    {
        TU << "// this should be const but eigen hates const :(\n";
        TU << type << " " << outputs[0].get_tensor().get_name() << "[] = {\n";
        for (size_t i = 0; i < value.size(); i++)
        {
            if (i != 0)
            {
                TU << ",\n";
            }
            TU << "    " << format_float_as_string(value[i]);
        }
        TU << "\n};";
    }
    TU << "\n";
}

void Emitter::EmitParameterizedConstantInt8(const ngraph::Node* n,
                                            const std::vector<TensorViewInfo>& inputs,
                                            const std::vector<TensorViewInfo>& outputs)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::Int8>*>(n)
                     ->get_value()
                     ->get_vector();
    const char* type = "int8_t";

    TU << "// " << n->get_name() << " EmitParameterizedConstantInt8\n";
    if (outputs[0].get_tensor().is_output())
    {
        // Special case where constant is stored directly in the output
        for (size_t i = 0; i < value.size(); i++)
        {
            TU << outputs[0].get_tensor().get_name() << "[" << i << "] = static_cast<" << type
               << ">(" << static_cast<int>(value[i]) << ");\n";
        }
    }
    else
    {
        TU << "// this should be const but eigen hates const :(\n";
        TU << type << " " << outputs[0].get_tensor().get_name() << "[] = {\n";
        for (size_t i = 0; i < value.size(); i++)
        {
            if (i != 0)
            {
                TU << ",\n";
            }
            TU << "    " << value[i];
        }
        TU << "\n};";
    }
    TU << "\n";
}

void Emitter::EmitParameterizedConstantInt32(const ngraph::Node* n,
                                             const std::vector<TensorViewInfo>& inputs,
                                             const std::vector<TensorViewInfo>& outputs)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::Int32>*>(n)
                     ->get_value()
                     ->get_vector();
    const char* type = "int32_t";

    TU << "// " << n->get_name() << " EmitParameterizedConstantInt32\n";
    if (outputs[0].get_tensor().is_output())
    {
        // Special case where constant is stored directly in the output
        for (size_t i = 0; i < value.size(); i++)
        {
            TU << outputs[0].get_tensor().get_name() << "[" << i << "] = static_cast<" << type
               << ">(" << value[i] << ");\n";
        }
    }
    else
    {
        TU << "// this should be const but eigen hates const :(\n";
        TU << type << " " << outputs[0].get_tensor().get_name() << "[] = {\n";
        for (size_t i = 0; i < value.size(); i++)
        {
            if (i != 0)
            {
                TU << ",\n";
            }
            TU << "    " << value[i];
        }
        TU << "\n};";
    }
    TU << "\n";
}

void Emitter::EmitParameterizedConstantInt64(const ngraph::Node* n,
                                             const std::vector<TensorViewInfo>& inputs,
                                             const std::vector<TensorViewInfo>& outputs)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::Int64>*>(n)
                     ->get_value()
                     ->get_vector();
    const char* type = "int64_t";

    TU << "// " << n->get_name() << " EmitParameterizedConstantInt64\n";
    if (outputs[0].get_tensor().is_output())
    {
        // Special case where constant is stored directly in the output
        for (size_t i = 0; i < value.size(); i++)
        {
            TU << outputs[0].get_tensor().get_name() << "[" << i << "] = static_cast<" << type
               << ">(" << value[i] << ");\n";
        }
    }
    else
    {
        TU << "// this should be const but eigen hates const :(\n";
        TU << type << " " << outputs[0].get_tensor().get_name() << "[] = {\n";
        for (size_t i = 0; i < value.size(); i++)
        {
            if (i != 0)
            {
                TU << ",\n";
            }
            TU << "    " << value[i];
        }
        TU << "\n};";
    }
    TU << "\n";
}

void Emitter::EmitParameterizedConstantUInt8(const ngraph::Node* n,
                                             const std::vector<TensorViewInfo>& inputs,
                                             const std::vector<TensorViewInfo>& outputs)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::UInt8>*>(n)
                     ->get_value()
                     ->get_vector();
    const char* type = "uint8_t";

    TU << "// " << n->get_name() << " EmitParameterizedConstantUInt8\n";
    if (outputs[0].get_tensor().is_output())
    {
        // Special case where constant is stored directly in the output
        for (size_t i = 0; i < value.size(); i++)
        {
            TU << outputs[0].get_tensor().get_name() << "[" << i << "] = static_cast<" << type
               << ">(" << static_cast<uint>(value[i]) << ");\n";
        }
    }
    else
    {
        TU << "// this should be const but eigen hates const :(\n";
        TU << type << " " << outputs[0].get_tensor().get_name() << "[] = {\n";
        for (size_t i = 0; i < value.size(); i++)
        {
            if (i != 0)
            {
                TU << ",\n";
            }
            TU << "    " << value[i];
        }
        TU << "\n};";
    }
    TU << "\n";
}

void Emitter::EmitParameterizedConstantUInt32(const ngraph::Node* n,
                                              const std::vector<TensorViewInfo>& inputs,
                                              const std::vector<TensorViewInfo>& outputs)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::UInt32>*>(n)
                     ->get_value()
                     ->get_vector();
    const char* type = "uint32_t";

    TU << "// " << n->get_name() << " EmitParameterizedConstantUInt32\n";
    if (outputs[0].get_tensor().is_output())
    {
        // Special case where constant is stored directly in the output
        for (size_t i = 0; i < value.size(); i++)
        {
            TU << outputs[0].get_tensor().get_name() << "[" << i << "] = static_cast<" << type
               << ">(" << value[i] << ");\n";
        }
    }
    else
    {
        TU << "// this should be const but eigen hates const :(\n";
        TU << type << " " << outputs[0].get_tensor().get_name() << "[] = {\n";
        for (size_t i = 0; i < value.size(); i++)
        {
            if (i != 0)
            {
                TU << ",\n";
            }
            TU << "    " << value[i];
        }
        TU << "\n};";
    }
    TU << "\n";
}

void Emitter::EmitParameterizedConstantUInt64(const ngraph::Node* n,
                                              const std::vector<TensorViewInfo>& inputs,
                                              const std::vector<TensorViewInfo>& outputs)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::UInt64>*>(n)
                     ->get_value()
                     ->get_vector();
    const char* type = "uint64_t";

    TU << "// " << n->get_name() << " EmitParameterizedConstantUInt64\n";
    if (outputs[0].get_tensor().is_output())
    {
        // Special case where constant is stored directly in the output
        for (size_t i = 0; i < value.size(); i++)
        {
            TU << outputs[0].get_tensor().get_name() << "[" << i << "] = static_cast<" << type
               << ">(" << value[i] << ");\n";
        }
    }
    else
    {
        TU << "// this should be const but eigen hates const :(\n";
        TU << type << " " << outputs[0].get_tensor().get_name() << "[] = {\n";
        for (size_t i = 0; i < value.size(); i++)
        {
            if (i != 0)
            {
                TU << ",\n";
            }
            TU << "    " << value[i];
        }
        TU << "\n};";
    }
    TU << "\n";
}

void Emitter::EmitBroadcast(const ngraph::Node* n,
                            const std::vector<TensorViewInfo>& inputs,
                            const std::vector<TensorViewInfo>& outputs)
{
    auto broadcast = static_cast<const op::Broadcast*>(n);

    auto arg_tensor_type =
        dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type());
    assert(arg_tensor_type);

    auto result_tensor_type = dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
    assert(result_tensor_type);

    auto arg_shape = arg_tensor_type->get_shape();
    auto result_shape = result_tensor_type->get_shape();

    if (broadcast->get_broadcast_axes().empty())
    {
        TU << "{   // " << n->get_name() << "\n";
        TU.indent++;
        TU << "memcpy(" << outputs[0].get_tensor().get_name() << ", "
           << inputs[0].get_tensor().get_name() << ", "
           << outputs[0].get_tensor_view_layout()->get_size() *
                  outputs[0].get_tensor_view_layout()->get_element_type().size()
           << ");\n";
        TU.indent--;
        TU << "}\n";
    }
    else if (arg_shape.size() == 0)
    {
        TU << "{   // " << n->get_name() << "\n";
        TU.indent++;
        TU << "" << emit_array1d(outputs[0]) << " =\n"
           << "    " << emit_array1d(inputs[0]) << "(0, 0);\n";
        TU.indent--;
        TU << "}\n";
    }
    else if (arg_shape.size() == 1 && result_shape.size() == 2)
    {
        if (broadcast->get_broadcast_axes() == AxisSet{1})
        {
            auto out_layout = outputs[0].get_layout<DenseTensorViewLayout>();

            TU << "{   // " << n->get_name() << "\n";
            TU.indent++;
            TU << "" << emit_matrix(outputs[0]) << ".colwise() =\n"
               << "    " << emit_vector(inputs[0]) << ";\n";
            TU.indent--;
            TU << "}\n";
        }
        else if (broadcast->get_broadcast_axes() == AxisSet{0})
        {
            auto out_layout = outputs[0].get_layout<DenseTensorViewLayout>();

            TU << "{   // " << n->get_name() << "\n";
            TU.indent++;
            TU << "" << emit_matrix(outputs[0]) << ".rowwise() =\n"
               << "    " << emit_vector(inputs[0]) << ".transpose();\n";
            TU.indent--;
            TU << "}\n";
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
        throw ngraph_error("Broadcast not implemented for given inputs");
    }
}

void Emitter::EmitConvert(const ngraph::Node* n,
                          const std::vector<TensorViewInfo>& inputs,
                          const std::vector<TensorViewInfo>& outputs)
{
    auto arg = n->get_arguments().at(0);

    auto arg_tensor_type = dynamic_pointer_cast<const TensorViewType>(arg->get_value_type());
    assert(arg_tensor_type);

    auto result_tensor_type = dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
    assert(result_tensor_type);

    auto& result_element_type = result_tensor_type->get_element_type();

    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << "\n"
       << "    .template cast<" << result_element_type.c_type_string() << ">();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitConstant(const ngraph::Node* n,
                           const std::vector<TensorViewInfo>& inputs,
                           const std::vector<TensorViewInfo>& outputs)
{
    auto c = static_cast<const op::Constant*>(n);
    auto c_tensor_type = dynamic_pointer_cast<const TensorViewType>(c->get_value_type());
    assert(c_tensor_type);
    auto& c_element_type = c_tensor_type->get_element_type();
    auto c_value_strings = c->get_value_strings();

    TU << "{   // " << n->get_name() << " EmitConstant\n";
    TU.indent++;
    for (size_t i = 0; i < c_value_strings.size(); i++)
    {
        TU << outputs[0].get_tensor().get_name() << "[" << i << "] = static_cast<"
           << c_element_type.c_type_string() << ">(" << c_value_strings[i] << ");\n";
    }
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitReshape(const ngraph::Node* n,
                          const std::vector<TensorViewInfo>& inputs,
                          const std::vector<TensorViewInfo>& outputs)
{
    auto reshape = static_cast<const op::Reshape*>(n);

    auto arg_type = reshape->get_arguments().at(0)->get_value_type();
    auto arg_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(arg_type);
    assert(arg_tensor_view_type);
    auto arg_shape = arg_tensor_view_type->get_shape();
    auto arg_rank = arg_shape.size();

    auto result_type = reshape->get_value_type();
    auto result_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(result_type);
    assert(result_tensor_view_type);
    auto result_shape = result_tensor_view_type->get_shape();
    auto& result_element_type = result_tensor_view_type->get_element_type();

    auto input_order = reshape->get_input_order();

    bool same_layout = std::is_sorted(input_order.begin(), input_order.end());

    size_t result_shape_product = 1;
    for (auto i : result_shape)
    {
        result_shape_product *= i;
    }

    // If there is no layout change or we are just going from 1^n to 1^m or a zero-size tensor,
    //  we can just copy.
    if (same_layout || result_shape_product < 2)
    {
        TU << "{   // " << n->get_name() << " 1\n";
        TU.indent++;
        TU << "memcpy(" << outputs[0].get_tensor().get_name() << ", "
           << inputs[0].get_tensor().get_name() << ", "
           << outputs[0].get_tensor_view_layout()->get_size() *
                  outputs[0].get_tensor_view_layout()->get_element_type().size()
           << ");\n";
        TU.indent--;
        TU << "}\n";
    }
    // If there *is* a layout change in the 2D case, we transpose the input.
    else if (arg_rank == 2)
    {
        auto arg0_layout = inputs[0].get_layout<DenseTensorViewLayout>();
        auto out_layout = outputs[0].get_layout<DenseTensorViewLayout>();

        // Emit an MKL transpose call if possible
        // clang-format off
        if (result_element_type == ngraph::element::Float32::element_type())
        {
            TU << "{   // " << n->get_name() << " 2\n";
            TU.indent++;
            TU << "mkl::MKL_Somatcopy('R', 'T', " << to_string(arg_shape[0]) << ",\n" <<
                "                   " << to_string(arg_shape[1]) << ", 1.0f,\n" <<
                "                   " << inputs[0].get_tensor().get_name() << ", "
                << to_string(arg_shape[1]) << ",\n" <<
                "                   " << outputs[0].get_tensor().get_name()
                << ", " << to_string(arg_shape[0]) << ");\n";
                TU.indent--;
                TU << "}\n";
        }
        // clang-format on
        else
        {
            TU << "{   // " << n->get_name() << " 3\n";
            TU.indent++;
            TU << "" << emit_matrix(outputs[0]) << " =\n"
               << "        " << emit_matrix(inputs[0]) << ".transpose();\n";
            TU.indent--;
            TU << "}\n";
        }
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        throw ngraph_error(
            "Axis permutation in reshape is not implemented yet for tensors with rank>2");
    }
}

void Emitter::EmitFunctionCall(const ngraph::Node* n,
                               const std::vector<TensorViewInfo>& inputs,
                               const std::vector<TensorViewInfo>& outputs)
{
    auto function_call = static_cast<const op::FunctionCall*>(n);
    shared_ptr<Function> function = function_call->get_function();

    TU << "{   // Call " << function->get_name() << "\n";
    TU.indent++;
    generate_call(inputs, outputs, function);
    TU.indent--;
    TU << "}\n";
}

// TODO: This and other ops include comments/notes that
// we don't want to just copy-paste here. Figure out a better way
// or just point to ngvm/external_function.cpp with a note that
// the compiled version of these ops is intended to have semantics identical
// to what's seen there (for now atleast)

void Emitter::EmitReduce(const ngraph::Node* n,
                         const std::vector<TensorViewInfo>& inputs,
                         const std::vector<TensorViewInfo>& outputs)
{
    auto reduce = static_cast<const op::Reduce*>(n);
    auto reduction_function = reduce->get_function();

    auto reductee_type = reduce->get_arguments().at(0)->get_value_type();
    auto reductee_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(reductee_type);
    assert(reductee_tensor_view_type);
    auto reductee_shape = reductee_tensor_view_type->get_shape();

    auto f_result_type = reduction_function->get_result_type();
    auto f_result_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(f_result_type);
    assert(f_result_tensor_view_type);
    auto& f_result_element_type = f_result_tensor_view_type->get_element_type();

    auto result_type = reduce->get_value_type();
    auto result_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(result_type);
    assert(result_tensor_view_type);
    auto result_shape = result_tensor_view_type->get_shape();

    auto& reduction_axes = reduce->get_reduction_axes();

    auto arg0_layout = inputs[0].get_layout<DenseTensorViewLayout>();

    // Trivial case: no reduction axes (this includes the scalar-reductee case).
    if (reduction_axes.empty())
    {
        TU << "{   // " << n->get_name() << " 1\n";
        TU.indent++;
        TU << "memcpy(" << outputs[0].get_tensor().get_name() << ", "
           << inputs[0].get_tensor().get_name() << ", "
           << outputs[0].get_tensor_view_layout()->get_size() *
                  outputs[0].get_tensor_view_layout()->get_element_type().size()
           << ");\n";
        TU.indent--;
        TU << "}\n";
    }
    // Behavior for zero-size axes bears some explanation here. XLA's reduce
    // operator provides an "base" element (usually, but not necessarily,
    // an identity element) that it apparently *may* choose to insert anywhere
    // in the reduction any number of times. For example, given:
    //
    //   reduce{{1,2,3},b,+)
    //
    // any of the following are valid reductions (I think!):
    //
    //   b+(b+1+2)+3
    //   b+(1+(2+3))
    //   (1+2)+3 (I think!)
    //
    // etc. Here we will choose never to instantiate the base element, which
    // works well with Eigen's default behavior for non-zero-length axes. The
    // exceptional case is when we reduce on a zero-length axis. In this case,
    // Eigen's default behavior is to put a zero in the output,  which is not
    // what we want, so we detect that case here and override with a copy
    // instruction (for reduce-to-scalar) or a broadcast (for reduce-to-vector)
    // from the base element.
    //
    // What I'm actually not sure about is whether the identity element is
    // required to appear at least once. If so, this will need to be reworked,
    // assuming we actually want to mimic XLA's semantics that closely, which
    // we may not.
    else if ((reductee_shape.size() == 1 && reduction_axes == AxisSet{0}) ||
             (reductee_shape.size() == 2 && reduction_axes == AxisSet{0, 1}))
    {
        if (reductee_shape.at(0) == 0 || (reductee_shape.size() == 2 && reductee_shape.at(1) == 0))
        {
            TU << "{   // " << n->get_name() << " 2\n";
            TU.indent++;
            TU << "memcpy(" << outputs[0].get_tensor().get_name() << ", "
               << inputs[1].get_tensor().get_name() << ", "
               << outputs[0].get_tensor_view_layout()->get_size() *
                      outputs[0].get_tensor_view_layout()->get_element_type().size()
               << ");\n";
            TU.indent--;
            TU << "}\n";
        }
        else
        {
            TU << "{   // " << n->get_name() << " 3\n";
            TU.indent++;
            string type = f_result_element_type.c_type_string();
            TU << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
            TU.indent++;
            TU << "\n";
            TU << type << " result;\n";
            TU << "void* inputs[] = {&x, &y};\n";
            TU << "void* outputs[] = {&result};\n";
            TU << reduction_function->get_name() << "(inputs, outputs);\n";
            TU << "return result;\n";
            TU.indent--;
            TU << "};\n";
            TU << "" << emit_array1d(outputs[0]) << " =\n"
               << "    " << emit_array1d(inputs[0]) << ".redux(f);\n";
            TU.indent--;
            TU << "}\n";
        }
    }
    else if (reductee_shape.size() == 2 && reduction_axes == AxisSet{1})
    {
        if (reductee_shape.at(1) == 0)
        {
            TU << "{   // " << n->get_name() << " 4\n";
            TU.indent++;
            TU << "" << emit_array1d(outputs[0]) << " =\n"
               << "    " << emit_array1d(inputs[1]) << "(0, 0);\n";
            TU.indent--;
            TU << "}\n";
        }
        else
        {
            // std::shared_ptr<CallFrame> cf =
            //     std::dynamic_pointer_cast<CallFrame>(external->make_call_frame());
            // ef->get_callees().emplace_back(cf);

            TU << "{   // " << n->get_name() << " 5\n";
            TU.indent++;
            string type = f_result_element_type.c_type_string();
            TU << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
            TU.indent++;
            TU << "\n";
            TU << type << " result;\n";
            TU << "void* inputs[] = {&x, &y};\n";
            TU << "void* outputs[] = {&result};\n";
            TU << reduction_function->get_name() << "(inputs, outputs);\n";
            TU << "return result;\n";
            TU.indent--;
            TU << "};\n";
            TU << "" << emit_vector(outputs[0]) << " =\n"
               << "        " << emit_matrix(inputs[0]) << ".rowwise().redux(f);\n";
            TU.indent--;
            TU << "}\n";
        }
    }
    else if (reductee_shape.size() == 2 && reduction_axes == AxisSet{0})
    {
        if (reductee_shape.at(0) == 0)
        {
            TU << "{   // " << n->get_name() << " 6\n";
            TU.indent++;
            TU << "" << emit_array1d(outputs[0]) << " =\n"
               << "    " << emit_array1d(inputs[1]) << "(0, 0);\n";
            TU.indent--;
            TU << "}\n";
        }
        else
        {
            TU << "{   // " << n->get_name() << " 7\n";
            TU.indent++;
            string type = f_result_element_type.c_type_string();
            TU << "auto f = [](" << type << " x, " << type << " y) -> " << type << "\n{";
            TU.indent++;
            TU << "\n";
            TU << type << " result;\n";
            TU << "void* inputs[] = {&x, &y};\n";
            TU << "void* outputs[] = {&result};\n";
            TU << reduction_function->get_name() << "(inputs, outputs);\n";
            TU << "return result;\n";
            TU.indent--;
            TU << "};\n";
            TU << "" << emit_vector(outputs[0]) << " =\n"
               << "    " << emit_matrix(inputs[0]) << ".colwise().redux(f);\n";
            TU.indent--;
            TU << "}\n";
        }
    }
    else
    {
        throw ngraph_error("Reduce: only vectors and matrices are currently supported");
    }
}

void Emitter::EmitSign(const ngraph::Node* n,
                       const std::vector<TensorViewInfo>& inputs,
                       const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << ".sign();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitSlice(const ngraph::Node* n,
                        const std::vector<TensorViewInfo>& inputs,
                        const std::vector<TensorViewInfo>& outputs)
{
    auto slice = static_cast<const op::Slice*>(n);

    for (auto d : slice->get_step())
    {
        if (1 != d)
        {
            throw ngraph_error("Slice does not support non-unit step yet");
        }
    }

    auto arg_type = slice->get_arguments().at(0)->get_value_type();
    auto arg_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(arg_type);
    assert(arg_tensor_view_type);
    auto arg_shape = arg_tensor_view_type->get_shape();
    auto arg_rank = arg_shape.size();

    auto& lower_bounds = slice->get_lower_bounds();
    auto& upper_bounds = slice->get_upper_bounds();

    // Scalar slice is necessarily just a copy.
    if (arg_rank == 0)
    {
        TU << "{   // " << n->get_name() << " 1\n";
        TU.indent++;
        TU << "memcpy(" << outputs[0].get_tensor().get_name() << ", "
           << inputs[0].get_tensor().get_name() << ", "
           << outputs[0].get_tensor_view_layout()->get_size() *
                  outputs[0].get_tensor_view_layout()->get_element_type().size()
           << ");\n";
        TU.indent--;
        TU << "}\n";
    }
    else if (arg_rank == 1)
    {
        TU << "{   // " << n->get_name() << " 2\n";
        TU.indent++;
        TU << "" << emit_vector(outputs[0]) << " =\n"
           << "    " << emit_vector(inputs[0]) << ".segment(\n"
           << "        " << to_string(lower_bounds[0]) << ", "
           << to_string(upper_bounds[0] - lower_bounds[0]) << ");\n";
        TU.indent--;
        TU << "}\n";
    }
    else if (arg_rank == 2)
    {
        auto arg0_layout = inputs[0].get_layout<DenseTensorViewLayout>();
        auto out_layout = outputs[0].get_layout<DenseTensorViewLayout>();

        TU << "{   // " << n->get_name() << " 3\n";
        TU.indent++;
        TU << "" << emit_matrix(outputs[0]) << " = \n"
           << "        " << emit_matrix(inputs[0]) << ".block(" << to_string(lower_bounds[0])
           << ", " << to_string(lower_bounds[1]) << ",\n"
           << "        " << to_string(upper_bounds[0] - lower_bounds[0]) << ",\n"
           << "        " << to_string(upper_bounds[1] - lower_bounds[1]) << ");\n";
        TU.indent--;
        TU << "}\n";
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        throw ngraph_error("Slice is not implemented yet for tensors with rank>2");
    }
}

void Emitter::EmitSum(const ngraph::Node* n,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{
    auto s = static_cast<const op::Sum*>(n);
    auto s_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(s->get_value_type());
    assert(s_tensor_view_type);
    auto s_shape = s_tensor_view_type->get_shape();

    auto arg = s->get_arguments().at(0);
    auto arg_type = arg->get_value_type();
    auto arg_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(arg_type);
    assert(arg_tensor_view_type);
    auto arg_shape = arg_tensor_view_type->get_shape();
    auto arg_rank = arg_shape.size();

    auto& reduction_axes = s->get_reduction_axes();

    // Trivial case: no reduction axes.
    if (reduction_axes.size() == 0)
    {
        TU << "{   // " << n->get_name() << "\n";
        TU.indent++;
        TU << "memcpy(" << outputs[0].get_tensor().get_name() << ", "
           << inputs[0].get_tensor().get_name() << ", "
           << outputs[0].get_tensor_view_layout()->get_size() *
                  outputs[0].get_tensor_view_layout()->get_element_type().size()
           << ");\n";
        TU.indent--;
        TU << "}\n";
    }
    // Full reduction? Then sum to scalar.
    else if ((arg_rank == 1 && reduction_axes == AxisSet{0}) ||
             (arg_rank == 2 && reduction_axes == AxisSet{0, 1}))
    {
        TU << "{   // " << n->get_name() << "\n";
        TU.indent++;
        TU << "" << emit_array1d(outputs[0]) << " =\n"
           << "    " << emit_array1d(inputs[0]) << ".sum();\n";
        TU.indent--;
        TU << "}\n";
    }
    else if (arg_rank == 2 && reduction_axes == AxisSet{1})
    {
        auto arg0_layout = inputs[0].get_layout<DenseTensorViewLayout>();

        TU << "{   // " << n->get_name() << "\n";
        TU.indent++;
        TU << "" << emit_vector(outputs[0]) << " =\n"
           << "    " << emit_matrix(inputs[0]) << ".rowwise().sum();\n";
        TU.indent--;
        TU << "}\n";
    }
    else if (arg_rank == 2 && reduction_axes == AxisSet{0})
    {
        auto arg0_layout = inputs[0].get_layout<DenseTensorViewLayout>();

        TU << "{   // " << n->get_name() << "\n";
        TU.indent++;
        TU << "" << emit_vector(outputs[0]) << " =\n"
           << "    " << emit_matrix(inputs[0]) << ".colwise().sum();\n";
        TU.indent--;
        TU << "}\n";
    }
    else
    {
        throw ngraph_error("Sum: only vectors and matrices are currently supported");
    }
}

void Emitter::EmitExp(const ngraph::Node* n,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << ".exp();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitSin(const ngraph::Node* n,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << ".sin();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitSinh(const ngraph::Node* n,
                       const std::vector<TensorViewInfo>& inputs,
                       const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << ".sinh();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitCos(const ngraph::Node* n,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << ".cos();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitCosh(const ngraph::Node* n,
                       const std::vector<TensorViewInfo>& inputs,
                       const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << ".cosh();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitTan(const ngraph::Node* n,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << ".tan();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitTanh(const ngraph::Node* n,
                       const std::vector<TensorViewInfo>& inputs,
                       const std::vector<TensorViewInfo>& outputs)
{
    // Eigen's generic_fast_tanh_float<float> is currently miscompiled by Clang/LLVM
    // so we fall-back to std::tanh
    // TODO: Implement our own internal fast/approximate tanh if this actually gets used
    // by models
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << "for (size_t i=0; i<" << outputs[0].get_tensor_view_layout()->get_size() << "; i++)\n";
    TU << "{\n";
    TU << "    " << outputs[0].get_tensor().get_name() << "[i] = std::tanh("
       << inputs[0].get_tensor().get_name() << "[i]);\n";
    TU << "}\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitAsin(const ngraph::Node* n,
                       const std::vector<TensorViewInfo>& inputs,
                       const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << ".asin();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitAcos(const ngraph::Node* n,
                       const std::vector<TensorViewInfo>& inputs,
                       const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << ".acos();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitAtan(const ngraph::Node* n,
                       const std::vector<TensorViewInfo>& inputs,
                       const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " =\n"
       << "    " << emit_array1d(inputs[0]) << ".atan();\n";
    TU.indent--;
    TU << "}\n";
}

void Emitter::EmitPower(const ngraph::Node* n,
                        const std::vector<TensorViewInfo>& inputs,
                        const std::vector<TensorViewInfo>& outputs)
{
    TU << "{   // " << n->get_name() << "\n";
    TU.indent++;
    TU << emit_array1d(outputs[0]) << " = \n";
    TU.indent++;
    TU << emit_array1d(inputs[0]) << ".pow(\n ";
    TU << emit_array1d(inputs[1]) << ");\n";
    TU.indent -= 2;
    TU << "}\n";
}

void Emitter::EmitReplaceSlice(const ngraph::Node* n,
                               const std::vector<TensorViewInfo>& inputs,
                               const std::vector<TensorViewInfo>& outputs)
{
    auto replace_slice = static_cast<const op::Slice*>(n);

    for (auto d : replace_slice->get_step())
    {
        if (1 != d)
        {
            throw ngraph_error("Replace-slice does not support non-unit step yet");
        }
    }

    auto arg0_type = replace_slice->get_arguments().at(0)->get_value_type();
    auto arg0_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(arg0_type);
    assert(arg0_tensor_view_type);
    auto arg0_shape = arg0_tensor_view_type->get_shape();
    auto arg0_rank = arg0_shape.size();

    auto arg1_type = replace_slice->get_arguments().at(1)->get_value_type();
    auto arg1_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(arg1_type);
    assert(arg1_tensor_view_type);
    auto arg1_shape = arg1_tensor_view_type->get_shape();
    auto arg1_rank = arg1_shape.size();

    auto& lower_bounds = replace_slice->get_lower_bounds();
    auto& upper_bounds = replace_slice->get_upper_bounds();

    // Scalar slice is necessarily just a copy.
    if (arg0_rank == 0)
    {
        TU << "{   // " << n->get_name() << " 1\n";
        TU.indent++;
        TU << "memcpy(" << outputs[0].get_tensor().get_name() << ", "
           << inputs[1].get_tensor().get_name() << ", "
           << outputs[0].get_tensor_view_layout()->get_size() *
                  outputs[0].get_tensor_view_layout()->get_element_type().size()
           << ");\n";
        TU.indent--;
        TU << "}\n";
    }
    else if (arg0_rank == 1)
    {
        TU << "{   // " << n->get_name() << " 2\n";
        TU.indent++;
        TU << "" << emit_vector(outputs[0]) << " =\n"
           << "    " << emit_vector(inputs[0]) << ";\n"
           << "" << emit_vector(outputs[0]) << ".segment(\n"
           << "    " << to_string(lower_bounds[0]) << ", "
           << to_string(upper_bounds[0] - lower_bounds[0]) << ") =\n"
           << "    " << emit_vector(inputs[1]) << ";\n";
        TU.indent--;
        TU << "}\n";
    }
    else if (arg0_rank == 2)
    {
        auto arg0_layout = inputs[0].get_layout<DenseTensorViewLayout>();
        auto out_layout = outputs[0].get_layout<DenseTensorViewLayout>();

        TU << "{   // " << n->get_name() << " 3\n";
        TU.indent++;
        TU << "" << emit_matrix(outputs[0]) << " =\n"
           << "    " << emit_matrix(inputs[0]) << ";\n"
           << "" << emit_matrix(outputs[0]) << ".block(\n"
           << "        " << to_string(lower_bounds[0]) << ",\n"
           << "        " << to_string(lower_bounds[1]) << ",\n"
           << "        " << to_string(upper_bounds[0] - lower_bounds[0]) << ",\n"
           << "        " << to_string(upper_bounds[1] - lower_bounds[1]) << ") =\n"
           << "    " << emit_matrix(inputs[1]) << ";\n";
        TU.indent--;
        TU << "}\n";
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        throw ngraph_error("Replace-slice is not implemented yet for tensors with rank>2");
    }
}

//------------------------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------------------------

void Emitter::generate_call(const std::vector<TensorViewInfo>& inputs,
                            const std::vector<TensorViewInfo>& outputs,
                            shared_ptr<Function> function)
{
    vector<string> input_names;
    vector<string> output_names;

    for (const TensorViewInfo& input : inputs)
    {
        input_names.push_back(input.get_tensor().get_name());
    }

    for (const TensorViewInfo& output : outputs)
    {
        output_names.push_back(output.get_tensor().get_name());
    }

    TU << "void* inputs[] =\n{";
    TU.indent++;
    TU << "\n" << join(input_names, ",\n");
    TU.indent--;
    TU << "\n};\n";

    TU << "void* outputs[] =\n{";
    TU.indent++;
    TU << "\n" << join(output_names, ",\n");
    TU.indent--;
    TU << "\n};\n";

    TU << "\n";
    TU << function->get_name() << "(inputs, outputs);\n";
}

static string format_name(const string& name)
{
    string rc;
    if (!name.empty())
    {
        rc = " " + name;
    }
    return rc;
}

string Emitter::emit_vector(const TensorViewInfo& tvi, const string& name)
{
    stringstream ss;

    const element::Type& et = tvi.get_tensor_view()->get_value_type()->get_element_type();
    ss << "EigenVector<" << et.c_type_string() << ">" << format_name(name) << "("
       << tvi.get_tensor().get_name() << ", " << eigen_vector_format(tvi) << ")";
    return ss.str();
}

string Emitter::emit_array1d(const TensorViewInfo& tvi, const string& name)
{
    stringstream ss;

    const element::Type& et = tvi.get_tensor_view()->get_value_type()->get_element_type();
    ss << "EigenArray1d<" << et.c_type_string() << ">" << format_name(name) << "("
       << tvi.get_tensor().get_name() << ", " << eigen_vector_format(tvi) << ")";
    return ss.str();
}

string Emitter::emit_matrix(const TensorViewInfo& tvi, const string& name)
{
    stringstream ss;
    auto layout = tvi.get_layout<DenseTensorViewLayout>();

    const element::Type& et = tvi.get_tensor_view()->get_value_type()->get_element_type();
    ss << "EigenMatrix<" << et.c_type_string() << ">" << format_name(name) << "("
       << tvi.get_tensor().get_name() << ", "
       << eigen_matrix_format(layout->get_shape(), layout->get_strides()) << ")";
    return ss.str();
}
