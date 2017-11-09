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

#define TI(x) type_index(typeid(x))

static unordered_map<type_index, string> element_type_names = {
    {TI(ngraph::element::Bool), "Bool"},
    {TI(ngraph::element::Float32), "Float32"},
    {TI(ngraph::element::Int8), "Int8"},
    {TI(ngraph::element::Int32), "Int32"},
    {TI(ngraph::element::Int64), "Int64"},
    {TI(ngraph::element::UInt8), "UInt8"},
    {TI(ngraph::element::UInt32), "UInt32"},
    {TI(ngraph::element::UInt64), "UInt64"}};

#define EIGEN_VECTOR_FORMAT(x) "fmt::V{" + to_string(x) + "}"

string eigen_vector_format(const runtime::TensorViewInfo& info)
{
    stringstream ss;
    ss << "fmt::V{" << info.get_layout<DenseTensorViewLayout>()->get_size() << "}";
    return ss.str();
}

static std::string EIGEN_MATRIX_FORMAT(const ngraph::Shape& shape, const ngraph::Strides& strides)
{
    stringstream ss;
    ss << "fmt::M{{" << join(shape) << "}, {" << join(strides) << "}}";
    return ss.str();
}

void Emitter::EMITTER_DECL(EmitNop)
{
}

void Emitter::EMITTER_DECL(EmitAdd)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();
    string type = et.c_type_string();

    TU.indent++;
    TU << "{ // " << n->get_name() << "\n";
    TU.indent++;
    TU << "EigenArray1d<" << type << ">(" << outputs[0].get_tensor().get_name() << ", "
       << eigen_vector_format(outputs[0]) << ") =\n";
    TU.indent++;
    TU << "EigenArray1d<" << type << ">(" << inputs[0].get_tensor().get_name() << ", "
       << eigen_vector_format(inputs[0]) << ") +\n";
    TU << "EigenArray1d<" << type << ">(" << inputs[1].get_tensor().get_name() << ", "
       << eigen_vector_format(inputs[1]) << ");\n";
    TU.indent -= 2;
    TU << "}\n";
    TU.indent--;
}

void Emitter::EMITTER_DECL(EmitDot)
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

        TU += "    {\n"
              "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(arg0_element_type)] +
                       ">(" + to_string(second.get_index()) + ");\n"
              "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(arg0_element_type)] +
                       ">(" + to_string(outputs[0].get_index()) + ");\n"
              "        EigenVector<" + element_type_names[TI(arg0_element_type)] +
                       ">(out, " EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") = "
              "call_frame->get_tensor_view_data<" + element_type_names[TI(arg0_element_type)] +
                       ">(" + to_string(first.get_index()) + ")[0] * EigenVector<" +
                       element_type_names[TI(arg0_element_type)] +
                       ">(arg1, " EIGEN_VECTOR_FORMAT(second.get_layout<DenseTensorViewLayout>()->get_size()) ");\n"
              "    }\n";
    }
    else if ((arg0_shape.size() == 1) && (arg1_shape.size() == 1))
    {
        TU += "    {\n"
              "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(arg0_element_type)] + ">(" +
                       to_string(inputs[0].get_index()) + ");\n"
              "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(arg0_element_type)] + ">(" +
                       to_string(inputs[1].get_index()) + ");\n"
              "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(arg0_element_type)] + ">(" +
                       to_string(outputs[0].get_index()) + ");\n"
              "        EigenVector<" + element_type_names[TI(arg0_element_type)] + ">(out, "
                       EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") << \n"
              "        EigenVector<" + element_type_names[TI(arg0_element_type)] + ">(arg0, "
                       EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").dot("
              "EigenVector<" + element_type_names[TI(arg0_element_type)] + ">(arg1, "
                       EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) "));\n"
              "    }\n";
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1))
    {
        auto arg0_layout = inputs[0].get_layout<DenseTensorViewLayout>();

        TU += "    {\n"
              "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(arg0_element_type)] + ">(" +
                       to_string(inputs[0].get_index()) + ");\n"
              "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(arg0_element_type)] + ">(" +
                       to_string(inputs[1].get_index()) + ");\n"
              "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(arg0_element_type)] + ">(" +
                       to_string(outputs[0].get_index()) + ");\n"
              "        EigenVector<" + element_type_names[TI(arg0_element_type)] + ">(out, "
                       EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") = \n"
              "        EigenMatrix<" + element_type_names[TI(arg0_element_type)] + ">(arg0, " +
                       EIGEN_MATRIX_FORMAT(arg0_layout->get_shape(), arg0_layout->get_strides()) + ") * "
              "EigenVector<" + element_type_names[TI(arg0_element_type)] + ">(arg1, "
                       EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ");\n"
              "    }\n";
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
            TU +=
                "    {\n"
                "        auto arg0 = call_frame->get_tensor_view_data<" +
                element_type_names[TI(arg0_element_type)] + ">(" + to_string(inputs[0].get_index()) +
                ");\n"
                "        auto arg1 = call_frame->get_tensor_view_data<" +
                element_type_names[TI(arg0_element_type)] + ">(" + to_string(inputs[1].get_index()) +
                ");\n"
                "        auto out  = call_frame->get_tensor_view_data<" +
                element_type_names[TI(arg0_element_type)] + ">(" + to_string(outputs[0].get_index()) +
                ");\n"
                "        cblas::cblas_sgemm(cblas::Layout::RowMajor, cblas::Transpose::None, cblas::Transpose::None, " +
                to_string(arg0_shape[0]) + ", " + to_string(arg1_shape[1]) + ", " + to_string(arg0_shape[1]) + ",\n"
                "        1.0f, arg0, " + to_string(max(1UL, arg0_shape[1])) + ", arg1, " + to_string(max(1UL, arg1_shape[1])) + ", 0.0f,\n"
                "        out, " + to_string(max(1UL, arg1_shape[1])) + ");\n"
                "    }\n";
        }
        // clang-format on
        else
        {
            TU +=
                "    {\n"
                "        auto arg0 = call_frame->get_tensor_view_data<" +
                element_type_names[TI(arg0_element_type)] + ">(" +
                to_string(inputs[0].get_index()) +
                ");\n"
                "        auto arg1 = call_frame->get_tensor_view_data<" +
                element_type_names[TI(arg0_element_type)] + ">(" +
                to_string(inputs[1].get_index()) +
                ");\n"
                "        auto out  = call_frame->get_tensor_view_data<" +
                element_type_names[TI(arg0_element_type)] + ">(" +
                to_string(outputs[0].get_index()) +
                ");\n"
                "        EigenMatrix<" +
                element_type_names[TI(arg0_element_type)] + ">(out, " +
                EIGEN_MATRIX_FORMAT(out_layout->get_shape(), out_layout->get_strides()) +
                ") = \n"
                "        EigenMatrix<" +
                element_type_names[TI(arg0_element_type)] + ">(arg0, " +
                EIGEN_MATRIX_FORMAT(arg0_layout->get_shape(), arg0_layout->get_strides()) +
                ") * "
                "EigenMatrix<" +
                element_type_names[TI(arg0_element_type)] + ">(arg1, " +
                EIGEN_MATRIX_FORMAT(arg1_layout->get_shape(), arg1_layout->get_strides()) +
                ");\n"
                "    }\n";
        }
    }
    else
    {
        throw ngraph_error("Dot product not implemented for given inputs");
    }
}

void Emitter::EMITTER_DECL(EmitMultiply)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        // auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        // auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        // auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") *\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ");\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitGetTupleElement)
{
    auto get_tuple_element = static_cast<const op::GetTupleElement*>(n);
    auto result_tensor_type = dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
    assert(result_tensor_type);
    auto& result_element_type = result_tensor_type->get_element_type();

    TU +=
        "    {\n"
        "        call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(result_element_type)] + ">(" + to_string(outputs.at(0).get_index()) +
        ")->get_vector() =\n"
        "        call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(result_element_type)] + ">(" +
        to_string(inputs.at(get_tuple_element->get_n()).get_index()) +
        ")->get_vector();\n"
        "    }\n";
}

void Emitter::EMITTER_DECL(EmitTuple)
{
    assert(inputs.size() == outputs.size());

    TU += "    {\n";
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto& et = inputs.at(i).get_tensor_view_layout()->get_element_type();
        TU += "        call_frame->get_parameterized_tensor_view<" + element_type_names[TI(et)] +
              ">(" + to_string(outputs.at(i).get_index()) +
              ")->get_vector() =\n"
              "        call_frame->get_parameterized_tensor_view<" +
              element_type_names[TI(et)] + ">(" + to_string(inputs.at(i).get_index()) +
              ")->get_vector();\n";
    }
    TU += "    }\n";
}

void Emitter::EMITTER_DECL(EmitAbs)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        Eigen::abs(EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) "));\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitConcat)
{
    auto result_tensor_type = dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
    assert(result_tensor_type);

    auto result_shape = result_tensor_type->get_shape();
    auto& result_element_type = result_tensor_type->get_element_type();

    if (result_shape.size() == 1)
    {
        TU +=
            "    {\n"
            "        auto out  = call_frame->get_tensor_view_data<" +
            element_type_names[TI(result_element_type)] + ">(" + to_string(outputs[0].get_index()) +
            ");\n"
            "        EigenVector<" +
            element_type_names[TI(result_element_type)] +
            "> out_vector(out, " EIGEN_VECTOR_FORMAT(
                outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ");\n";

        size_t concat_pos = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            TU += "        out_vector.segment(" + to_string(concat_pos) + ", " +
                  to_string(inputs[i].get_tensor_view_layout()->get_shape().at(0)) +
                  ") << "
                  "EigenVector<" +
                  element_type_names[TI(result_element_type)] +
                  ">(call_frame->"
                  "get_tensor_view_data<" +
                  element_type_names[TI(result_element_type)] + ">(" +
                  to_string(inputs[i].get_index()) +
                  "), " EIGEN_VECTOR_FORMAT(
                      inputs[i].get_layout<DenseTensorViewLayout>()->get_size()) ");\n";
            concat_pos += inputs[i].get_tensor_view_layout()->get_shape().at(0);
        }

        TU += "    }\n";
    }
    else if (result_shape.size() == 2)
    {
        auto out_layout = outputs[0].get_layout<DenseTensorViewLayout>();
        auto axis = (dynamic_cast<const op::Concat*>(n))->get_concatenation_axis();

        TU +=
            "    {\n"
            "        auto out  = call_frame->get_tensor_view_data<" +
            element_type_names[TI(result_element_type)] + ">(" + to_string(outputs[0].get_index()) +
            ");\n"
            "        EigenMatrix<" +
            element_type_names[TI(result_element_type)] + "> out_matrix(out, " +
            EIGEN_MATRIX_FORMAT(out_layout->get_shape(), out_layout->get_strides()) + ");\n";

        size_t concat_pos[2]{0, 0};
        for (size_t i = 0; i < inputs.size(); i++)
        {
            auto arg_layout = inputs[i].get_layout<DenseTensorViewLayout>();
            auto& arg_shape = inputs[i].get_tensor_view_layout()->get_shape();

            TU += "        out_matrix.block(" + to_string(concat_pos[0]) + ", " +
                  to_string(concat_pos[1]) + ", " + to_string(arg_shape.at(0)) + ", " +
                  to_string(arg_shape.at(1)) +
                  ") << "
                  "EigenMatrix<" +
                  element_type_names[TI(result_element_type)] +
                  ">(call_frame->"
                  "get_tensor_view_data<" +
                  element_type_names[TI(result_element_type)] + ">(" +
                  to_string(inputs[i].get_index()) + "), " +
                  EIGEN_MATRIX_FORMAT(arg_layout->get_shape(), arg_layout->get_strides()) + ");\n";

            concat_pos[axis] += arg_shape.at(axis);
        }

        TU += "    }\n";
    }
}

void Emitter::EMITTER_DECL(EmitDivide)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") /\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ");\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitEqual)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<Bool>(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<Bool>(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        (EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") ==\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ")).template cast<char>();\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitGreater)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<Bool>(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<Bool>(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        (EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") >\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ")).template cast<char>();\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitGreaterEq)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<Bool>(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<Bool>(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        (EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") >=\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ")).template cast<char>();\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitLess)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<Bool>(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<Bool>(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        (EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") <\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ")).template cast<char>();\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitLessEq)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<Bool>(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<Bool>(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        (EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") <=\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ")).template cast<char>();\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitLog)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        Eigen::log(EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) "));\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitMaximum)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").max(\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) "));\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitMinimum)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").min(\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) "));\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitNegative)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        -EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ");\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitNotEqual)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<Bool>(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<Bool>(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        (EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") !=\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ")).template cast<char>();\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitSelect)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(1)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<Bool>(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        auto arg2 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[2].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        EigenArray1d<Bool>(arg0, " EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ")\n"
          ".select(EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) "),\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg2, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) "));\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitSubtract)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") -\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ");\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitParameterizedConstantBool)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::Bool>*>(n)
                     ->get_value()
                     ->get_vector();

    TU +=
        "    {\n"
        "        call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(ngraph::element::Bool)] + ">(" + to_string(outputs[0].get_index()) +
        ")->get_vector() = std::vector<" + element_type_names[TI(ngraph::element::Bool)] +
        "::type>{";

    for (size_t i = 0; i < value.size(); i++)
    {
        if (i)
            TU += ", ";
        if (value[i])
        {
            TU += "true";
        }
        else
        {
            TU += "false";
        }
    }

    TU += "};\n    }\n";
}

void Emitter::EMITTER_DECL(EmitParameterizedConstantFloat32)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::Float32>*>(n)
                     ->get_value()
                     ->get_vector();

    TU +=
        "    {\n"
        "        call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(ngraph::element::Float32)] + ">(" +
        to_string(outputs[0].get_index()) + ")->get_vector() = std::vector<" +
        element_type_names[TI(ngraph::element::Float32)] + "::type>{";

    for (size_t i = 0; i < value.size(); i++)
    {
        if (i)
            TU += ", ";
        TU += to_string(value[i]) + "f";
    }

    TU += "};\n    }\n";
}

void Emitter::EMITTER_DECL(EmitParameterizedConstantInt8)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::Int8>*>(n)
                     ->get_value()
                     ->get_vector();

    TU +=
        "    {\n"
        "        call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(ngraph::element::Int8)] + ">(" + to_string(outputs[0].get_index()) +
        ")->get_vector() = std::vector<" + element_type_names[TI(ngraph::element::Int8)] +
        "::type>{";

    for (size_t i = 0; i < value.size(); i++)
    {
        if (i)
            TU += ", ";
        TU += to_string(value[i]);
    }

    TU += "};\n    }\n";
}

void Emitter::EMITTER_DECL(EmitParameterizedConstantInt32)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::Int32>*>(n)
                     ->get_value()
                     ->get_vector();

    TU +=
        "    {\n"
        "        call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(ngraph::element::Int32)] + ">(" + to_string(outputs[0].get_index()) +
        ")->get_vector() = std::vector<" + element_type_names[TI(ngraph::element::Int32)] +
        "::type>{";

    for (size_t i = 0; i < value.size(); i++)
    {
        if (i)
            TU += ", ";
        TU += to_string(value[i]);
    }

    TU += "};\n    }\n";
}

void Emitter::EMITTER_DECL(EmitParameterizedConstantInt64)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::Int64>*>(n)
                     ->get_value()
                     ->get_vector();

    TU +=
        "    {\n"
        "        call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(ngraph::element::Int64)] + ">(" + to_string(outputs[0].get_index()) +
        ")->get_vector() = std::vector<" + element_type_names[TI(ngraph::element::Int64)] +
        "::type>{";

    for (size_t i = 0; i < value.size(); i++)
    {
        if (i)
            TU += ", ";
        TU += to_string(value[i]);
    }

    TU += "};\n    }\n";
}

void Emitter::EMITTER_DECL(EmitParameterizedConstantUInt8)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::UInt8>*>(n)
                     ->get_value()
                     ->get_vector();

    TU +=
        "    {\n"
        "        call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(ngraph::element::UInt8)] + ">(" + to_string(outputs[0].get_index()) +
        ")->get_vector() = std::vector<" + element_type_names[TI(ngraph::element::UInt8)] +
        "::type>{";

    for (size_t i = 0; i < value.size(); i++)
    {
        if (i)
            TU += ", ";
        TU += to_string(value[i]);
    }

    TU += "};\n    }\n";
}

void Emitter::EMITTER_DECL(EmitParameterizedConstantUInt32)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::UInt32>*>(n)
                     ->get_value()
                     ->get_vector();

    TU +=
        "    {\n"
        "        call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(ngraph::element::UInt32)] + ">(" + to_string(outputs[0].get_index()) +
        ")->get_vector() = std::vector<" + element_type_names[TI(ngraph::element::UInt32)] +
        "::type>{";

    for (size_t i = 0; i < value.size(); i++)
    {
        if (i)
            TU += ", ";
        TU += to_string(value[i]);
    }

    TU += "};\n    }\n";
}

void Emitter::EMITTER_DECL(EmitParameterizedConstantUInt64)
{
    auto value = dynamic_cast<const op::ParameterizedConstant<ngraph::element::UInt64>*>(n)
                     ->get_value()
                     ->get_vector();

    TU +=
        "    {\n"
        "        call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(ngraph::element::UInt64)] + ">(" + to_string(outputs[0].get_index()) +
        ")->get_vector() = std::vector<" + element_type_names[TI(ngraph::element::UInt64)] +
        "::type>{";

    for (size_t i = 0; i < value.size(); i++)
    {
        if (i)
            TU += ", ";
        TU += to_string(value[i]);
    }

    TU += "};\n    }\n";
}

void Emitter::EMITTER_DECL(EmitBroadcast)
{
    auto broadcast = static_cast<const op::Broadcast*>(n);

    auto arg_tensor_type =
        dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type());
    assert(arg_tensor_type);

    auto result_tensor_type = dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
    assert(result_tensor_type);

    auto arg_shape = arg_tensor_type->get_shape();
    auto result_shape = result_tensor_type->get_shape();
    auto& result_element_type = result_tensor_type->get_element_type();

    if (broadcast->get_broadcast_axes().empty())
    {
        TU +=
            "    {\n"
            "        call_frame->get_parameterized_tensor_view<" +
            element_type_names[TI(result_element_type)] + ">(" + to_string(outputs[0].get_index()) +
            ")->get_vector() =\n"
            "        call_frame->get_parameterized_tensor_view<" +
            element_type_names[TI(result_element_type)] + ">(" + to_string(inputs[0].get_index()) +
            ")->get_vector();\n"
            "    }\n";
    }
    else if (arg_shape.size() == 0)
    {
        TU += "    {\n"
              "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(result_element_type)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
              "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(result_element_type)] + ">(" + to_string(outputs[0].get_index()) + ");\n"
              "        EigenArray1d<" + element_type_names[TI(result_element_type)] + ">(out, "
                       EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
              "        EigenArray1d<" + element_type_names[TI(result_element_type)] + ">(arg0, "
                       EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ")(0, 0);\n"
              "    }\n";
    }
    else if (arg_shape.size() == 1 && result_shape.size() == 2)
    {
        if (broadcast->get_broadcast_axes() == AxisSet{1})
        {
            auto out_layout = outputs[0].get_layout<DenseTensorViewLayout>();

            TU += "    {\n"
                  "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(result_element_type)] +
                           ">(" + to_string(inputs[0].get_index()) + ");\n"
                  "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(result_element_type)] +
                           ">(" + to_string(outputs[0].get_index()) + ");\n"
                  "        EigenMatrix<" + element_type_names[TI(result_element_type)] + ">(out, " +
                           EIGEN_MATRIX_FORMAT(out_layout->get_shape(), out_layout->get_strides()) + ").colwise() =\n"
                  "        EigenVector<" + element_type_names[TI(result_element_type)] + ">(arg0, "
                           EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ");\n"
                  "    }\n";
        }
        else if (broadcast->get_broadcast_axes() == AxisSet{0})
        {
            auto out_layout = outputs[0].get_layout<DenseTensorViewLayout>();

            TU += "    {\n"
                  "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(result_element_type)] +
                           ">(" + to_string(inputs[0].get_index()) + ");\n"
                  "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(result_element_type)] +
                           ">(" + to_string(outputs[0].get_index()) + ");\n"
                  "        EigenMatrix<" + element_type_names[TI(result_element_type)] + ">(out, " +
                           EIGEN_MATRIX_FORMAT(out_layout->get_shape(), out_layout->get_strides()) + ").rowwise() =\n"
                  "        EigenVector<" + element_type_names[TI(result_element_type)] + ">(arg0, "
                           EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").transpose();\n"
                  "    }\n";
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

void Emitter::EMITTER_DECL(EmitConvert)
{
    auto arg = n->get_arguments().at(0);

    auto arg_tensor_type = dynamic_pointer_cast<const TensorViewType>(arg->get_value_type());
    assert(arg_tensor_type);

    auto& arg_element_type = arg_tensor_type->get_element_type();

    auto result_tensor_type = dynamic_pointer_cast<const TensorViewType>(n->get_value_type());
    assert(result_tensor_type);

    auto& result_element_type = result_tensor_type->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(arg_element_type)] +
                               ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(result_element_type)] +
                               ">(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<" + element_type_names[TI(result_element_type)] + ">(out, "
                       EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        EigenArray1d<" + element_type_names[TI(arg_element_type)] + ">(arg0, "
                       EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ")\n"
          ".template cast<typename " + element_type_names[TI(result_element_type)] + "::type>();\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitConstant)
{
    auto c = static_cast<const op::Constant*>(n);
    auto c_tensor_type = dynamic_pointer_cast<const TensorViewType>(c->get_value_type());
    assert(c_tensor_type);
    auto& c_element_type = c_tensor_type->get_element_type();
    auto c_value_strings = c->get_value_strings();

    TU +=
        "    {\n"
        "        call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(c_element_type)] + ">(" + to_string(outputs[0].get_index()) +
        ")->get_vector() = std::vector<" + element_type_names[TI(c_element_type)] + "::type>{";

    for (size_t i = 0; i < c_value_strings.size(); i++)
    {
        if (i)
            TU += ", ";
        TU += c_value_strings[i];
    }

    TU += "};\n    }\n";
}

void Emitter::EMITTER_DECL(EmitReshape)
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

    // If there is no layout change or we are just going from 1^n to 1^m or a zero-size tensor, we can just copy.
    if (same_layout || result_shape_product < 2)
    {
        TU +=
            "    {\n"
            "        call_frame->get_parameterized_tensor_view<" +
            element_type_names[TI(result_element_type)] + ">(" +
            to_string(outputs.at(0).get_index()) +
            ")->get_vector() =\n"
            "        call_frame->get_parameterized_tensor_view<" +
            element_type_names[TI(result_element_type)] + ">(" +
            to_string(inputs.at(0).get_index()) +
            ")->get_vector();\n"
            "    }\n";
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
            TU +=
                "    {\n"
                "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(result_element_type)] +
                ">(" + to_string(inputs[0].get_index()) + ");\n"
                "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(result_element_type)] +
                ">(" + to_string(outputs[0].get_index()) + ");\n"
                "        mkl::MKL_Somatcopy('R', 'T', " + to_string(arg_shape[0]) + ",\n"
                "                          " + to_string(arg_shape[1]) + ", 1.0f,\n"
                "                           arg0, " + to_string(arg_shape[1]) + ",\n"
                "                           out, " + to_string(arg_shape[0]) + ");\n"
                "    }\n";
        }
        // clang-format on
        else
        {
            TU +=
                "    {\n"
                "        auto arg0 = call_frame->get_tensor_view_data<" +
                element_type_names[TI(result_element_type)] + ">(" +
                to_string(inputs[0].get_index()) +
                ");\n"
                "        auto out  = call_frame->get_tensor_view_data<" +
                element_type_names[TI(result_element_type)] + ">(" +
                to_string(outputs[0].get_index()) +
                ");\n"
                "        EigenMatrix<" +
                element_type_names[TI(result_element_type)] + ">(out, " +
                EIGEN_MATRIX_FORMAT(out_layout->get_shape(), out_layout->get_strides()) +
                ") =\n"
                "        EigenMatrix<" +
                element_type_names[TI(result_element_type)] + ">(arg0, " +
                EIGEN_MATRIX_FORMAT(arg0_layout->get_shape(), arg0_layout->get_strides()) +
                ").transpose();\n"
                "    }\n";
        }
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        throw ngraph_error(
            "Axis permutation in reshape is not implemented yet for tensors with rank>2");
    }
}

void Emitter::EMITTER_DECL(EmitFunctionCall)
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

    std::shared_ptr<CallFrame> cf =
        std::dynamic_pointer_cast<CallFrame>(external->make_call_frame());

    ef->get_callees().emplace_back(cf);

    TU +=
        "    {\n"
        "        auto cf = callees.at(" +
        to_string(ef->get_callees().size() - 1) +
        ");\n"
        "        std::vector<std::shared_ptr<ngraph::runtime::Value>> inputs;\n"
        "        std::vector<std::shared_ptr<ngraph::runtime::Value>> outputs;\n";

    for (const auto& in : inputs)
    {
        TU += "        inputs.emplace_back(call_frame->get_tensor_view(" +
              to_string(in.get_index()) + "));\n";
    }
    for (const auto& out : outputs)
    {
        TU += "        outputs.emplace_back(call_frame->get_tensor_view(" +
              to_string(out.get_index()) + "));\n";
    }

    TU +=
        "        (*cf)(inputs, outputs);\n"
        "    }\n";
}

// TODO: This and other ops include comments/notes that
// we don't want to just copy-paste here. Figure out a better way
// or just point to ngvm/external_function.cpp with a note that
// the compiled version of these ops is intended to have semantics identical
// to what's seen there (for now atleast)

void Emitter::EMITTER_DECL(EmitReduce)
{
    auto reduce = static_cast<const op::Reduce*>(n);
    auto reduction_function = reduce->get_reduction_function();

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
        TU +=
            "    {\n"
            "        call_frame->get_parameterized_tensor_view<" +
            element_type_names[TI(f_result_element_type)] + ">(" +
            to_string(outputs.at(0).get_index()) +
            ")->get_vector() =\n"
            "        call_frame->get_parameterized_tensor_view<" +
            element_type_names[TI(f_result_element_type)] + ">(" +
            to_string(inputs.at(0).get_index()) +
            ")->get_vector();\n"
            "    }\n";
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
            TU +=
                "    {\n"
                "        call_frame->get_parameterized_tensor_view<" +
                element_type_names[TI(f_result_element_type)] + ">(" +
                to_string(outputs.at(0).get_index()) +
                ")->get_vector() =\n"
                "        call_frame->get_parameterized_tensor_view<" +
                element_type_names[TI(f_result_element_type)] + ">(" +
                to_string(inputs.at(1).get_index()) +
                ")->get_vector();\n"
                "    }\n";
        }
        else
        {
            std::shared_ptr<CallFrame> cf =
                std::dynamic_pointer_cast<CallFrame>(external->make_call_frame());
            ef->get_callees().emplace_back(cf);

            TU +=
                "    {\n"
                "        using ET = " + element_type_names[TI(f_result_element_type)] + ";\n"
                "        auto cf = callees.at(" + to_string(ef->get_callees().size() - 1) + ");\n"
                "        auto f = [cf](typename ET::type x, typename ET::type y) -> typename ET::type {\n"
                "            auto tx = ngraph::runtime::make_tensor<ET>(ngraph::Shape{});\n"
                "            *tx = std::vector<typename ET::type>({x});\n"
                "            auto ty = ngraph::runtime::make_tensor<ET>(ngraph::Shape{});\n"
                "            *ty = std::vector<typename ET::type>({y});\n"
                "            auto tr = ngraph::runtime::make_tensor<ET>(ngraph::Shape{});\n"
                "            (*cf)({tx, ty}, {tr});\n"
                "            return tr->get_vector()[0];\n"
                "        };\n"
                "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(f_result_element_type)] +
                ">(" + to_string(inputs[0].get_index()) + ");\n"
                "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(f_result_element_type)] +
                ">(" + to_string(outputs[0].get_index()) + ");\n"
                "        EigenArray1d<" + element_type_names[TI(f_result_element_type)] + ">(out, "
                EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
                "        EigenArray1d<" + element_type_names[TI(f_result_element_type)] + ">(arg0, "
                EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").redux(f);\n"
                "    }\n";
        }
    }
    else if (reductee_shape.size() == 2 && reduction_axes == AxisSet{1})
    {
        if (reductee_shape.at(1) == 0)
        {
            TU += "    {\n"
                "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(f_result_element_type)] +
                ">(" + to_string(inputs[1].get_index()) + ");\n"
                "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(f_result_element_type)] +
                ">(" + to_string(outputs[0].get_index()) + ");\n"
                "        EigenArray1d<" + element_type_names[TI(f_result_element_type)] + ">(out, "
                EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
                "        EigenArray1d<" + element_type_names[TI(f_result_element_type)] + ">(arg1, "
                EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ")(0, 0);\n"
                "    }\n";
        }
        else
        {
            std::shared_ptr<CallFrame> cf =
                std::dynamic_pointer_cast<CallFrame>(external->make_call_frame());
            ef->get_callees().emplace_back(cf);

            TU +=
                "    {\n"
                "        using ET = " + element_type_names[TI(f_result_element_type)] + ";\n"
                "        auto cf = callees.at(" + to_string(ef->get_callees().size() - 1) + ");\n"
                "        auto f = [cf](typename ET::type x, typename ET::type y) -> typename ET::type {\n"
                "            auto tx = ngraph::runtime::make_tensor<ET>(ngraph::Shape{});\n"
                "            *tx = std::vector<typename ET::type>({x});\n"
                "            auto ty = ngraph::runtime::make_tensor<ET>(ngraph::Shape{});\n"
                "            *ty = std::vector<typename ET::type>({y});\n"
                "            auto tr = ngraph::runtime::make_tensor<ET>(ngraph::Shape{});\n"
                "            (*cf)({tx, ty}, {tr});\n"
                "            return tr->get_vector()[0];\n"
                "        };\n"
                "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(f_result_element_type)] +
                ">(" + to_string(inputs[0].get_index()) + ");\n"
                "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(f_result_element_type)] +
                ">(" + to_string(outputs[0].get_index()) + ");\n"
                "        EigenVector<" + element_type_names[TI(f_result_element_type)] + ">(out, "
                EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
                "        EigenMatrix<" + element_type_names[TI(f_result_element_type)] + ">(arg0, " +
                EIGEN_MATRIX_FORMAT(arg0_layout->get_shape(), arg0_layout->get_strides()) + ").rowwise().redux(f);\n"
                "    }\n";
        }
    }
    else if (reductee_shape.size() == 2 && reduction_axes == AxisSet{0})
    {
        if (reductee_shape.at(0) == 0)
        {
            TU += "    {\n"
                "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(f_result_element_type)] +
                ">(" + to_string(inputs[1].get_index()) + ");\n"
                "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(f_result_element_type)] +
                ">(" + to_string(outputs[0].get_index()) + ");\n"
                "        EigenArray1d<" + element_type_names[TI(f_result_element_type)] + ">(out, "
                EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
                "        EigenArray1d<" + element_type_names[TI(f_result_element_type)] + ">(arg1, "
                EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ")(0, 0);\n"
                "    }\n";
        }
        else
        {
            std::shared_ptr<CallFrame> cf =
                std::dynamic_pointer_cast<CallFrame>(external->make_call_frame());
            ef->get_callees().emplace_back(cf);

            TU +=
                "    {\n"
                "        using ET = " + element_type_names[TI(f_result_element_type)] + ";\n"
                "        auto cf = callees.at(" + to_string(ef->get_callees().size() - 1) + ");\n"
                "        auto f = [cf](typename ET::type x, typename ET::type y) -> typename ET::type {\n"
                "            auto tx = ngraph::runtime::make_tensor<ET>(ngraph::Shape{});\n"
                "            *tx = std::vector<typename ET::type>({x});\n"
                "            auto ty = ngraph::runtime::make_tensor<ET>(ngraph::Shape{});\n"
                "            *ty = std::vector<typename ET::type>({y});\n"
                "            auto tr = ngraph::runtime::make_tensor<ET>(ngraph::Shape{});\n"
                "            (*cf)({tx, ty}, {tr});\n"
                "            return tr->get_vector()[0];\n"
                "        };\n"
                "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(f_result_element_type)] +
                ">(" + to_string(inputs[0].get_index()) + ");\n"
                "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(f_result_element_type)] +
                ">(" + to_string(outputs[0].get_index()) + ");\n"
                "        EigenVector<" + element_type_names[TI(f_result_element_type)] + ">(out, "
                EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
                "        EigenMatrix<" + element_type_names[TI(f_result_element_type)] + ">(arg0, " +
                EIGEN_MATRIX_FORMAT(arg0_layout->get_shape(), arg0_layout->get_strides()) + ").colwise().redux(f);\n"
                "    }\n";
        }
    }
    else
    {
        throw ngraph_error("Reduce: only vectors and matrices are currently supported");
    }
}

void Emitter::EMITTER_DECL(EmitSign)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU += "    {\n"
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(outputs[0].get_index()) + ");\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
                   EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").sign();\n"
          "    }\n";
}

void Emitter::EMITTER_DECL(EmitSlice)
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
    auto& arg_element_type = arg_tensor_view_type->get_element_type();

    auto& lower_bounds = slice->get_lower_bounds();
    auto& upper_bounds = slice->get_upper_bounds();

    // Scalar slice is necessarily just a copy.
    if (arg_rank == 0)
    {
        TU +=
            "    {\n"
            "        call_frame->get_parameterized_tensor_view<" +
            element_type_names[TI(arg_element_type)] + ">(" + to_string(outputs.at(0).get_index()) +
            ")->get_vector() =\n"
            "        call_frame->get_parameterized_tensor_view<" +
            element_type_names[TI(arg_element_type)] + ">(" + to_string(inputs.at(0).get_index()) +
            ")->get_vector();\n"
            "    }\n";
    }
    else if (arg_rank == 1)
    {
        TU +=
            "    {\n"
            "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(arg_element_type)] +
            ">(" + to_string(inputs[0].get_index()) + ");\n"
            "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(arg_element_type)] +
            ">(" + to_string(outputs[0].get_index()) + ");\n"
            "        EigenVector<" + element_type_names[TI(arg_element_type)] +
            ">(out, " EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
            "        EigenVector<" + element_type_names[TI(arg_element_type)] +
            ">(arg0, " EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").segment(\n"
            "        " + to_string(lower_bounds[0]) + ", " + to_string(upper_bounds[0] - lower_bounds[0]) + ");\n"
            "    }\n";
    }
    else if (arg_rank == 2)
    {
        auto arg0_layout = inputs[0].get_layout<DenseTensorViewLayout>();
        auto out_layout = outputs[0].get_layout<DenseTensorViewLayout>();

        TU +=
            "    {\n"
            "        auto arg0 = call_frame->get_tensor_view_data<" +
            element_type_names[TI(arg_element_type)] + ">(" + to_string(inputs[0].get_index()) +
            ");\n"
            "        auto out  = call_frame->get_tensor_view_data<" +
            element_type_names[TI(arg_element_type)] + ">(" + to_string(outputs[0].get_index()) +
            ");\n"
            "        EigenMatrix<" +
            element_type_names[TI(arg_element_type)] + ">(out, " +
            EIGEN_MATRIX_FORMAT(out_layout->get_shape(), out_layout->get_strides()) +
            ") = \n"
            "        EigenMatrix<" +
            element_type_names[TI(arg_element_type)] + ">(arg0, " +
            EIGEN_MATRIX_FORMAT(arg0_layout->get_shape(), arg0_layout->get_strides()) + ").block(" +
            to_string(lower_bounds[0]) + ", " + to_string(lower_bounds[1]) +
            ",\n"
            "        " +
            to_string(upper_bounds[0] - lower_bounds[0]) +
            ",\n"
            "        " +
            to_string(upper_bounds[1] - lower_bounds[1]) +
            ");\n"
            "    }\n";
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        throw ngraph_error("Slice is not implemented yet for tensors with rank>2");
    }
}

void Emitter::EMITTER_DECL(EmitSum)
{
    auto s = static_cast<const op::Sum*>(n);
    auto s_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(s->get_value_type());
    assert(s_tensor_view_type);
    auto& s_element_type = s_tensor_view_type->get_element_type();
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
        TU +=
            "    {\n"
            "        call_frame->get_parameterized_tensor_view<" +
            element_type_names[TI(s_element_type)] + ">(" + to_string(outputs.at(0).get_index()) +
            ")->get_vector() =\n"
            "        call_frame->get_parameterized_tensor_view<" +
            element_type_names[TI(s_element_type)] + ">(" + to_string(inputs.at(0).get_index()) +
            ")->get_vector();\n"
            "    }\n";
    }
    // Full reduction? Then sum to scalar.
    else if ((arg_rank == 1 && reduction_axes == AxisSet{0}) ||
             (arg_rank == 2 && reduction_axes == AxisSet{0, 1}))
    {
        TU +=
            "    {\n"
            "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(s_element_type)] +
            ">(" + to_string(inputs[0].get_index()) + ");\n"
            "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(s_element_type)] +
            ">(" + to_string(outputs[0].get_index()) + ");\n"
            "        EigenArray1d<" + element_type_names[TI(s_element_type)] + ">(out, "
            EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
            "        EigenArray1d<" + element_type_names[TI(s_element_type)] + ">(arg0, "
            EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").sum();\n"
            "    }\n";
    }
    else if (arg_rank == 2 && reduction_axes == AxisSet{1})
    {
        auto arg0_layout = inputs[0].get_layout<DenseTensorViewLayout>();

        TU +=
            "    {\n"
            "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(s_element_type)] +
            ">(" + to_string(inputs[0].get_index()) + ");\n"
            "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(s_element_type)] +
            ">(" + to_string(outputs[0].get_index()) + ");\n"
            "        EigenVector<" + element_type_names[TI(s_element_type)] + ">(out, "
            EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
            "        EigenMatrix<" + element_type_names[TI(s_element_type)] + ">(arg0, " +
            EIGEN_MATRIX_FORMAT(arg0_layout->get_shape(), arg0_layout->get_strides()) + ").rowwise().sum();\n"
            "    }\n";
    }
    else if (arg_rank == 2 && reduction_axes == AxisSet{0})
    {
        auto arg0_layout = inputs[0].get_layout<DenseTensorViewLayout>();

        TU +=
            "    {\n"
            "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(s_element_type)] +
            ">(" + to_string(inputs[0].get_index()) + ");\n"
            "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(s_element_type)] +
            ">(" + to_string(outputs[0].get_index()) + ");\n"
            "        EigenVector<" + element_type_names[TI(s_element_type)] + ">(out, "
            EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
            "        EigenMatrix<" + element_type_names[TI(s_element_type)] + ">(arg0, " +
            EIGEN_MATRIX_FORMAT(arg0_layout->get_shape(), arg0_layout->get_strides()) + ").colwise().sum();\n"
            "    }\n";
    }
    else
    {
        throw ngraph_error("Sum: only vectors and matrices are currently supported");
    }
}

void Emitter::EMITTER_DECL(EmitExp)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU +=
        "    {\n"
        "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(inputs[0].get_index()) + ");\n"
        "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(outputs[0].get_index()) + ");\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
        EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
        EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").exp();\n"
        "    }\n";
}

void Emitter::EMITTER_DECL(EmitSin)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU +=
        "    {\n"
        "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(inputs[0].get_index()) + ");\n"
        "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(outputs[0].get_index()) + ");\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
        EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
        EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").sin();\n"
        "    }\n";
}

void Emitter::EMITTER_DECL(EmitSinh)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU +=
        "    {\n"
        "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(inputs[0].get_index()) + ");\n"
        "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(outputs[0].get_index()) + ");\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
        EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
        EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").sinh();\n"
        "    }\n";
}

void Emitter::EMITTER_DECL(EmitCos)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU +=
        "    {\n"
        "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(inputs[0].get_index()) + ");\n"
        "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(outputs[0].get_index()) + ");\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
        EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
        EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").cos();\n"
        "    }\n";
}

void Emitter::EMITTER_DECL(EmitCosh)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU +=
        "    {\n"
        "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(inputs[0].get_index()) + ");\n"
        "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(outputs[0].get_index()) + ");\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
        EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
        EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").cosh();\n"
        "    }\n";
}

void Emitter::EMITTER_DECL(EmitTan)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU +=
        "    {\n"
        "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(inputs[0].get_index()) + ");\n"
        "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(outputs[0].get_index()) + ");\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
        EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
        EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").tan();\n"
        "    }\n";
}

void Emitter::EMITTER_DECL(EmitTanh)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    // Eigen's generic_fast_tanh_float<float> is currently miscompiled by Clang/LLVM
    // so we fall-back to std::tanh
    // TODO: Implement our own internal fast/approximate tanh if this actually gets used
    // by models
    TU +=
        "    {\n"
        "        auto& arg0 = call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) +
        ")->get_vector();\n"
        "        auto& out  = call_frame->get_parameterized_tensor_view<" +
        element_type_names[TI(et)] + ">(" + to_string(outputs[0].get_index()) +
        ")->get_vector();\n"
        "        std::transform(arg0.begin(), arg0.end(), out.begin(), [](" +
        element_type_names[TI(et)] + "::type x) -> " + element_type_names[TI(et)] +
        "::type { return std::tanh(x); });\n"
        "    }\n";
}

void Emitter::EMITTER_DECL(EmitAsin)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU +=
        "    {\n"
        "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(inputs[0].get_index()) + ");\n"
        "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(outputs[0].get_index()) + ");\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
        EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
        EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").asin();\n"
        "    }\n";
}

void Emitter::EMITTER_DECL(EmitAcos)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU +=
        "    {\n"
        "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(inputs[0].get_index()) + ");\n"
        "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(outputs[0].get_index()) + ");\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
        EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
        EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").acos();\n"
        "    }\n";
}

void Emitter::EMITTER_DECL(EmitAtan)
{
    const element::Type& et =
        (dynamic_pointer_cast<const TensorViewType>(n->get_arguments().at(0)->get_value_type()))
            ->get_element_type();

    TU +=
        "    {\n"
        "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(inputs[0].get_index()) + ");\n"
        "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" +
        to_string(outputs[0].get_index()) + ");\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(out, "
        EIGEN_VECTOR_FORMAT(outputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") =\n"
        "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg0, "
        EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").atan();\n"
        "    }\n";
}
