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

#include <iostream>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/get_tuple_element.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/runtime/cpu/emitter.hpp"
#include "ngraph/runtime/cpu/external_function.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"

using namespace std;
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

static std::string EIGEN_MATRIX_FORMAT(const ngraph::Shape& shape, const ngraph::Strides& strides)
{
    std::string I;
    for (size_t i = 0; i < shape.size(); i++)
    {
        if (!i)
        {
            I += "fmt::M{{" + to_string(shape[i]);
        }
        else
        {
            I += ", " + to_string(shape[i]);
        }
    }
    I += "}, ";
    for (size_t i = 0; i < strides.size(); i++)
    {
        if (!i)
        {
            I += "{" + to_string(strides[i]);
        }
        else
        {
            I += ", " + to_string(strides[i]);
        }
    }
    I += "}}";
    return I;
}

void Emitter::EMITTER_DECL(EmitNop)
{
}

void Emitter::EMITTER_DECL(EmitAdd)
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
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ") +\n"
          "        EigenArray1d<" + element_type_names[TI(et)] + ">(arg1, "
                   EIGEN_VECTOR_FORMAT(inputs[1].get_layout<DenseTensorViewLayout>()->get_size()) ");\n"
          "    }\n";
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
        else
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
          "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[0].get_index()) + ");\n"
          "        auto arg1 = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(inputs[1].get_index()) + ");\n"
          "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(et)] + ">(" + to_string(outputs[0].get_index()) + ");\n"
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
                   EIGEN_VECTOR_FORMAT(inputs[0].get_layout<DenseTensorViewLayout>()->get_size()) ").max("
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
        ")->get_vector() = std::vector<" + element_type_names[TI(c_element_type)] +
        "::type>{";

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
            element_type_names[TI(result_element_type)] + ">(" + to_string(outputs.at(0).get_index()) +
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

        TU += "    {\n"
            "        auto arg0 = call_frame->get_tensor_view_data<" + element_type_names[TI(result_element_type)] +
            ">(" + to_string(inputs[0].get_index()) + ");\n"
            "        auto out  = call_frame->get_tensor_view_data<" + element_type_names[TI(result_element_type)] +
            ">(" + to_string(outputs[0].get_index()) + ");\n"
            "        EigenMatrix<" + element_type_names[TI(result_element_type)] + ">(out, " +
            EIGEN_MATRIX_FORMAT(out_layout->get_shape(), out_layout->get_strides()) + ") =\n"
            "        EigenMatrix<" + element_type_names[TI(result_element_type)] + ">(arg0, " +
            EIGEN_MATRIX_FORMAT(arg0_layout->get_shape(), arg0_layout->get_strides()) + ").transpose();\n"
            "    }\n";
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        throw ngraph_error(
            "Axis permutation in reshape is not implemented yet for tensors with rank>2");
    }
}
