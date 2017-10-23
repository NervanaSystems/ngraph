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
#include <vector>
#include <typeindex>
#include <string>
#include <unordered_map>

#include "ngraph/node.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"
#include "ngraph/runtime/cpu/external_function.hpp"
#include "ngraph/runtime/cpu/emitter.hpp"

using namespace std;
using namespace ngraph::runtime::cpu;

using ngraph::descriptor::layout::DenseTensorViewLayout;

#define TI(x) type_index(typeid(x))

static unordered_map<type_index, string> element_type_names = {{TI(ngraph::element::Bool), "Bool"},
                                                               {TI(ngraph::element::Float32), "Float32"},
                                                               {TI(ngraph::element::Int8), "Int8"},
                                                               {TI(ngraph::element::Int32), "Int32"},
                                                               {TI(ngraph::element::Int64), "Int64"},
                                                               {TI(ngraph::element::UInt8), "UInt8"},
                                                               {TI(ngraph::element::UInt32), "UInt32"},
                                                               {TI(ngraph::element::UInt64), "UInt64"}
                                                              };


#define EIGEN_VECTOR_FORMAT(x) "{" + to_string(x) + "}"
#define EIGEN_MATRIX_FORMAT(x)

void Emitter::EmitNop(const ngraph::Node* n,
                      ExternalFunction* ef,
                      FunctionMap& function_map,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{

}

void Emitter::EmitAdd(const ngraph::Node* n,
                      ExternalFunction* ef,
                      FunctionMap& function_map,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{
    const element::Type& et = (dynamic_pointer_cast<const TensorViewType>(
                                       n->get_arguments().at(0)->get_value_type()))
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

void Emitter::EmitDot(const ngraph::Node* n,
                      ExternalFunction* ef,
                      FunctionMap& function_map,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs)
{
}

void Emitter::EmitMultiply(const ngraph::Node* n,
                           ExternalFunction* ef,
                           FunctionMap& function_map,
                           const std::vector<TensorViewInfo>& inputs,
                           const std::vector<TensorViewInfo>& outputs)
{
    const element::Type& et = (dynamic_pointer_cast<const TensorViewType>(
                                       n->get_arguments().at(0)->get_value_type()))
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
