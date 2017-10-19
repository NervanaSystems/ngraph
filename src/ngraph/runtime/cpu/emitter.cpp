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

#include "ngraph/node.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"
#include "ngraph/runtime/cpu/external_function.hpp"
#include "ngraph/runtime/cpu/emitter.hpp"

using namespace std;
using namespace ngraph::runtime::cpu;

void Emitter::EmitNop(const ngraph::Node* n,
                      ExternalFunction* ef,
                      FunctionMap& function_map,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs) const
{

}

void Emitter::EmitAdd(const ngraph::Node* n,
                      ExternalFunction* ef,
                      FunctionMap& function_map,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs) const
{

}
         
void Emitter::EmitDot(const ngraph::Node* n,
                      ExternalFunction* ef,
                      FunctionMap& function_map,
                      const std::vector<TensorViewInfo>& inputs,
                      const std::vector<TensorViewInfo>& outputs) const
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
        cout << "Emitting scalar-tensor product\n";
    }
    else if (arg1_shape.size() == 0)
    {
        cout << "Emitting scalar-tensor product\n";   
    }

    // If arg0 and arg1 are both vectors, emit a dot product.
    else if (arg0_shape.size() == 1 && arg1_shape.size() == 1)
    {
        cout << "Emitting dot product\n";
    }

    // If arg0 is a matrix and arg1 is a vector, emit a matrix-vector product.
    else if (arg0_shape.size() == 2 && arg1_shape.size() == 1)
    {
        cout << "Emitting matrix-vector product\n";
    }

    // If arg0 and arg1 are both matrices, emit a matrix product.
    else if (arg0_shape.size() == 2 && arg1_shape.size() == 2)
    {
        cout << "Emitting matrix multiply\n";
    }

    else
    {
        throw ngraph_error("Dot product for tensors with rank>2 not implemented yet.");
    }
}

void Emitter::EmitMultiply(const ngraph::Node* n,
                           ExternalFunction* ef,
                           FunctionMap& function_map,
                           const std::vector<TensorViewInfo>& inputs,
                           const std::vector<TensorViewInfo>& outputs) const
{

}
