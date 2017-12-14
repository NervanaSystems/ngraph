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

#include "ngraph/runtime/ngvm/ngvm_backend.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/parameterized_tensor_view.hpp"

using namespace ngraph::runtime::ngvm;

std::shared_ptr<ngraph::runtime::CallFrame>
    NGVMBackend::make_call_frame(const std::shared_ptr<ExternalFunction>& external_function)
{
    return external_function->make_call_frame();
}

std::shared_ptr<ngraph::runtime::TensorView>
    NGVMBackend::make_primary_tensor_view(const ngraph::element::Type& element_type,
                                          const Shape& shape)
{
    std::shared_ptr<TensorView> rc;
    if (element_type == element::Bool::element_type())
    {
        rc = std::make_shared<ParameterizedTensorView<element::Bool>>(shape);
    }
    else if (element_type == element::Float32::element_type())
    {
        rc = std::make_shared<ParameterizedTensorView<element::Float32>>(shape);
    }
    else if (element_type == element::Float64::element_type())
    {
        rc = std::make_shared<ParameterizedTensorView<element::Float64>>(shape);
    }
    else if (element_type == element::Int8::element_type())
    {
        rc = std::make_shared<ParameterizedTensorView<element::Int8>>(shape);
    }
    else if (element_type == element::Int16::element_type())
    {
        rc = std::make_shared<ParameterizedTensorView<element::Int16>>(shape);
    }
    else if (element_type == element::Int32::element_type())
    {
        rc = std::make_shared<ParameterizedTensorView<element::Int32>>(shape);
    }
    else if (element_type == element::Int64::element_type())
    {
        rc = std::make_shared<ParameterizedTensorView<element::Int64>>(shape);
    }
    else if (element_type == element::UInt8::element_type())
    {
        rc = std::make_shared<ParameterizedTensorView<element::UInt8>>(shape);
    }
    else if (element_type == element::UInt16::element_type())
    {
        rc = std::make_shared<ParameterizedTensorView<element::UInt16>>(shape);
    }
    else if (element_type == element::UInt32::element_type())
    {
        rc = std::make_shared<ParameterizedTensorView<element::UInt32>>(shape);
    }
    else if (element_type == element::UInt64::element_type())
    {
        rc = std::make_shared<ParameterizedTensorView<element::UInt64>>(shape);
    }
    else
    {
        throw std::invalid_argument("Unknown element type in make_primary_tensor_view");
    }
    return rc;
}
