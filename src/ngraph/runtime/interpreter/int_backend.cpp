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

#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/interpreter/int_tensor_view.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<runtime::CallFrame> runtime::interpreter::INT_Backend::make_call_frame(
    const shared_ptr<ExternalFunction>& external_function)
{
    NGRAPH_INFO;
    return external_function->make_call_frame();
}

shared_ptr<runtime::TensorView>
    runtime::interpreter::INT_Backend::make_primary_tensor_view(const element::Type& element_type,
                                                                const Shape& shape)
{
    auto rc = make_shared<runtime::interpreter::INT_TensorView>(element_type, shape, "external");
    return static_pointer_cast<runtime::TensorView>(rc);
}
