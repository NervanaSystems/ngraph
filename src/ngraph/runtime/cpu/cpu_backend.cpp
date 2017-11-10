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

#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/external_function.hpp"

using namespace ngraph;
using namespace std;

extern "C" void
    allocate_aligned_buffer(size_t size, size_t alignment, char** allocated, char** aligned_ptr);

std::shared_ptr<ngraph::runtime::CallFrame> runtime::cpu::CPUBackend::make_call_frame(
    const std::shared_ptr<ExternalFunction>& external_function)
{
    return external_function->make_call_frame();
}

std::shared_ptr<ngraph::runtime::TensorView>
    runtime::cpu::CPUBackend::make_primary_tensor_view(const ngraph::element::Type& element_type,
                                                       const Shape& shape)
{
    size_t size = ngraph::shape_size(shape);
    size_t tensor_size = size * element_type.size();
    char* allocated;
    char* alligned;
    allocate_aligned_buffer(tensor_size, runtime::cpu::alignment, &allocated, &alligned);
    m_tensor_buffer = shared_ptr<char>(new char[size], alligned);
}
