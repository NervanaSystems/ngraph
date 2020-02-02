//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/runtime/gpu/gpu_call_frame.hpp"

using namespace ngraph;

runtime::gpu::GPUCallFrame::GPUCallFrame(const size_t& num_inputs, const size_t& num_outputs)
    : m_inputs(num_inputs, nullptr)
    , m_outputs(num_outputs, nullptr)
{
}

void runtime::gpu::GPUCallFrame::resolve_reservations(
    const GPUCompiledFunction* compiled_function,
    const std::unordered_map<std::string, size_t>& memory_reservations)
{
    auto& mem_primitives = compiled_function->get_primitive_emitter()->get_memory_primitives();
    for (auto const& p : memory_reservations)
    {
        // mem_primitives may return pointers for constant or workspace reservations
        m_memory_reservations[p.first] = static_cast<unsigned char*>(mem_primitives.at(p.second)());
    }
}

void runtime::gpu::GPUCallFrame::resolve_inputs(void** inputs, size_t num_inputs)
{
    // num_inputs is > 0 iff we are resolving inputs from a nested function call
    if (num_inputs == 0)
    {
        num_inputs = m_inputs.size();
    }
    for (size_t i = 0; i < num_inputs; i++)
    {
        void* input = inputs[i];
        m_inputs[i] = static_cast<unsigned char*>(input);
    }
}

void runtime::gpu::GPUCallFrame::resolve_outputs(void** outputs, size_t num_outputs)
{
    // num_outputs is > 0 iff we are resolving outputs from a nested function call
    if (num_outputs == 0)
    {
        num_outputs = m_outputs.size();
    }
    for (size_t i = 0; i < num_outputs; i++)
    {
        void* output = outputs[i];
        m_outputs[i] = static_cast<unsigned char*>(output);
    }
}

// returns pointers of any TensorRole
std::vector<void*>
    runtime::gpu::GPUCallFrame::get_tensor_io(const std::vector<GPUTensorWrapper>& tensors)
{
    std::vector<void*> ptrs;
    for (auto const& tensor : tensors)
    {
        auto offset = tensor.get_offset();
        auto ptr = get_pointer(offset.first, offset.second, tensor.get_name());
        ptrs.push_back(ptr);
    }
    return ptrs;
}

void* runtime::gpu::GPUCallFrame::get_pointer(const TensorRole& type,
                                              const size_t& offset,
                                              const std::string& name)
{
    switch (type)
    {
    case TensorRole::CONSTANT:
    case TensorRole::INTERMEDIATE:
        return static_cast<void*>(m_memory_reservations.at(name) + offset);
    case TensorRole::INPUT: return static_cast<void*>(m_inputs.at(offset));
    case TensorRole::OUTPUT: return static_cast<void*>(m_outputs.at(offset));
    case TensorRole::UNKNOWN:
    default: throw ngraph_error("GPUCallFrame encountered unknown or uninitialized tensor type");
    };
}
