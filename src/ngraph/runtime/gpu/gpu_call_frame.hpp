//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include "ngraph/function.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_compiled_function.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_wrapper.hpp"
#include "ngraph/runtime/gpu/gpu_invoke.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPUCallFrame;
            {
            public:
                using TensorType = GPUTensorWrapper::TensorType;
                GPUCallFrame() = default;
                void resolve_reservations(const GPU_CompiledFunction* compiled_function, const std::unordered_map<std::string, size_t>& memory_reservations)
                {
                    auto& mem_primitives = compiled_function->get_primitive_emitter()->get_memory_primitives();
                    for (auto const& p : memory_reservations)
                    {
                        // mem_primitives may return pointers for constant or workspace reservations
                        m_memory_reservations[p.first] = static_cast<unsigned char*>(mem_primitives.at(p.second)());
                    }
                }
                void resolve_inputs(void** inputs, size_t n)
                {
                    for (size_t i = 0; i < n; i++)
                    {
                        m_inputs.push_back(inputs[i]);
                    }
                }
                void resolve_outputs(void** outputs, size_t n)
                {
                    for (size_t i = 0; i < n; i++)
                    {
                        m_outputs.push_back(outputs[i]);
                    }
                }

                // returns pointers of any GPUTensorWrapper::TensorType
                void** get_tensor_io(const std::vector<GPUTensorWrapper>& tensors)
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

            private:
                void* get_pointer(const TensorType& type, const size_t& offset, const std::string& name = "")
                {
                    switch(type):
                    {
                    case TensorType::CONSTANT:
                    case TensorType::INTERMEDIATE:
                        return static_cast<void*>(m_memory_reservations.at(name) + offset);
                    case TensorType::INPUT:
                        return static_cast<void*>(m_inputs.at(offset));
                    case TensorType::OUTPUT:
                        return static_cast<void*>(m_outputs.at(offset));
                    case TensorType::UNKNOWN:
                    default:
                        throw ngraph_error("GPUCallFrame encountered unknown or uninitialized tensor type");
                    }

                }
                std::unordered_map<std::string, unsigned char*> m_memory_reservations;
                std::vector<unsigned char*> m_inputs;
                std::vector<unsigned char*> m_outputs;
            };
        }
    }
}
