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

#include "ngraph/runtime/gpu/gpu_compiled_function.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPUCallFrame
            {
            public:
                using TensorType = GPUTensorWrapper::TensorType;

                GPUCallFrame(const size_t& num_inputs, const size_t& num_outputs);
                void resolve_reservations(
                    const GPU_CompiledFunction* compiled_function,
                    const std::unordered_map<std::string, size_t>& memory_reservations);
                void resolve_inputs(void** inputs);
                void resolve_inputs(const std::vector<void*>& inputs);
                void resolve_outputs(void** outputs);
                void resolve_outputs(const std::vector<void*>& outputs);
                std::vector<void*> get_tensor_io(const std::vector<GPUTensorWrapper>& tensors);

            private:
                void* get_pointer(const TensorType& type,
                                  const size_t& offset,
                                  const std::string& name = "");

                std::unordered_map<std::string, unsigned char*> m_memory_reservations;
                std::vector<unsigned char*> m_inputs;
                std::vector<unsigned char*> m_outputs;
            };
        }
    }
}
