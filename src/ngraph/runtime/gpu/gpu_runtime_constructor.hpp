//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include <string>
#include <unordered_map>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_call_frame.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPUCallFrame;
            class GPURuntimeConstructor
            {
            public:
                using op_runtime_t =
                    std::function<void(GPUCallFrame& call_frame, GPURuntimeContext* ctx)>;
                using op_order_t = std::unordered_map<std::shared_ptr<Function>,
                                                      std::vector<std::shared_ptr<Node>>>;

                GPURuntimeConstructor(const op_order_t& ordered_ops);
                void add(const std::string& name, const op_runtime_t& step);
                void add_call(const std::string& caller,
                              const std::string& callee,
                              const std::vector<runtime::gpu::GPUTensorWrapper>& args,
                              const std::vector<runtime::gpu::GPUTensorWrapper>& out);
                EntryPoint build(const std::string& function, GPUCallFrame& call_frame);

            private:
                std::unordered_map<std::string, std::vector<op_runtime_t>> m_runtime;
            };
        }
    }
}
