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
#include "ngraph/runtime/gpu/gpu_call_frame.hpp"

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
                using op_runtime_t = std::function<void(GPUCallFrame& call_frame, GPURuntimeContext* ctx)>;
                using op_order_t = std::unordered_map<std::shared_ptr<Function>, std::list<std::shared_ptr<Node>>>;
                GPURuntimeConstructor(const op_order_t& ordered_ops)
                {
                    size_t num_ops = 0;
                    for (auto const& ops : ordered_ops)
                    {
                        num_ops += ops.second.size();
                    }
                    m_runtime.reserve(num_ops);
                }

                void add(const op_runtime_t& step)
                {
                    m_runtime.push_back(step);
                }

                EntryPoint build(GPUCallFrame& call_frame)
                {
                    return [=](void** inputs, void** outputs, GPURuntimeContext* ctx) mutable
                    {
                        call_frame.resolve_inputs(inputs);
                        call_frame.resolve_outputs(outputs);
                        for (auto const& step : m_runtime)
                        {
                            step(call_frame, ctx);
                        }
                    };
                }

            private:
                std::vector<op_runtime_t> m_runtime;
            };
        }
    }
}
