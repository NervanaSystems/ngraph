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

#include "ngraph/runtime/nvgpu/nvgpu_runtime_constructor.hpp"

using namespace ngraph;

runtime::nvgpu::NVRuntimeConstructor::NVRuntimeConstructor(const op_order_t& ordered_ops)
{
    for (auto const& ops : ordered_ops)
    {
        m_runtime[ops.first->get_name()].reserve(ops.second.size());
    }
}

void runtime::nvgpu::NVRuntimeConstructor::add(const std::string& name, const op_runtime_t& step)
{
    m_runtime[name].push_back(step);
}

void runtime::nvgpu::NVRuntimeConstructor::add_call(
    const std::string& caller,
    const std::string& callee,
    const std::vector<runtime::nvgpu::NVTensorWrapper>& args,
    const std::vector<runtime::nvgpu::NVTensorWrapper>& out)
{
    auto& runtime = m_runtime[callee];
    auto call = [args, out, &runtime](NVCallFrame& caller_frame, NVRuntimeContext* ctx) mutable {
        // extract memory pointers from the callers stack
        auto inputs = caller_frame.get_tensor_io(args);
        auto outputs = caller_frame.get_tensor_io(out);
        // create a new call frame for the nested function
        NVCallFrame callee_frame = caller_frame;
        // resolve the inputs of the new call frame
        callee_frame.resolve_inputs(inputs.data(), inputs.size());
        callee_frame.resolve_outputs(outputs.data(), outputs.size());
        for (auto const& step : runtime)
        {
            step(callee_frame, ctx);
        }
    };
    add(caller, call);
}

runtime::nvgpu::EntryPoint runtime::nvgpu::NVRuntimeConstructor::build(const std::string& function,
                                                                       NVCallFrame& call_frame)
{
    auto& runtime = m_runtime.at(function);
    return [call_frame, &runtime](void** inputs, void** outputs, NVRuntimeContext* ctx) mutable {
        call_frame.resolve_inputs(inputs);
        call_frame.resolve_outputs(outputs);
        for (auto const& step : runtime)
        {
            step(call_frame, ctx);
        }
    };
}
