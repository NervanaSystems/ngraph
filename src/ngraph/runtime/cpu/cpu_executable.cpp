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

#if defined(NGRAPH_TBB_ENABLE)
#include <tbb/tbb_stddef.h>
#endif

#include "cpu_backend_visibility.h"

#include "ngraph/component_manager.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_builder_registry.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_executable.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor.hpp"
#include "ngraph/runtime/cpu/static_initialize.hpp"
#include "ngraph/util.hpp"

#ifdef NGRAPH_MLIR_ENABLE
#include "contrib/mlir/backend/cpu/cpu_backend.hpp"
#include "contrib/mlir/core/compiler.hpp"
#endif

using namespace ngraph;
using namespace std;

runtime::cpu::CPU_Executable::CPU_Executable(shared_ptr<Function> func,
                                             ngraph::pass::PassConfig& pass_config,
                                             Allocator* allocator,
                                             bool performance_counters_enabled)
{
    FunctionInstance& instance = m_function_instance;
    if (instance.m_external_function == nullptr)
    {
        instance.m_external_function = make_shared<CPU_ExternalFunction>(func);
        instance.m_external_function->m_emit_timing = performance_counters_enabled;
        auto cf = instance.m_external_function->make_call_frame(pass_config, allocator);
        instance.m_call_frame = dynamic_pointer_cast<CPU_CallFrame>(cf);
    }
    set_parameters_and_results(*func);
}

std::shared_ptr<ngraph::runtime::cpu::CPU_CallFrame> runtime::cpu::CPU_Executable::get_call_frame()
{
    FunctionInstance& instance = m_function_instance;
    return instance.m_call_frame;
}

bool runtime::cpu::CPU_Executable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                        const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    bool rc = true;

    FunctionInstance& instance = m_function_instance;
    if (instance.m_external_function == nullptr)
    {
        NGRAPH_INFO;
        throw runtime_error("compile() must be called before call().");
    }

    instance.m_call_frame->call(outputs, inputs);

    return rc;
}

void runtime::cpu::CPU_Backend::remove_compiled_function(shared_ptr<Executable> exec)
{
    std::lock_guard<std::mutex> guard(m_exec_map_mutex);
    for (auto it = m_exec_map.begin(); it != m_exec_map.end(); ++it)
    {
        if (it->second == exec)
        {
            m_exec_map.erase(it);
            break;
        }
    }
}

runtime::Allocator* runtime::cpu::CPU_Backend::get_host_memory_allocator()
{
    if (!m_allocator)
    {
        return runtime::get_default_allocator();
    }
    return m_allocator;
}

void runtime::cpu::CPU_Backend::set_host_memory_allocator(Allocator* allocator)
{
    if (m_allocator)
    {
        // Resources allocated with the existing allocator might still be around and expect it
        // to be available for freeing. We cannot switch to the new allocator
        throw ngraph_error(
            "Allocator already exists. Changing allocators mid-execution is not permitted.");
    }
    m_allocator = allocator;
}

vector<runtime::PerformanceCounter> runtime::cpu::CPU_Executable::get_performance_data() const
{
    vector<runtime::PerformanceCounter> rc;
    const FunctionInstance& instance = m_function_instance;
    if (instance.m_external_function != nullptr)
    {
        rc.insert(rc.end(),
                  instance.m_external_function->get_perf_counters().begin(),
                  instance.m_external_function->get_perf_counters().end());
    }
    return rc;
}

shared_ptr<ngraph::op::Parameter> runtime::cpu::CPU_Executable::get_parameter(size_t index) const
{
    const ParameterVector& parameters = get_parameters();
    NGRAPH_CHECK(index < parameters.size(), "create_tensor for input out of bounds");
    return parameters[index];
}

shared_ptr<ngraph::op::Result> runtime::cpu::CPU_Executable::get_result(size_t index) const
{
    const ResultVector& results = get_results();
    NGRAPH_CHECK(index < results.size(), "create_tensor for input out of bounds");
    return results[index];
}

shared_ptr<runtime::Tensor> runtime::cpu::CPU_Executable::create_input_tensor(size_t input_index)
{
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    return make_shared<runtime::cpu::CPUTensor>(parameter->get_element_type(),
                                                parameter->get_shape());
}

shared_ptr<runtime::Tensor> runtime::cpu::CPU_Executable::create_input_tensor(size_t input_index,
                                                                              void* memory_pointer)
{
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    return make_shared<runtime::cpu::CPUTensor>(
        parameter->get_element_type(), parameter->get_shape(), memory_pointer);
}

shared_ptr<runtime::Tensor> runtime::cpu::CPU_Executable::create_output_tensor(size_t output_index)
{
    shared_ptr<op::Result> result = get_result(output_index);
    return make_shared<runtime::cpu::CPUTensor>(result->get_element_type(), result->get_shape());
}

shared_ptr<runtime::Tensor> runtime::cpu::CPU_Executable::create_output_tensor(size_t output_index,
                                                                               void* memory_pointer)
{
    shared_ptr<op::Result> result = get_result(output_index);
    return make_shared<runtime::cpu::CPUTensor>(
        result->get_element_type(), result->get_shape(), memory_pointer);
}

vector<shared_ptr<runtime::Tensor>>
    runtime::cpu::CPU_Executable::create_input_tensor(size_t input_index, size_t pipeline_depth)
{
    return create_input_tensor(input_index, pipeline_depth, std::vector<void*>{});
}
vector<shared_ptr<runtime::Tensor>> runtime::cpu::CPU_Executable::create_input_tensor(
    size_t input_index, size_t pipeline_depth, std::vector<void*> memory_pointers)
{
    bool mem_ptr_size = memory_pointers.size();
    if (mem_ptr_size > 0)
    {
        NGRAPH_CHECK(pipeline_depth == mem_ptr_size,
                     "create_input_tensor mismatch in pipeline_depth and memory_pointers");
    }
    vector<shared_ptr<runtime::cpu::CPUTensor>> tensors;
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        shared_ptr<runtime::cpu::CPUTensor> tensor;
        auto t =
            make_shared<runtime::cpu::CPUTensor>(parameter->get_element_type(),
                                                 parameter->get_shape(),
                                                 mem_ptr_size > 0 ? memory_pointers[i] : nullptr);
        tensor = static_pointer_cast<runtime::cpu::CPUTensor>(t);
        tensors.push_back(tensor);
    }
    vector<shared_ptr<runtime::Tensor>> result_tensors;
    for (const shared_ptr<runtime::cpu::CPUTensor>& tensor : tensors)
    {
        result_tensors.push_back(tensor);
    }
    return result_tensors;
}

vector<shared_ptr<runtime::Tensor>>
    runtime::cpu::CPU_Executable::create_output_tensor(size_t output_index, size_t pipeline_depth)
{
    return create_output_tensor(output_index, pipeline_depth, std::vector<void*>{});
}
vector<shared_ptr<runtime::Tensor>> runtime::cpu::CPU_Executable::create_output_tensor(
    size_t output_index, size_t pipeline_depth, std::vector<void*> memory_pointers)
{
    bool mem_ptr_size = memory_pointers.size();
    if (mem_ptr_size > 0)
    {
        NGRAPH_CHECK(pipeline_depth == mem_ptr_size,
                     "create_output_tensor mismatch in pipeline_depth and memory_pointers");
    }
    vector<shared_ptr<runtime::cpu::CPUTensor>> tensors;
    shared_ptr<op::Result> result = get_result(output_index);
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        shared_ptr<runtime::cpu::CPUTensor> tensor;
        auto t =
            make_shared<runtime::cpu::CPUTensor>(result->get_element_type(),
                                                 result->get_shape(),
                                                 mem_ptr_size > 0 ? memory_pointers[i] : nullptr);
        tensor = static_pointer_cast<runtime::cpu::CPUTensor>(t);
        tensors.push_back(tensor);
    }
    vector<shared_ptr<runtime::Tensor>> result_tensors;
    for (const shared_ptr<runtime::cpu::CPUTensor>& tensor : tensors)
    {
        result_tensors.push_back(tensor);
    }
    return result_tensors;
}
