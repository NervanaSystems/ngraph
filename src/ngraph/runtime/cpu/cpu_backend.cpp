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

#include <tbb/tbb_stddef.h>

#include "cpu_backend_visibility.h"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/static_initialize.hpp"
#include "ngraph/util.hpp"

#ifdef NGRAPH_MLIR_ENABLE
#include "contrib/mlir/compiler.hpp"
#endif

using namespace ngraph;
using namespace std;

runtime::BackendConstructor* runtime::cpu::get_backend_constructor_pointer()
{
    class CPU_BackendConstructor : public runtime::BackendConstructor
    {
    public:
        std::shared_ptr<runtime::Backend> create(const std::string& config) override
        {
            // Force TBB to link to the backend
            tbb::TBB_runtime_interface_version();
            return make_shared<runtime::cpu::CPU_Backend>();
        }
    };

    static unique_ptr<runtime::BackendConstructor> s_backend_constructor(
        new CPU_BackendConstructor());
    return s_backend_constructor.get();
}

#if !defined(NGRAPH_CPU_STATIC_LIB_ENABLE)
extern "C" CPU_BACKEND_API runtime::BackendConstructor* get_backend_constructor_pointer()
{
    return runtime::cpu::get_backend_constructor_pointer();
}
#endif

void runtime::cpu::static_initialize()
{
    static bool s_is_initialized = false;
    if (!s_is_initialized)
    {
        s_is_initialized = true;
        BackendManager::register_backend("CPU", runtime::cpu::get_backend_constructor_pointer());
    }
}

namespace
{
    static class CPUStaticInit
    {
    public:
        CPUStaticInit()
        {
            runtime::BackendManager::register_backend(
                "CPU", runtime::cpu::get_backend_constructor_pointer());
        }
        ~CPUStaticInit() {}
    } s_cpu_static_init;
}

runtime::cpu::CPU_Backend::~CPU_Backend()
{
    m_exec_map.clear();
}

shared_ptr<runtime::cpu::CPU_CallFrame> runtime::cpu::CPU_Backend::make_call_frame(
    const shared_ptr<runtime::cpu::CPU_ExternalFunction>& external_function,
    ngraph::pass::PassConfig& pass_config,
    Allocator* allocator)
{
    return external_function->make_call_frame(pass_config, allocator);
}

shared_ptr<runtime::Tensor>
    runtime::cpu::CPU_Backend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    return make_shared<runtime::cpu::CPUTensorView>(element_type, shape);
}

shared_ptr<runtime::Tensor> runtime::cpu::CPU_Backend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::cpu::CPUTensorView>(element_type, shape, memory_pointer);
}

shared_ptr<runtime::Executable>
    runtime::cpu::CPU_Backend::compile(shared_ptr<Function> func, bool performance_counters_enabled)
{
    ngraph::pass::PassConfig pass_config;
    return compile(func, pass_config, performance_counters_enabled);
}

shared_ptr<runtime::Executable>
    runtime::cpu::CPU_Backend::compile(shared_ptr<Function> func,
                                       ngraph::pass::PassConfig& pass_config,
                                       bool performance_counters_enabled)
{
#ifdef NGRAPH_MLIR_ENABLE
    if (std::getenv("NGRAPH_MLIR") != nullptr)
    {
        // Initialize MLIR compiler
        ngmlir::MLIRCompiler::init_mlir();
    }
#endif

    shared_ptr<runtime::Executable> rc;
    // we will protect the access to map (m_exec_map) across multiple threads by creating a
    // lock_gaurd
    // m_exec_map_mutex will be released once the object `guard` goes out of scope
    {
        std::lock_guard<std::mutex> guard(m_exec_map_mutex);
        auto it = m_exec_map.find(func);
        if (it != m_exec_map.end())
        {
            rc = it->second;
            return rc;
        }
    }
    rc = make_shared<CPU_Executable>(
        func, pass_config, get_host_memory_allocator(), performance_counters_enabled);
    {
        std::lock_guard<std::mutex> guard(m_exec_map_mutex);
        m_exec_map.insert({func, rc});
        return rc;
    }
}

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

bool runtime::cpu::CPU_Backend::is_supported(const Node& op) const
{
    return true;
}

bool runtime::cpu::CPU_Backend::is_supported_property(const Property prop) const
{
    if (prop == Property::memory_attach)
    {
        return true;
    }

    return false;
}
