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

#include "ngraph/env_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_builder_registry.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
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

extern "C" CPU_BACKEND_API void ngraph_register_cpu_backend()
{
    runtime::BackendManager::register_backend("CPU", [](const std::string& /* config */) {
        static bool is_initialized = false;
        if (!is_initialized)
        {
            // Some pass patterns need to be fixed
            set_remove_goe(false);
#if defined(NGRAPH_TBB_ENABLE)
            // Force TBB to link to the backend
            tbb::TBB_runtime_interface_version();
#endif
            ngraph::runtime::cpu::register_builders();
            is_initialized = true;
        }
        return make_shared<runtime::cpu::CPU_Backend>();
    });
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
    return make_shared<runtime::cpu::CPUTensor>(element_type, shape);
}

shared_ptr<runtime::Tensor> runtime::cpu::CPU_Backend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::cpu::CPUTensor>(element_type, shape, memory_pointer);
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
    if (getenv_bool("NGRAPH_MLIR"))
    {
        // Initialize MLIR compiler
        ngmlir::MLIRCompiler::init();
        // Initialize MLIR backend
        ngmlir::MLIRCPUBackend::init();
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

bool runtime::cpu::CPU_Backend::is_supported(const Node& /* op */) const
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
