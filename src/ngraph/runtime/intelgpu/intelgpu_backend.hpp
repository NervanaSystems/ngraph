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

#include <map>
#include <memory>

#include <CPP/engine.hpp>
#include <CPP/network.hpp>

#include "ngraph/runtime/backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace intelgpu
        {
            class IntelGPUBackend;
        }
    }
}

class ngraph::runtime::intelgpu::IntelGPUBackend : public runtime::Backend
{
public:
    IntelGPUBackend();
    std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type,
                      const Shape& shape,
                      void* memory_pointer) override;

    std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type, const Shape& shape) override;

    runtime::Handle compile(const std::shared_ptr<Function>& func) override;

    bool call(runtime::Handle handle,
              const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
              const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;

    void remove_compiled_function(runtime::Handle handle) override;
    void enable_performance_data(bool enable) override;
    std::vector<PerformanceCounter> get_performance_data(runtime::Handle handle) const override;

private:
    class FunctionInstance
    {
    public:
        std::shared_ptr<cldnn::network> ocl_network = nullptr;
        bool m_performance_counters_enabled = false;
        std::shared_ptr<Function> m_function;
    };

    std::vector<std::shared_ptr<FunctionInstance>> m_instances;
    std::shared_ptr<cldnn::engine> ocl_engine;

    bool m_disable_backend_optimizations = false;
    bool m_performance_counters_enabled = false;

    // Statistic related things
    void print_call_performance(const std::shared_ptr<cldnn::network> network,
                                const std::shared_ptr<Function> func,
                                size_t time_compile,
                                size_t time_call,
                                double mem_before_call,
                                double mem_after_compilation,
                                double mem_after_call) const;

    bool m_profile_enable = false;
    long m_profile_lines_limit_count = 10;
    bool m_dump_graph_enable = false;
    bool m_cldnn_graph_optimize = true;
    bool m_cldnn_dump_enable = false;
    bool m_function_cache_disabled = false;
    std::string m_cldnn_dump_dir = std::string("intelgpu_codegen");
    std::string delim = std::string(":");
};
