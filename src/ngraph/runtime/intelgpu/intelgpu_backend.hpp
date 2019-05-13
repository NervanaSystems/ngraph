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

#pragma once

#include <map>
#include <memory>

#include <CPP/engine.hpp>

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

    std::shared_ptr<runtime::Executable> compile(std::shared_ptr<Function> func,
                                                 bool enable_timing = false) override;
    void remove_compiled_function(std::shared_ptr<runtime::Executable> exec) override;

    bool is_supported_property(const Property prop) const override;

    bool is_supported(const Node& node) const override;

    static bool is_supported_impl(const Node& node);

private:
    std::shared_ptr<cldnn::engine> cldnn_engine;
    std::map<std::shared_ptr<Function>, std::shared_ptr<runtime::Executable>> cldnn_networks;

    bool m_profile_enable = false;
    long m_profile_lines_limit_count = 10;
    bool m_dump_graph_enable = false;
    bool m_cldnn_graph_optimize = true;
    bool m_cldnn_dump_enable = false;
    bool m_function_cache_disabled = false;
    long m_disable_backend_optimizations = 0;
    std::string m_cldnn_dump_dir = std::string("intelgpu_codegen");
};
