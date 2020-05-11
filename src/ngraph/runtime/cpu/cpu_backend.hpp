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

#include <map>
#include <memory>
#include <mutex>

#include "cpu_backend_visibility.h"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/runtime/allocator.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/cpu/cpu_executable.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_ExternalFunction;
            class CPU_CallFrame;
            BackendConstructor CPU_BACKEND_API get_backend_constructor_pointer();
            class CPU_BACKEND_API CPU_Backend : public runtime::Backend
            {
            public:
                ~CPU_Backend() override;

                std::shared_ptr<CPU_CallFrame>
                    make_call_frame(const std::shared_ptr<CPU_ExternalFunction>& external_function,
                                    ngraph::pass::PassConfig& pass_config,
                                    Allocator* allocator);

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape,
                                  void* memory_pointer) override;

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape) override;

                std::shared_ptr<ngraph::runtime::Executable>
                    compile(std::shared_ptr<Function> func,
                            bool enable_performance_counters = false) override;

                std::shared_ptr<ngraph::runtime::Executable>
                    compile(std::shared_ptr<Function> func,
                            ngraph::pass::PassConfig& pass_config,
                            bool enable_performance_counters = false) override;

                void remove_compiled_function(std::shared_ptr<Executable> exec) override;

                Allocator* get_host_memory_allocator() override;
                void set_host_memory_allocator(Allocator* allocator) override;

                bool is_supported(const Node& node) const override;
                bool is_supported_property(const Property prop) const override;

            private:
                // this mutex will be used to protect the addition and deletion
                // of function to m_exec_map across multiple threads
                std::mutex m_exec_map_mutex;
                std::unordered_map<std::shared_ptr<Function>, std::shared_ptr<Executable>>
                    m_exec_map;
                Allocator* m_allocator;
            };
        }
    }
}
