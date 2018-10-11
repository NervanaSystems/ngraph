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
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace hybrid
        {
            class HYBRIDBackend;

            using backend_map_t = std::map<size_t, std::string>;
        }
    }
}

class ngraph::runtime::hybrid::HYBRIDBackend : public Backend
{
public:
    HYBRIDBackend(const backend_map_t&);
    std::shared_ptr<TensorView>
        create_tensor(const element::Type& type, const Shape& shape, void* memory_pointer) override;

    std::shared_ptr<TensorView> create_tensor(const element::Type& type,
                                              const Shape& shape) override;

    bool compile(std::shared_ptr<Function> function) override;

    bool call(std::shared_ptr<Function> function,
              const std::vector<std::shared_ptr<TensorView>>& outputs,
              const std::vector<std::shared_ptr<TensorView>>& intputs) override;

    void enable_performance_data(std::shared_ptr<Function> func, bool enable) override;
    std::vector<PerformanceCounter>
        get_performance_data(std::shared_ptr<Function> func) const override;

private:
    class FunctionInstance
    {
    public:
        bool m_is_compiled = false;
        bool m_nan_check_enabled = false;
        bool m_performance_counters_enabled = false;
        std::unordered_map<const Node*, stopwatch> m_timer_map;
        std::shared_ptr<Function> m_function;
    };
    std::map<std::shared_ptr<Function>, FunctionInstance> m_function_map;
    backend_map_t m_backend_map;
};
