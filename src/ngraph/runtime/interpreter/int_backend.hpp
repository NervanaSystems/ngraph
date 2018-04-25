/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <map>
#include <memory>

#include "ngraph/runtime/backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class ExternalFunction;
            class INT_CallFrame;

            class INT_Backend : public runtime::Backend
            {
            public:
                std::shared_ptr<INT_CallFrame> make_call_frame(
                    const std::shared_ptr<ngraph::runtime::interpreter::ExternalFunction>&
                        external_function);

                std::shared_ptr<ngraph::runtime::TensorView>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape,
                                  void* memory_pointer) override;

                std::shared_ptr<ngraph::runtime::TensorView>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape) override;

                bool compile(std::shared_ptr<Function> func) override;

                bool call(std::shared_ptr<Function> func,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) override;

                void set_nan_check(std::shared_ptr<Function> func, bool);

                void enable_performance_data(std::shared_ptr<Function> func, bool enable) override;
                std::vector<PerformanceCounter>
                    get_performance_data(std::shared_ptr<Function> func) const override;

            private:
                class FunctionInstance
                {
                public:
                    std::shared_ptr<interpreter::ExternalFunction> m_external_function;
                    std::shared_ptr<interpreter::INT_CallFrame> m_call_frame;
                    bool m_nan_check_enabled = false;
                    bool m_performance_counters_enabled = false;
                };

                std::map<std::shared_ptr<Function>, FunctionInstance> m_function_map;
                static bool init;
            };
        }
    }
}
