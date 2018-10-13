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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace hybrid
        {
            class HYBRIDBackend : public runtime::Backend
            {
            public:
                std::shared_ptr<Tensor> create_tensor(const element::Type& type,
                                                      const Shape& shape,
                                                      void* memory_pointer) override;

                std::shared_ptr<Tensor> create_tensor(const element::Type& type,
                                                      const Shape& shape) override;

                bool compile(std::shared_ptr<Function> function) override;

                bool call(std::shared_ptr<Function> function,
                          const std::vector<std::shared_ptr<Tensor>>& outputs,
                          const std::vector<std::shared_ptr<Tensor>>& intputs) override;

            private:
                class FunctionInstance
                {
                public:
                    std::shared_ptr<Function> m_function;
                    std::vector<std::shared_ptr<Function>> m_sub_functions;
                    std::unordered_map<std::shared_ptr<op::Parameter>, std::shared_ptr<op::Result>>
                        m_map_parameter_to_result;
                };

                std::shared_ptr<runtime::Backend> get_cached_backend(Placement placement);

                std::map<Placement, std::shared_ptr<runtime::Backend>> m_cached_backends;
                std::map<std::shared_ptr<Function>, FunctionInstance> m_function_map;
            };
        }
    }
}
