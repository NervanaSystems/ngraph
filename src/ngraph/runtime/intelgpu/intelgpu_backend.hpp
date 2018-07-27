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
    std::shared_ptr<ngraph::runtime::TensorView>
        create_tensor(const ngraph::element::Type& element_type,
                      const Shape& shape,
                      void* memory_pointer) override;

    std::shared_ptr<ngraph::runtime::TensorView>
        create_tensor(const ngraph::element::Type& element_type, const Shape& shape) override;

    bool compile(std::shared_ptr<Function> func) override;

    bool call(std::shared_ptr<Function> func,
              const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
              const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) override;

private:
    class FunctionInstance
    {
    public:
        std::shared_ptr<cldnn::network> ocl_network = nullptr;
    };

    std::map<std::shared_ptr<Function>, FunctionInstance> ocl_networks;
    std::shared_ptr<cldnn::engine> ocl_engine;
};
