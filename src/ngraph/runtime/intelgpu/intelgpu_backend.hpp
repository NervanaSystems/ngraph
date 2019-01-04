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

    std::unique_ptr<runtime::Executable>
        compile(std::shared_ptr<Function> func,
                bool enable_performance_collection = false) override;

private:
    std::shared_ptr<cldnn::engine> ocl_engine;
};
