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
#include <string>
#include <vector>

#include "ngraph/runtime/backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace hybrid
        {
            class HybridBackend;
        }
    }
}

class ngraph::runtime::hybrid::HybridBackend : public ngraph::runtime::Backend
{
public:
    HybridBackend(const std::vector<std::shared_ptr<runtime::Backend>>& backend_list);

    std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type,
                      const ngraph::Shape& shape) override;

    std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type,
                      const ngraph::Shape& shape,
                      void* memory_pointer) override;

    std::shared_ptr<Executable> compile(std::shared_ptr<ngraph::Function> func,
                                        bool enable_performance_data = false) override;

    bool is_supported(const ngraph::Node& node) const override;

    void set_debug_enabled(bool flag) { m_debug_enabled = flag; }
protected:
    std::vector<std::shared_ptr<runtime::Backend>> m_backend_list;
    bool m_debug_enabled = false;
};
