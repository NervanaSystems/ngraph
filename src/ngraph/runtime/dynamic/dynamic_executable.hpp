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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/executable.hpp"
#include "ngraph/runtime/executable_cache.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace dynamic
        {
            class DynamicExecutable;
        }
    }
}

///
/// \brief Wrapper class used to provide an Executable that supports dynamic
///        tensors on top of a backend that does not support dynamic tensors
///        natively.
///
/// This class intercepts `call` and:
///
/// 1. creates a clone of the stored function with shapes tailored to the
///    actual runtime inputs;
/// 2. compiles the clone using the wrapped backend;
/// 3. fowards the input tensors to the clone executable for actual execution.
///
/// `DynamicExecutable` objects are produced by `DynamicBackend::compile()`.
///
class ngraph::runtime::dynamic::DynamicExecutable : public ngraph::runtime::Executable
{
public:
    DynamicExecutable(std::shared_ptr<Function> wrapped_function,
                      std::shared_ptr<ngraph::runtime::Backend> wrapped_backend,
                      bool enable_performance_collection = false);
    virtual bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                      const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;

private:
    std::shared_ptr<ngraph::Function> m_wrapped_function;
    std::shared_ptr<ngraph::runtime::Backend> m_wrapped_backend;
    std::shared_ptr<ngraph::runtime::ExecutableCache> m_cache =
        std::make_shared<ngraph::runtime::ExecutableCache>();
    bool m_enable_performance_collection;
};
