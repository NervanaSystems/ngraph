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

namespace ngraph
{
    namespace runtime
    {
        namespace dynamic
        {
            class DynamicBackend;
        }
    }
}

///
/// \brief Wrapper class used to provide dynamic tensor support on backends
///        that otherwise do not support dynamic tensors.
///
/// The main function of this class is to intercept `create_dynamic_tensor`
/// and `compile`:
///
/// * `create_dynamic_tensor` will return a special `DynamicTensor` object
///   whose shape can be updated after creation. Internally, `DynamicTensor`
///   wraps static tensors managed by the wrapped backend.
/// * `compile` will return a special `DynamicExecutable` object, which allows
///   dynamic shapes to be supported via graph cloning.
///
/// This class is instantiated by `ngraph::runtime::Backend::create`.
///
class ngraph::runtime::dynamic::DynamicBackend : public Backend
{
public:
    DynamicBackend(std::shared_ptr<ngraph::runtime::Backend> wrapped_backend);

    std::shared_ptr<Tensor> create_tensor() override;

    std::shared_ptr<Tensor>
        create_tensor(element::Type type, const Shape& shape, void* memory_pointer) override;

    std::shared_ptr<Tensor> create_tensor(element::Type type, const Shape& shape) override;

    std::shared_ptr<Tensor> create_dynamic_tensor(element::Type type,
                                                  const PartialShape& shape) override;

    bool supports_dynamic_tensors() override { return true; }
    std::shared_ptr<Executable> compile(std::shared_ptr<Function> function,
                                        bool enable_performance_data = false) override;

private:
    std::shared_ptr<ngraph::runtime::Backend> m_wrapped_backend;
};
