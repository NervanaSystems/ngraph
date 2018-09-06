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

#include <memory>  // std::shared_ptr
#include <string>  // std::string
#include <utility> // std::move
#include <vector>  // std::vector

#include "ngraph/function.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        /// \brief ONNXIFI extensions to nGraph backend
        class Backend
        {
        public:
            Backend(const Backend&) = delete;
            Backend& operator=(const Backend&) = delete;

            Backend(Backend&&) = default;
            Backend& operator=(Backend&&) = default;

            Backend() = delete;

            explicit Backend(const std::string& type)
                : m_type{type}
            {
            }

            const std::string& get_type() const { return m_type; }
            bool compile(const std::shared_ptr<Function>& function) const
            {
                return get().compile(function);
            }

            bool call(const std::shared_ptr<Function>& function,
                      const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                      const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) const
            {
                return get().call(function, outputs, inputs);
            }

            bool call_with_validate(
                const std::shared_ptr<Function>& function,
                const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) const
            {
                return get().call_with_validate(function, outputs, inputs);
            }

        private:
            std::string m_type{};
            mutable std::shared_ptr<runtime::Backend> m_backend{nullptr};

            runtime::Backend& get() const
            {
                if (m_backend == nullptr)
                {
                    m_backend = runtime::Backend::create(m_type);
                }
                return *m_backend;
            }
        };

    } // namespace onnxifi

} // namespace ngraph
