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

#include <memory>  // std::shared_ptr
#include <string>  // std::string
#include <utility> // std::move
#include <vector>  // std::vector

#include "ngraph/function.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/executable.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        /// \brief ONNXIFI extensions to nGraph Executable
        class Executable
        {
        public:
            Executable(const Executable&) = delete;
            Executable& operator=(const Executable&) = delete;

            Executable(Executable&&) = default;
            Executable& operator=(Executable&&) = default;

            explicit Executable(const std::shared_ptr<runtime::Executable>& executable)
                : m_executable{executable}
            {
            }

            bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                      const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) const
            {
                return m_executable->call(outputs, inputs);
            }

            bool call_with_validate(
                const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) const
            {
                return m_executable->call_with_validate(outputs, inputs);
            }

        private:
            mutable std::shared_ptr<runtime::Executable> m_executable{nullptr};
        };

    } // namespace onnxifi

} // namespace ngraph
