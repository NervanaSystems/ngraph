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

#include <memory> // std::shared_ptr
#include <vector> // std::vector

#include <onnxifi.h>

#include "ngraph/function.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        /// \brief Representation of onnxGraph
        class Graph
        {
        public:
            Graph(const Graph&) = default;
            Graph& operator=(const Graph&) = default;

            Graph() = default;

            Graph(Graph&&) noexcept = default;
            Graph& operator=(Graph&&) noexcept = default;

            void set_inputs(::onnxTensorDescriptorV1* inputs, std::size_t inputs_count);
            void set_outputs(::onnxTensorDescriptorV1* outputs, std::size_t outputs_count);

            const std::vector<std::shared_ptr<runtime::Tensor>>& get_inputs() const
            {
                return m_inputs;
            }

            const std::vector<std::shared_ptr<runtime::Tensor>>& get_outputs() const
            {
                return m_outputs;
            }

            const std::shared_ptr<Function>& get_ng_function() const
            {
                return m_function;
            }

            bool operator == (const Graph& other) const noexcept
            {
                return (m_function == other.m_function);
            }

            bool operator != (const Graph& other) const noexcept
            {
                return !(*this == other);
            }

        private:
            std::shared_ptr<Function> m_function{nullptr};
            std::vector<std::shared_ptr<runtime::Tensor>> m_inputs{}, m_outputs{};
        };

    } // namespace onnxifi

} // namespace ngraph
