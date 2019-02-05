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

#include <memory> // std::shared_ptr
#include <vector> // std::vector

#include <onnxifi.h>

#include "ngraph/function.hpp"

#include "backend.hpp"
#include "span.hpp"
#include "tensor.hpp"
#include "weight.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        /// \brief Representation of onnxGraph
        class Graph
        {
        public:
            Graph(const Graph&) = delete;
            Graph& operator=(const Graph&) = delete;

            Graph() = delete;
            ~Graph();

            Graph(Graph&&) noexcept = default;
            Graph& operator=(Graph&&) noexcept = delete;

            explicit Graph(const Backend& backend)
                : m_backend{&backend}
            {
            }

            void load(std::istream& sin, const Span<::onnxTensorDescriptorV1>& weights);

            void set_inputs(const Span<::onnxTensorDescriptorV1>& inputs);
            void set_outputs(const Span<::onnxTensorDescriptorV1>& outputs);

            void configure_memory_fences(const ::onnxMemoryFenceV1* input_fence,
                                         ::onnxMemoryFenceV1* output_fence);

            bool run_graph();

            void from_ng_outputs(const std::vector<std::shared_ptr<runtime::Tensor>>& ng_outputs,
                                 std::vector<Tensor>& output) const
            {
                for (std::size_t i{0}; i < ng_outputs.size(); ++i)
                {
                    output[i].from_ng(*ng_outputs[i]);
                }
            }

        private:
            runtime::Handle m_handle{nullptr};
            std::vector<std::shared_ptr<runtime::Tensor>> m_ng_inputs{};
            std::vector<Tensor> m_outputs{};
            std::vector<std::shared_ptr<runtime::Tensor>> m_ng_outputs{};
            const Backend* m_backend;
            const ::onnxMemoryFenceV1* m_input_fence{nullptr};
            ::onnxMemoryFenceV1* m_output_fence{nullptr};
        };

    } // namespace onnxifi

} // namespace ngraph
