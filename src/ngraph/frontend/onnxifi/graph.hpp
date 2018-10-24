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

#include <memory> // std::shared_ptr
#include <vector> // std::vector

#include <onnx.hpp>
#include <onnxifi.h>

#include "ngraph/function.hpp"
#include "ngraph/runtime/tensor.hpp"

#include "backend.hpp"
#include "span.hpp"
#include "tensor.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        class Backend;

        /// \brief Representation of onnxGraph
        class Graph
        {
        public:
            Graph(const Graph&) = delete;
            Graph& operator=(const Graph&) = delete;

            Graph() = delete;

            Graph(Graph&&) noexcept = default;
            Graph& operator=(Graph&&) noexcept = delete;

            explicit Graph(const Backend& backend)
                : m_backend{backend}
            {
            }

            Graph(const Backend& backend, std::istream& sin)
                : m_function{onnx_import::import_onnx_function(sin)}
                , m_backend{backend}
            {
            }

            explicit operator ::onnxGraph() const;

            void set_weights(const Span<::onnxTensorDescriptorV1>& weights);
            void set_inputs(const Span<::onnxTensorDescriptorV1>& inputs);
            void set_outputs(const Span<::onnxTensorDescriptorV1>& outputs);

            bool compile();

            void configure_memory_fences(const ::onnxMemoryFenceV1* input_fence,
                                         ::onnxMemoryFenceV1* output_fence);

            void from_stream(std::istream& sin);

            bool operator==(const Graph& other) const noexcept;
            bool operator!=(const Graph& other) const noexcept;

            bool run_graph();

        private:
            std::shared_ptr<Function> m_function{nullptr};
            std::vector<InputTensor> m_inputs{};
            std::vector<OutputTensor> m_outputs{};
            const Backend& m_backend;
            const ::onnxMemoryFenceV1* m_input_fence{nullptr};
            ::onnxMemoryFenceV1* m_output_fence{nullptr};
        };

        inline bool Graph::operator==(const Graph& other) const noexcept
        {
            return (m_function == other.m_function);
        }

        inline bool Graph::operator!=(const Graph& other) const noexcept
        {
            return !(*this == other);
        }

        inline void Graph::from_stream(std::istream& sin)
        {
            m_function = onnx_import::import_onnx_function(sin);
        }

        inline Graph::operator ::onnxGraph() const { return (::onnxGraph)(this); }
        inline void Graph::set_inputs(const Span<::onnxTensorDescriptorV1>& inputs)
        {
            if ((inputs.data() != nullptr) && inputs.empty())
            {
                throw status::invalid_size{};
            }
            if (inputs.is_valid())
            {
                for (const auto& descriptor : inputs)
                {
                    m_inputs.emplace_back(descriptor);
                }
            }
        }

        inline void Graph::set_outputs(const Span<::onnxTensorDescriptorV1>& outputs)
        {
            if (outputs.data() == nullptr)
            {
                throw status::null_pointer{};
            }
            if (outputs.empty())
            {
                throw status::invalid_size{};
            }
            for (const auto& descriptor : outputs)
            {
                m_outputs.emplace_back(descriptor);
            }
        }

    } // namespace onnxifi

} // namespace ngraph
