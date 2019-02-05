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

#include <map>   // std::map
#include <mutex> // std::mutex

#include "backend.hpp"
#include "graph.hpp"
#include "span.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        class GraphManager
        {
        public:
            GraphManager(const GraphManager&) = delete;
            GraphManager& operator=(const GraphManager&) = delete;

            GraphManager(GraphManager&&) = delete;
            GraphManager& operator=(GraphManager&&) = delete;

            static void init_graph(const Backend& backend,
                                   const Span<char>& model,
                                   const Span<::onnxTensorDescriptorV1>& weights,
                                   ::onnxGraph* graph)
            {
                instance()._init_graph(backend, model, weights, graph);
            }

            static void release_graph(::onnxGraph graph) { instance()._release_graph(graph); }
        private:
            std::mutex m_mutex{};
            std::map<::onnxGraph, std::unique_ptr<Graph>> m_graphs{};

            GraphManager() = default;

            static GraphManager& instance()
            {
                static GraphManager graph_manager{};
                return graph_manager;
            }

            void _init_graph(const Backend& backend,
                             const Span<char>& onnx_model,
                             const Span<::onnxTensorDescriptorV1>& weights,
                             ::onnxGraph* graph_handle);

            void _release_graph(::onnxGraph handle);
        };

    } // namespace onnxifi

} // namespace ngraph
