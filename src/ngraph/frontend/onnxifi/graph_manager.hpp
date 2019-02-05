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

#include <functional> // std::reference_wrapper
#include <map>        // std::map
#include <mutex>      // std::mutex
#include <thread>     // std::thread

#include <onnxifi.h>

#include "backend.hpp"
#include "event.hpp"
#include "graph.hpp"
#include "queue.hpp"
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

            static void run_graph(::onnxGraph graph,
                                  const ::onnxMemoryFenceV1* input_fence,
                                  ::onnxMemoryFenceV1* output_fence)
            {
                instance()._run_graph(graph, input_fence, output_fence);
            }

            static void release_graph(::onnxGraph graph) { instance()._release_graph(graph); }
            static void set_graph_io(::onnxGraph graph,
                                     const Span<::onnxTensorDescriptorV1>& inputs,
                                     const Span<::onnxTensorDescriptorV1>& outputs)
            {
                instance()._set_graph_io(graph, outputs, inputs);
            }

        private:
            std::mutex m_mutex{};
            std::map<::onnxGraph, std::unique_ptr<Graph>> m_graphs{};
            Queue<std::reference_wrapper<Graph>> m_task_queue;
            std::atomic_bool m_quit{false};
            EventAuto m_event{};
            std::thread m_thread{[&] {
                while (true)
                {
                    while (m_task_queue.empty())
                    {
                        m_event.wait_for(std::chrono::milliseconds{100});
                        if (m_quit)
                        {
                            break;
                        }
                    }
                    if (m_quit)
                    {
                        break;
                    }
                    Graph& graph = m_task_queue.front();
                    m_task_queue.pop();
                    if (!graph.run_graph())
                    {
                        // todo: log failure running computation on graph
                    }
                }
            }};

            GraphManager() = default;

            ~GraphManager() { _terminate(); }
            static GraphManager& instance()
            {
                static GraphManager graph_manager{};
                return graph_manager;
            }

            void _terminate()
            {
                m_quit.store(true);
                if (m_thread.joinable())
                {
                    m_thread.join();
                }
            }

            void _run_graph(::onnxGraph handle,
                            const ::onnxMemoryFenceV1* input_fence,
                            ::onnxMemoryFenceV1* output_fence);

            void _set_graph_io(::onnxGraph handle,
                               const Span<::onnxTensorDescriptorV1>& outputs,
                               const Span<::onnxTensorDescriptorV1>& inputs);

            void _init_graph(const Backend& backend,
                             const Span<char>& onnx_model,
                             const Span<::onnxTensorDescriptorV1>& weights,
                             ::onnxGraph* graph_handle);

            void _release_graph(::onnxGraph handle);
        };

    } // namespace onnxifi

} // namespace ngraph
