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

#include "graph_manager.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        namespace
        {
            struct Buffer : std::streambuf
            {
                Buffer(const Buffer&) = default;
                Buffer& operator=(const Buffer&) = default;

                Buffer(Buffer&&) = default;
                Buffer& operator=(Buffer&&) = default;

                Buffer() = delete;

                explicit Buffer(const Span<char>& buffer)
                    : m_begin{const_cast<char*>(buffer.begin())}
                    , m_end{const_cast<char*>(buffer.end())}
                {
                    setg(m_begin, m_begin, m_end);
                }

                Buffer(char* buffer, std::size_t size)
                    : m_begin{buffer}
                    , m_end{buffer + size}
                {
                    if (buffer == nullptr)
                    {
                        throw status::null_pointer{};
                    }
                    if (size == 0)
                    {
                        throw status::invalid_size{};
                    }
                    setg(m_begin, m_begin, m_end);
                }

                Buffer(void* buffer, std::size_t size)
                    : Buffer{reinterpret_cast<char*>(buffer), size}
                {
                }

                pos_type seekoff(off_type off,
                                 std::ios_base::seekdir dir,
                                 std::ios_base::openmode which) override
                {
                    switch (dir)
                    {
                    case std::ios_base::cur: gbump(static_cast<int>(off)); break;
                    case std::ios_base::end: setg(m_begin, m_end + off, m_end); break;
                    case std::ios_base::beg: setg(m_begin, m_begin + off, m_end); break;
                    default: break;
                    }
                    return gptr() - eback();
                }

                pos_type seekpos(std::streampos pos, std::ios_base::openmode mode) override
                {
                    return seekoff(pos - pos_type(off_type{0}), std::ios_base::beg, mode);
                }

            private:
                char *m_begin{nullptr}, *m_end{nullptr};
            };

        } // namespace <anonymous>

        void GraphManager::allocate(const Backend& backend,
                                    const Span<char>& onnx_model,
                                    const Span<::onnxTensorDescriptorV1>& weights,
                                    ::onnxGraph* graph_handle)
        {
            if (graph_handle == nullptr)
            {
                throw status::null_pointer{};
            }
            if (onnx_model.empty())
            {
                throw status::invalid_size{};
            }
            if (!onnx_model.is_valid())
            {
                throw status::null_pointer{};
            }
            Buffer buffer{onnx_model};
            std::istream sin{&buffer};
            std::unique_ptr<Graph> graph{new Graph{backend}};
            graph->load(sin, weights);
            graph->compile();
            std::lock_guard<std::mutex> lock{m_mutex};
            auto it =
                m_graphs.emplace(reinterpret_cast<::onnxGraph>(graph.get()), std::move(graph));
            if (!it.second)
            {
                throw status::no_system_resources{};
            }
            *graph_handle = (it.first)->first;
        }

        void GraphManager::set_io(::onnxGraph handle,
                                  const Span<::onnxTensorDescriptorV1>& outputs,
                                  const Span<::onnxTensorDescriptorV1>& inputs)
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            auto& graph = m_graphs.at(handle);
            graph->set_inputs(inputs);
            graph->set_outputs(outputs);
        }

    } // namespace onnxifi

} // namespace ngraph
