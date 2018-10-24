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

#include <map>     // std::map
#include <memory>  // std::shared_ptr
#include <mutex>   // std::mutex
#include <string>  // std::string
#include <utility> // std::move
#include <vector>  // std::vector

#include <onnxifi.h>

#include "ngraph/function.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"

#include "exceptions.hpp"
#include "graph.hpp"
#include "span.hpp"
#include "tensor.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        class Graph;

        /// \brief ONNXIFI extensions to nGraph backend
        class Backend
        {
        public:
            Backend(const Backend&) = delete;
            Backend& operator=(const Backend&) = delete;

            Backend(Backend&& other, const std::lock_guard<std::mutex>&) noexcept
                : m_type{std::move(other.m_type)},
                  m_backend{std::move(other.m_backend)},
                  m_handle{other.m_handle}
            {
            }

            Backend(Backend&& other) noexcept
                : Backend{std::move(other), std::lock_guard<std::mutex>{other.m_mutex}}
            {
            }

            Backend& operator=(Backend&& other) noexcept
            {
                if (&other != this)
                {
                    std::unique_lock<std::mutex> lock{m_mutex, std::defer_lock};
                    std::unique_lock<std::mutex> other_lock{other.m_mutex, std::defer_lock};
                    std::lock(lock, other_lock);
                    m_handle = other.m_handle;
                    m_type = std::move(other.m_type);
                    m_backend = std::move(other.m_backend);
                }
                return *this;
            }

            Backend() = delete;

            explicit Backend(std::string type)
                : m_type{std::move(type)}
            {
            }

            bool compile(const std::shared_ptr<Function>& function) const
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                return get().compile(function);
            }

            bool call(const std::shared_ptr<Function>& function,
                      const std::vector<InputTensor>& inputs,
                      std::vector<OutputTensor>& outputs) const
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                auto ng_outputs = to_ng_outputs(outputs);
                auto ng_inputs = to_ng_inputs(inputs);
                bool result{get().call(function, ng_outputs, ng_inputs)};
                from_ng_outputs(ng_outputs, outputs);
                return result;
            }

            void from_ng_outputs(const std::vector<std::shared_ptr<runtime::Tensor>>& ng_outputs,
                std::vector<OutputTensor>& output) const
            {
                for (std::size_t i{0}; i < ng_outputs.size(); ++i)
                {
                    output[i].from_ng(*ng_outputs[i]);
                }
            }

            std::vector<std::shared_ptr<runtime::Tensor>>
            to_ng_outputs(const std::vector<OutputTensor>& outputs) const
            {
                std::vector<std::shared_ptr<runtime::Tensor>> result;
                for (const auto& tensor : outputs)
                {
                    result.emplace_back(tensor.to_ng(*m_backend));
                }
                return result;
            }

            std::vector<std::shared_ptr<runtime::Tensor>>
                to_ng_inputs(const std::vector<InputTensor>& inputs) const
            {
                std::vector<std::shared_ptr<runtime::Tensor>> result;
                for (const auto& tensor : inputs)
                {
                    result.emplace_back(tensor.to_ng(*m_backend));
                }
                return result;
            }

            // Implementation of onnxGetBackendInfo() interface function.
            // Refer to https://github.com/onnx/onnx/blob/master/onnx/onnxifi.h for details.
            // Each function method is responsible for obtaining value of a single
            // attribute. The function method names are reflecting the attribute names.

            void get_onnxifi_version(void* info_value, std::size_t* info_value_size) const;
            void get_name(void* info_value, std::size_t* info_value_size) const;
            void get_vendor(void* info_value, std::size_t* info_value_size) const;
            void get_version(void* info_value, std::size_t* info_value_size) const;
            void get_extensions(void* info_value, std::size_t* info_value_size) const;
            void get_device(void* info_value, std::size_t* info_value_size) const;
            void get_device_type(void* info_value, std::size_t* info_value_size) const;
            void get_onnx_ir_version(void* info_value, std::size_t* info_value_size) const;
            void get_opset_version(void* info_value, std::size_t* info_value_size) const;
            void get_capabilities(void* info_value, std::size_t* info_value_size) const;
            void get_init_properties(void* info_value, std::size_t* info_value_size) const;
            void get_memory_types(void* info_value, std::size_t* info_value_size) const;
            void get_graph_init_properties(void* info_value, std::size_t* info_value_size) const;
            void get_synchronization_types(void* info_value, std::size_t* info_value_size) const;
            void get_memory_size(void* info_value, std::size_t* info_value_size) const;
            void get_max_graph_size(void* info_value, std::size_t* info_value_size) const;
            void get_max_graph_count(void* info_value, std::size_t* info_value_size) const;

            ::onnxBackend init_handle()
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                if (m_backend == nullptr)
                {
                    m_handle = reinterpret_cast<::onnxBackend>(
                        (m_backend = runtime::Backend::create(m_type)).get());
                }
                return m_handle;
            }

            ::onnxBackend get_handle() const
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                return m_handle;
            }

            bool operator==(const Backend& other) const noexcept
            {
                std::unique_lock<std::mutex> lock{m_mutex, std::defer_lock};
                std::unique_lock<std::mutex> other_lock{other.m_mutex, std::defer_lock};
                std::lock(lock, other_lock);
                return (m_backend.get() == other.m_backend.get());
            }

            bool operator!=(const Backend& other) const noexcept { return !(other == *this); }
            bool operator==(::onnxBackend other) const noexcept
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                return (reinterpret_cast<::onnxBackend>(m_backend.get()) == other);
            }

            const std::string& get_type() const { return m_type; }
        private:
            mutable std::mutex m_mutex{};
            std::string m_type{};
            std::shared_ptr<runtime::Backend> m_backend{nullptr};
            ::onnxBackend m_handle{nullptr};

            runtime::Backend& get() const
            {
                if (m_backend == nullptr)
                {
                    throw status::backend_unavailable{};
                }
                return *m_backend;
            }
        };

    } // namespace onnxifi

} // namespace ngraph
