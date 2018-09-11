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

#include <cstddef> // std::size_t, std::uintptr_t
#include <map>     // std::map
#include <mutex>   // std::mutex

#include <onnxifi.h>

#include "ngraph/runtime/backend.hpp"

#include "backend.hpp"
#include "exceptions.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        /// \brief ONNXIFI backend manager
        class BackendManager
        {
        public:
            BackendManager(const BackendManager&) = delete;
            BackendManager& operator=(const BackendManager&) = delete;

            BackendManager(BackendManager&&) = delete;
            BackendManager& operator=(BackendManager&&) = delete;

            static void get_backend_ids(::onnxBackendID* backend_ids, std::size_t* count)
            {
                instance().get_ids(backend_ids, count);
            }

            static void release_backend_id(::onnxBackendID backend_id)
            {
                instance().release_id(backend_id);
            }

            static void get_backend_info(::onnxBackendID backend_id,
                                         ::onnxBackendInfo info_type,
                                         void* info_value,
                                         std::size_t* info_value_size);

            static void init_backend(::onnxBackendID backend_id, ::onnxBackend* backend);

            static void init_graph(::onnxBackend backend,
                                   const void* onnx_model,
                                   std::size_t onnx_model_size,
                                   const ::onnxTensorDescriptorV1* weights,
                                   std::size_t weights_count,
                                   ::onnxGraph* graph);

        private:
            mutable std::mutex m_mutex{};
            std::map<::onnxBackendID, Backend> m_registered_backends{};

            BackendManager();

            static BackendManager& instance()
            {
                static BackendManager backend_manager;
                return backend_manager;
            }

            void release_id(::onnxBackendID id)
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                m_registered_backends.erase(id);
            }

            void get_ids(::onnxBackendID* backend_ids, std::size_t* count) const;
            Backend& get_backend(::onnxBackend backend);

            Backend& get_backend_by_id(::onnxBackendID id)
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                return m_registered_backends.at(id);
            }
        };

    } // namespace onnxifi

} // namespace ngraph
