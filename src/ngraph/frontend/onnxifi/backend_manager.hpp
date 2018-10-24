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

            static void get_backend_ids(::onnxBackendID* backend_ids, std::size_t* count);
            static void release_backend_id(::onnxBackendID backend_id);
            static void release_backend(::onnxBackend backend);

            static void get_backend_info(::onnxBackendID backend_id,
                                         ::onnxBackendInfo info_type,
                                         void* info_value,
                                         std::size_t* info_value_size);

            static void init_backend(::onnxBackendID backend_id, ::onnxBackend* backend);
            static Backend& get_backend(::onnxBackend backend);

        private:
            mutable std::mutex m_mutex{};
            std::map<::onnxBackendID, Backend> m_registered_backends{};

            BackendManager();

            static BackendManager& instance()
            {
                static BackendManager backend_manager;
                return backend_manager;
            }

            void release_id(::onnxBackendID id);
            void release(::onnxBackend backend);
            void get_ids(::onnxBackendID* backend_ids, std::size_t* count) const;

            Backend& get_by_handle(::onnxBackend handle);
            Backend& get_by_id(::onnxBackendID id);
        };

    } // namespace onnxifi

} // namespace ngraph
