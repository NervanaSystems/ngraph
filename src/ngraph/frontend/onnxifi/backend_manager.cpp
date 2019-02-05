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

#include <cstdlib> // std::size_t, std::uintptr_t

#include <onnxifi.h>
#include "onnx.hpp"

#include "ngraph/runtime/backend_manager.hpp"

#include "backend.hpp"
#include "backend_manager.hpp"
#include "exceptions.hpp"
#include "ngraph/runtime/backend_manager.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        BackendManager::BackendManager()
        {
            // Create ONNXIFI backend for each registered nGraph backend.
            // Use pointer to temporary to capture the unique handle. The handles
            // must be consistent within a session.
            // In spec, backends are hot-pluggable. This means two calls to
            // onnxGetBackendIDs() may result in different number of backends.
            // For now, we don't do the re-discovery.
            auto registered_backends = runtime::BackendManager::get_registered_backends();
            for (auto& type : registered_backends)
            {
                m_registered_backends.emplace(reinterpret_cast<::onnxBackendID>(&type),
                                              Backend{type});
            }
        }

        void BackendManager::get_backend_ids(::onnxBackendID* backend_ids, std::size_t* count)
        {
            instance()._get_backend_ids(backend_ids, count);
        }

        void BackendManager::release_backend_id(::onnxBackendID backend_id)
        {
            instance()._release_backend_id(backend_id);
        }

        Backend& BackendManager::get_backend(::onnxBackend backend)
        {
            return instance()._from_handle(backend);
        }

        void BackendManager::_get_backend_ids(::onnxBackendID* backend_ids,
                                              std::size_t* count) const
        {
            if (count == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*count};
            *count = m_registered_backends.size();
            if ((requested < *count) || (backend_ids == nullptr))
            {
                throw status::fallback{};
            }
            {
                std::lock_guard<decltype(m_mutex)> lock{m_mutex};
                std::transform(std::begin(m_registered_backends),
                               std::end(m_registered_backends),
                               backend_ids,
                               [](const std::map<::onnxBackendID, Backend>::value_type& pair) {
                                   return pair.first;
                               });
            }
        }

        Backend& BackendManager::_from_id(::onnxBackendID id)
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            return m_registered_backends.at(id);
        }

        Backend& BackendManager::_from_handle(::onnxBackend handle)
        {
            if (handle != nullptr)
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                for (auto& pair : m_registered_backends)
                {
                    if (pair.second.get_handle() == handle)
                    {
                        return pair.second;
                    }
                }
            }
            throw status::invalid_backend{};
        }

        void BackendManager::_release_backend_id(::onnxBackendID id)
        {
            if (m_registered_backends.find(id) == std::end(m_registered_backends))
            {
                throw status::invalid_id{};
            }
            // nGraph ONNXIFI backend does not release backend ids
        }

    } // namespace onnxifi

} // namespace ngraph
