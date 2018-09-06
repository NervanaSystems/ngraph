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

#include <cstdlib>   // std::size_t, std::uintptr_t
#include <stdexcept> // std::invalid_agrument, std::out_of_rage

#include <onnxifi.h>

#include "ngraph/runtime/backend_manager.hpp"

#include "backend.hpp"
#include "backend_manager.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        BackendManager::BackendManager()
        {
            // Create ONNXIFI backend for each registered nGraph backend.
            // Use pointer to temporary to capture the unique handle. The handles
            // must be consistent within the single session.
            // In spec, backends are hot-pluggable. This means two calls to
            // onnxGetBackendIDs() may result in different number of backends.
            // For now, we don't do the re-discovery.
            auto registered_backends = runtime::BackendManager::get_registered_backends();
            for (const auto& type : registered_backends)
            {
                m_registered_backends.emplace(reinterpret_cast<std::uintptr_t>(&type),
                                              Backend{type});
            }
        }

        void BackendManager::get_registered_ids(::onnxBackendID* backend_ids,
                                                std::size_t* count) const
        {
            if (count == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*count};
            *count = m_registered_backends.size();
            if ((requested < *count) || (backend_ids == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            {
                std::lock_guard<decltype(m_mutex)> lock{m_mutex};
                std::transform(std::begin(m_registered_backends),
                               std::end(m_registered_backends),
                               backend_ids,
                               [](const std::map<std::uintptr_t, Backend>::value_type& pair)
                                   -> ::onnxBackendID {
                                   return reinterpret_cast<::onnxBackendID>(pair.first);
                               });
            }
        }
    }
}
