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
            auto registered_backends = runtime::BackendManager::get_registered_backends();
            for (const auto& type : registered_backends)
            {
                m_registered_backends.emplace(reinterpret_cast<std::uintptr_t>(&type),
                                              Backend{type});
            }
        }

        void BackendManager::get_ids(::onnxBackendID* backendIDs, std::size_t* count) const
        {
            if (count == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*count};
            *count = m_registered_backends.size();
            if (requested < *count)
            {
                throw std::length_error{"not enough space"};
            }
            if (backendIDs != nullptr)
            {
                std::lock_guard<decltype(m_mutex)> lock{m_mutex};
                std::transform(std::begin(m_registered_backends),
                               std::end(m_registered_backends),
                               backendIDs,
                               [](const std::map<std::uintptr_t, Backend>::value_type& pair)
                                   -> ::onnxBackendID {
                                   return reinterpret_cast<::onnxBackendID>(pair.first);
                               });
            }
        }

        void BackendManager::get_backend_info(::onnxBackendID backendID,
                                              onnxBackendInfo infoType,
                                              void* infoValue,
                                              std::size_t* infoValueSize)
        {
            const auto& backend = instance().get_backend(backendID);
            switch (infoType)
            {
            case ONNXIFI_BACKEND_ONNXIFI_VERSION:
                backend.get_onnxifi_version(infoValue, infoValueSize);
                break;
            case ONNXIFI_BACKEND_NAME: backend.get_name(infoValue, infoValueSize); break;
            case ONNXIFI_BACKEND_VENDOR: backend.get_vendor(infoValue, infoValueSize); break;
            case ONNXIFI_BACKEND_VERSION: backend.get_version(infoValue, infoValueSize); break;
            case ONNXIFI_BACKEND_EXTENSIONS:
                backend.get_extensions(infoValue, infoValueSize);
                break;
            case ONNXIFI_BACKEND_DEVICE: backend.get_device(infoValue, infoValueSize); break;
            case ONNXIFI_BACKEND_DEVICE_TYPE:
                backend.get_device_type(infoValue, infoValueSize);
                break;
            case ONNXIFI_BACKEND_ONNX_IR_VERSION:
                backend.get_onnx_ir_version(infoValue, infoValueSize);
                break;
            case ONNXIFI_BACKEND_OPSET_VERSION:
                backend.get_opset_version(infoValue, infoValueSize);
                break;
            case ONNXIFI_BACKEND_CAPABILITIES:
                backend.get_capabilities(infoValue, infoValueSize);
                break;
            case ONNXIFI_BACKEND_INIT_PROPERTIES:
                backend.get_init_properties(infoValue, infoValueSize);
                break;
            case ONNXIFI_BACKEND_MEMORY_TYPES:
                backend.get_memory_types(infoValue, infoValueSize);
                break;
            case ONNXIFI_BACKEND_GRAPH_INIT_PROPERTIES:
                backend.get_graph_init_properties(infoValue, infoValueSize);
                break;
            case ONNXIFI_BACKEND_SYNCHRONIZATION_TYPES:
                backend.get_synchronization_types(infoValue, infoValueSize);
                break;
            case ONNXIFI_BACKEND_MEMORY_SIZE:
                backend.get_memory_size(infoValue, infoValueSize);
                break;
            case ONNXIFI_BACKEND_MAX_GRAPH_SIZE:
                backend.get_max_graph_size(infoValue, infoValueSize);
                break;
            case ONNXIFI_BACKEND_MAX_GRAPH_COUNT:
                backend.get_max_graph_count(infoValue, infoValueSize);
                break;
            default: throw std::range_error{"invalid info type"};
            }
        }

    } // namespace onnxifi

} // namespace ngraph
