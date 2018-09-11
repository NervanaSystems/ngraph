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

#include <cstdlib> // std::size_t, std::uintptr_t

#include "onnx.hpp"
#include <onnxifi.h>

#include "ngraph/runtime/backend_manager.hpp"

#include "backend.hpp"
#include "backend_manager.hpp"
#include "exceptions.hpp"

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
            for (auto& type : registered_backends)
            {
                m_registered_backends.emplace(reinterpret_cast<::onnxBackendID>(&type),
                                              Backend{type});
            }
        }

        void BackendManager::get_ids(::onnxBackendID* backend_ids, std::size_t* count) const
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

        Backend& BackendManager::get_backend(::onnxBackend backend)
        {
            for (auto& pair : m_registered_backends)
            {
                if (pair.second.get_handle() == backend)
                {
                    return pair.second;
                }
            }
            throw status::invalid_backend{};
        }

        void BackendManager::get_backend_info(::onnxBackendID backend_id,
                                              ::onnxBackendInfo info_type,
                                              void* info_value,
                                              std::size_t* info_value_size)
        {
            const auto& backend = instance().get_backend_by_id(backend_id);
            switch (info_type)
            {
            case ONNXIFI_BACKEND_ONNXIFI_VERSION:
                backend.get_onnxifi_version(info_value, info_value_size);
                break;
            case ONNXIFI_BACKEND_NAME: backend.get_name(info_value, info_value_size); break;
            case ONNXIFI_BACKEND_VENDOR: backend.get_vendor(info_value, info_value_size); break;
            case ONNXIFI_BACKEND_VERSION: backend.get_version(info_value, info_value_size); break;
            case ONNXIFI_BACKEND_EXTENSIONS:
                backend.get_extensions(info_value, info_value_size);
                break;
            case ONNXIFI_BACKEND_DEVICE: backend.get_device(info_value, info_value_size); break;
            case ONNXIFI_BACKEND_DEVICE_TYPE:
                backend.get_device_type(info_value, info_value_size);
                break;
            case ONNXIFI_BACKEND_ONNX_IR_VERSION:
                backend.get_onnx_ir_version(info_value, info_value_size);
                break;
            case ONNXIFI_BACKEND_OPSET_VERSION:
                backend.get_opset_version(info_value, info_value_size);
                break;
            case ONNXIFI_BACKEND_CAPABILITIES:
                backend.get_capabilities(info_value, info_value_size);
                break;
            case ONNXIFI_BACKEND_INIT_PROPERTIES:
                backend.get_init_properties(info_value, info_value_size);
                break;
            case ONNXIFI_BACKEND_MEMORY_TYPES:
                backend.get_memory_types(info_value, info_value_size);
                break;
            case ONNXIFI_BACKEND_GRAPH_INIT_PROPERTIES:
                backend.get_graph_init_properties(info_value, info_value_size);
                break;
            case ONNXIFI_BACKEND_SYNCHRONIZATION_TYPES:
                backend.get_synchronization_types(info_value, info_value_size);
                break;
            case ONNXIFI_BACKEND_MEMORY_SIZE:
                backend.get_memory_size(info_value, info_value_size);
                break;
            case ONNXIFI_BACKEND_MAX_GRAPH_SIZE:
                backend.get_max_graph_size(info_value, info_value_size);
                break;
            case ONNXIFI_BACKEND_MAX_GRAPH_COUNT:
                backend.get_max_graph_count(info_value, info_value_size);
                break;
            default: throw status::unsupported_attribute{};
            }
        }

        void BackendManager::init_backend(::onnxBackendID backend_id, ::onnxBackend* backend)
        {
            if (backend == nullptr)
            {
                throw status::null_pointer{};
            }
            *backend = instance().get_backend_by_id(backend_id).init_handle();
        }
        
        void BackendManager::init_graph(::onnxBackend  backend,
                                        const void* onnx_model,
                                        std::size_t onnx_model_size,
                                        const ::onnxTensorDescriptorV1* weights,
                                        std::size_t weights_count,
                                        ::onnxGraph* graph)
        {
            if (graph == nullptr)
            {
                throw status::null_pointer{};
            }
            *graph = nullptr;
            if (weights_count != 0)
            {
                if (weights == nullptr)
                {
                    throw status::null_pointer{};
                }
                else
                {
                    // only weights embedded in a model are supported for the moment
                    throw status::internal{};
                }
            }
            *graph = instance().get_backend(backend).init_graph(onnx_model, onnx_model_size);
        }

    } // namespace onnxifi

} // namespace ngraph
