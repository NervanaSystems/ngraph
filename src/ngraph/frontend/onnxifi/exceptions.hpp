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

#include <onnxifi.h>

namespace ngraph
{
    namespace onnxifi
    {
        namespace status
        {
            struct runtime
            {
                explicit constexpr runtime(::onnxStatus status)
                    : m_status{status}
                {
                }

                constexpr ::onnxStatus get_status() const { return m_status; }
            private:
                ::onnxStatus m_status;
            };

            struct internal : runtime
            {
                constexpr internal()
                    : runtime{ONNXIFI_STATUS_INTERNAL_ERROR}
                {
                }
            };

            struct fallback : runtime
            {
                constexpr fallback()
                    : runtime{ONNXIFI_STATUS_FALLBACK}
                {
                }
            };

            struct invalid_id : runtime
            {
                constexpr invalid_id()
                    : runtime{ONNXIFI_STATUS_INVALID_ID}
                {
                }
            };

            struct invalid_size : runtime
            {
                constexpr invalid_size()
                    : runtime{ONNXIFI_STATUS_INVALID_SIZE}
                {
                }
            };

            struct null_pointer : runtime
            {
                constexpr null_pointer()
                    : runtime{ONNXIFI_STATUS_INVALID_POINTER}
                {
                }
            };

            struct invalid_protobuf : runtime
            {
                constexpr invalid_protobuf()
                    : runtime{ONNXIFI_STATUS_INVALID_PROTOBUF}
                {
                }
            };

            struct invalid_model : runtime
            {
                constexpr invalid_model()
                    : runtime{ONNXIFI_STATUS_INVALID_MODEL}
                {
                }
            };

            struct invalid_backend : runtime
            {
                constexpr invalid_backend()
                    : runtime{ONNXIFI_STATUS_INVALID_BACKEND}
                {
                }
            };

            struct invalid_graph : runtime
            {
                constexpr invalid_graph()
                    : runtime{ONNXIFI_STATUS_INVALID_GRAPH}
                {
                }
            };

            struct invalid_event : runtime
            {
                constexpr invalid_event()
                    : runtime{ONNXIFI_STATUS_INVALID_EVENT}
                {
                }
            };

            struct invalid_state : runtime
            {
                constexpr invalid_state()
                    : runtime{ONNXIFI_STATUS_INVALID_STATE}
                {
                }
            };

            struct invalid_name : runtime
            {
                constexpr invalid_name()
                    : runtime{ONNXIFI_STATUS_INVALID_NAME}
                {
                }
            };

            struct invalid_shape : runtime
            {
                constexpr invalid_shape()
                    : runtime{ONNXIFI_STATUS_INVALID_SHAPE}
                {
                }
            };

            struct invalid_datatype : runtime
            {
                constexpr invalid_datatype()
                    : runtime{ONNXIFI_STATUS_INVALID_DATATYPE}
                {
                }
            };

            struct invalid_memory_type : runtime
            {
                constexpr invalid_memory_type()
                    : runtime{ONNXIFI_STATUS_INVALID_MEMORY_TYPE}
                {
                }
            };

            struct invalid_memory_location : runtime
            {
                constexpr invalid_memory_location()
                    : runtime{ONNXIFI_STATUS_INVALID_MEMORY_LOCATION}
                {
                }
            };

            struct invalid_fence_type : runtime
            {
                constexpr invalid_fence_type()
                    : runtime{ONNXIFI_STATUS_INVALID_FENCE_TYPE}
                {
                }
            };

            struct invalid_property : runtime
            {
                constexpr invalid_property()
                    : runtime{ONNXIFI_STATUS_INVALID_PROPERTY}
                {
                }
            };

            struct unsupported_tag : runtime
            {
                constexpr unsupported_tag()
                    : runtime{ONNXIFI_STATUS_UNSUPPORTED_TAG}
                {
                }
            };

            struct unsupported_version : runtime
            {
                constexpr unsupported_version()
                    : runtime{ONNXIFI_STATUS_UNSUPPORTED_VERSION}
                {
                }
            };

            struct unsupported_operator : runtime
            {
                constexpr unsupported_operator()
                    : runtime{ONNXIFI_STATUS_UNSUPPORTED_OPERATOR}
                {
                }
            };

            struct unsupported_attribute : runtime
            {
                constexpr unsupported_attribute()
                    : runtime{ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE}
                {
                }
            };

            struct unsupported_shape : runtime
            {
                constexpr unsupported_shape()
                    : runtime{ONNXIFI_STATUS_UNSUPPORTED_SHAPE}
                {
                }
            };

            struct unsupported_datatype : runtime
            {
                constexpr unsupported_datatype()
                    : runtime{ONNXIFI_STATUS_UNSUPPORTED_DATATYPE}
                {
                }
            };

            struct unsupported_memory_type : runtime
            {
                constexpr unsupported_memory_type()
                    : runtime{ONNXIFI_STATUS_UNSUPPORTED_MEMORY_TYPE}
                {
                }
            };

            struct unsupported_fence_type : runtime
            {
                constexpr unsupported_fence_type()
                    : runtime{ONNXIFI_STATUS_UNSUPPORTED_FENCE_TYPE}
                {
                }
            };

            struct unsupported_property : runtime
            {
                constexpr unsupported_property()
                    : runtime{ONNXIFI_STATUS_UNSUPPORTED_PROPERTY}
                {
                }
            };

            struct unidentified_name : runtime
            {
                constexpr unidentified_name()
                    : runtime{ONNXIFI_STATUS_UNIDENTIFIED_NAME}
                {
                }
            };

            struct mismatching_shape : runtime
            {
                constexpr mismatching_shape()
                    : runtime{ONNXIFI_STATUS_MISMATCHING_SHAPE}
                {
                }
            };

            struct mismatching_datatype : runtime
            {
                constexpr mismatching_datatype()
                    : runtime{ONNXIFI_STATUS_MISMATCHING_DATATYPE}
                {
                }
            };

            struct no_system_memory : runtime
            {
                constexpr no_system_memory()
                    : runtime{ONNXIFI_STATUS_NO_SYSTEM_MEMORY}
                {
                }
            };

            struct no_device_memory : runtime
            {
                constexpr no_device_memory()
                    : runtime{ONNXIFI_STATUS_NO_DEVICE_MEMORY}
                {
                }
            };

            struct no_system_resources : runtime
            {
                constexpr no_system_resources()
                    : runtime{ONNXIFI_STATUS_NO_SYSTEM_RESOURCES}
                {
                }
            };

            struct no_device_resources : runtime
            {
                constexpr no_device_resources()
                    : runtime{ONNXIFI_STATUS_NO_DEVICE_RESOURCES}
                {
                }
            };

            struct backend_unavailable : runtime
            {
                constexpr backend_unavailable()
                    : runtime{ONNXIFI_STATUS_BACKEND_UNAVAILABLE}
                {
                }
            };

        } // namespace error

    } // namespace onnxifi

} // namespace ngraph
