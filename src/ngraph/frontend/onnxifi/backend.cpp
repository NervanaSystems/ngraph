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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include <onnxifi.h>

#include "backend.hpp"
#include "exceptions.hpp"

#include "onnx.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        void Backend::get_capabilities(void* info_value, std::size_t* info_value_size) const
        {
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = sizeof(::onnxEnum);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            *reinterpret_cast<::onnxEnum*>(info_value) = ONNXIFI_CAPABILITY_THREAD_SAFE;
        }

        void Backend::get_device(void* info_value, std::size_t* info_value_size) const
        {
            constexpr char prefix[] = "nGraph ";
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::string device{prefix + m_type};
            std::size_t requested{*info_value_size};
            *info_value_size = device.size();
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            std::memcpy(info_value, device.data(), device.size());
        }

        void Backend::get_device_type(void* info_value, std::size_t* info_value_size) const
        {
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = sizeof(::onnxEnum);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }

            if ((m_type == "GPU") || (m_type == "INTELGPU"))
            {
                *reinterpret_cast<::onnxEnum*>(info_value) = ONNXIFI_DEVICE_TYPE_GPU;
            }
            else if (m_type == "FPGA")
            {
                *reinterpret_cast<::onnxEnum*>(info_value) = ONNXIFI_DEVICE_TYPE_FPGA;
            }
            else if (m_type == "NNP")
            {
                *reinterpret_cast<::onnxEnum*>(info_value) = ONNXIFI_DEVICE_TYPE_NPU;
            }
            else
            {
                *reinterpret_cast<::onnxEnum*>(info_value) = ONNXIFI_DEVICE_TYPE_CPU;
            }
        }

        void Backend::get_extensions(void* info_value, std::size_t* info_value_size) const
        {
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = sizeof(char);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            *reinterpret_cast<char*>(info_value) = '\0';
        }

        void Backend::get_graph_init_properties(void* info_value,
                                                std::size_t* info_value_size) const
        {
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = sizeof(char);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            *reinterpret_cast<char*>(info_value) = '\0';
        }

        void Backend::get_onnxifi_version(void* info_value, std::size_t* info_value_size) const
        {
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = sizeof(uint64_t);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            *reinterpret_cast<uint64_t*>(info_value) = 1;
        }

        void Backend::get_name(void* info_value, std::size_t* info_value_size) const
        {
            constexpr char prefix[] = "ngraph:";
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::string name{prefix + m_type};
            std::size_t requested{*info_value_size};
            *info_value_size = name.size();
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            std::memcpy(info_value, name.data(), name.size());
        }

        void Backend::get_vendor(void* info_value, std::size_t* info_value_size) const
        {
            constexpr char vendor[] = "Intel Corporation";
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = std::strlen(vendor);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            std::memcpy(info_value, vendor, *info_value_size);
        }

        void Backend::get_version(void* info_value, std::size_t* info_value_size) const
        {
            constexpr char version[] = NGRAPH_VERSION;
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = std::strlen(version);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            std::memcpy(info_value, version, *info_value_size);
        }

        void Backend::get_onnx_ir_version(void* info_value, std::size_t* info_value_size) const
        {
            constexpr char version[] = ONNX_VERSION;
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = std::strlen(version);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            std::memcpy(info_value, version, *info_value_size);
        }

        void Backend::get_opset_version(void* info_value, std::size_t* info_value_size) const
        {
            constexpr char version[] = ONNX_OPSET_VERSION;
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = std::strlen(version);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            std::memcpy(info_value, version, *info_value_size);
        }

        void Backend::get_init_properties(void* info_value, std::size_t* info_value_size) const
        {
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = sizeof(char);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            *reinterpret_cast<char*>(info_value) = '\0';
        }

        void Backend::get_memory_types(void* info_value, std::size_t* info_value_size) const
        {
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = sizeof(::onnxBitfield);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            *reinterpret_cast<::onnxBitfield*>(info_value) = ONNXIFI_MEMORY_TYPE_CPU;
        }

        void Backend::get_synchronization_types(void* info_value,
                                                std::size_t* info_value_size) const
        {
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = sizeof(::onnxBitfield);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            *reinterpret_cast<::onnxBitfield*>(info_value) = ONNXIFI_SYNCHRONIZATION_EVENT;
        }

        void Backend::get_memory_size(void* info_value, std::size_t* info_value_size) const
        {
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = sizeof(uint64_t);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            *reinterpret_cast<uint64_t*>(info_value) = std::numeric_limits<uint64_t>::max();
        }

        void Backend::get_max_graph_size(void* info_value, std::size_t* info_value_size) const
        {
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = sizeof(uint64_t);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            *reinterpret_cast<uint64_t*>(info_value) = std::numeric_limits<uint64_t>::max();
        }

        void Backend::get_max_graph_count(void* info_value, std::size_t* info_value_size) const
        {
            if (info_value_size == nullptr)
            {
                throw status::null_pointer{};
            }
            std::size_t requested{*info_value_size};
            *info_value_size = sizeof(uint64_t);
            if ((requested < *info_value_size) || (info_value == nullptr))
            {
                throw status::fallback{};
            }
            *reinterpret_cast<uint64_t*>(info_value) = std::numeric_limits<uint64_t>::max();
        }

    } // namespace onnxifi

} // namespace ngraph
