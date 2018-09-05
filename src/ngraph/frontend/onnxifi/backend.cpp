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

namespace ngraph
{
    namespace onnxifi
    {
        void Backend::get_capabilities(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(::onnxEnum);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            *reinterpret_cast<::onnxEnum*>(infoValue) = ONNXIFI_CAPABILITY_THREAD_SAFE;
        }

        void Backend::get_device(void* infoValue, std::size_t* infoValueSize) const
        {
            const constexpr char prefix[] = "nGraph_";
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::string device{prefix + m_type};
            std::size_t requested{*infoValueSize};
            *infoValueSize = device.size();
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            std::memcpy(infoValue, device.data(), device.size());
        }

        void Backend::get_device_type(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(::onnxEnum);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            if ((m_type == "CPU") || (m_type == "INTERPRETER"))
            {
                *reinterpret_cast<::onnxEnum*>(infoValue) = ONNXIFI_DEVICE_TYPE_CPU;
            }
            else if ((m_type == "GPU") || (m_type == "INTELGPU"))
            {
                *reinterpret_cast<::onnxEnum*>(infoValue) = ONNXIFI_DEVICE_TYPE_GPU;
            }
            else if (m_type == "FPGA")
            {
                *reinterpret_cast<::onnxEnum*>(infoValue) = ONNXIFI_DEVICE_TYPE_FPGA;
            }
            else if (m_type == "NNP")
            {
                *reinterpret_cast<::onnxEnum*>(infoValue) = ONNXIFI_DEVICE_TYPE_NPU;
            }
            else
            {
                *reinterpret_cast<::onnxEnum*>(infoValue) = ONNXIFI_DEVICE_TYPE_HETEROGENEOUS;
            }
        }

        void Backend::get_extensions(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(char);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            *reinterpret_cast<char*>(infoValue) = '\0';
        }

        void Backend::get_graph_init_properties(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(char);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            *reinterpret_cast<char*>(infoValue) = '\0';
        }

        void Backend::get_onnxifi_version(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(uint64_t);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            *reinterpret_cast<uint64_t*>(infoValue) = 1;
        }

        void Backend::get_name(void* infoValue, std::size_t* infoValueSize) const
        {
            const constexpr char prefix[] = "ngraph:";
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::string name{prefix + m_type};
            std::size_t requested{*infoValueSize};
            *infoValueSize = name.size();
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            std::memcpy(infoValue, name.data(), name.size());
        }

        void Backend::get_vendor(void* infoValue, std::size_t* infoValueSize) const
        {
            const constexpr char vendor[] = "Intel Corporation";
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = std::strlen(vendor);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            std::memcpy(infoValue, vendor, *infoValueSize);
        }

        void Backend::get_version(void* infoValue, std::size_t* infoValueSize) const
        {
            const constexpr char version[] = NGRAPH_VERSION;
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = std::strlen(version);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            std::memcpy(infoValue, version, *infoValueSize);
        }

        void Backend::get_onnx_ir_version(void* infoValue, std::size_t* infoValueSize) const
        {
            const constexpr char version[] = ONNX_VERSION;
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = std::strlen(version);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            std::memcpy(infoValue, version, *infoValueSize);
        }

        void Backend::get_opset_version(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = std::strlen(ONNX_OPSET_VERSION);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            std::memcpy(infoValue, ONNX_OPSET_VERSION, *infoValueSize);
        }

        void Backend::get_init_properties(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(char);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            *reinterpret_cast<char*>(infoValue) = '\0';
        }

        void Backend::get_memory_types(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(::onnxBitfield);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            *reinterpret_cast<::onnxBitfield*>(infoValue) = ONNXIFI_MEMORY_TYPE_CPU;
        }

        void Backend::get_synchronization_types(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(::onnxBitfield);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            *reinterpret_cast<::onnxBitfield*>(infoValue) = ONNXIFI_SYNCHRONIZATION_EVENT;
        }

        void Backend::get_memory_size(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(uint64_t);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            *reinterpret_cast<uint64_t*>(infoValue) = std::numeric_limits<uint64_t>::max();
        }

        void Backend::get_max_graph_size(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(uint64_t);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            *reinterpret_cast<uint64_t*>(infoValue) = std::numeric_limits<uint64_t>::max();
        }

        void Backend::get_max_graph_count(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(uint64_t);
            if ((requested < *infoValueSize) || (infoValue == nullptr))
            {
                throw std::length_error{"not enough space"};
            }
            *reinterpret_cast<uint64_t*>(infoValue) = std::numeric_limits<uint64_t>::max();
        }

    } // namespace onnxifi

} // namespace ngraph
