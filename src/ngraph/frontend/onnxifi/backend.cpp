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
            if (requested < *infoValueSize)
            {
                throw std::length_error{"not enough space"};
            }
            if (infoValue != nullptr)
            {
                *reinterpret_cast<::onnxEnum*>(infoValue) = ONNXIFI_CAPABILITY_THREAD_SAFE;
            }
        }

        void Backend::get_device(void* infoValue, std::size_t* infoValueSize) const
        {
            constexpr char name[] = "nGraph ";
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof name + m_type.size();
            if (requested < *infoValueSize)
            {
                throw std::length_error{"not enough space"};
            }
            if (infoValue != nullptr)
            {
                std::memcpy(infoValue, name, sizeof name);
                std::memcpy(
                    reinterpret_cast<char*>(infoValue) + sizeof name, m_type.data(), m_type.size());
            }
        }

        void Backend::get_device_type(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(::onnxEnum);
            if (requested < *infoValueSize)
            {
                throw std::length_error{"not enough space"};
            }
            if (infoValue != nullptr)
            {
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
        }

        void Backend::get_extensions(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            *infoValueSize = 0;
            if (infoValue != nullptr)
            {
                *reinterpret_cast<char*>(infoValue) = '\0';
            }
        }

        void Backend::get_graph_init_properties(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            *infoValueSize = 0;
            if (infoValue != nullptr)
            {
                *reinterpret_cast<char*>(infoValue) = '\0';
            }
        }

        void Backend::get_onnxifi_version(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof(uint64_t);
            if (requested < *infoValueSize)
            {
                throw std::length_error{"not enough space"};
            }
            if (infoValue != nullptr)
            {
                *reinterpret_cast<uint64_t*>(infoValue) = 1;
            }
        }

        void Backend::get_name(void* infoValue, std::size_t* infoValueSize) const
        {
            constexpr char name[] = "ngraph:";
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof name + m_type.size();
            if (requested < *infoValueSize)
            {
                throw std::length_error{"not enough space"};
            }
            if (infoValue != nullptr)
            {
                std::memcpy(infoValue, name, sizeof name);
                std::memcpy(
                    reinterpret_cast<char*>(infoValue) + sizeof name, m_type.data(), m_type.size());
            }
        }

        void Backend::get_vendor(void* infoValue, std::size_t* infoValueSize) const
        {
            constexpr char vendor[] = "Intel Corporation";
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof vendor;
            if (requested < *infoValueSize)
            {
                throw std::length_error{"not enough space"};
            }
            if (infoValue != nullptr)
            {
                std::memcpy(infoValue, vendor, sizeof vendor);
            }
        }

        void Backend::get_version(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            constexpr char version[] = NGRAPH_VERSION;
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof version;
            if (requested < *infoValueSize)
            {
                throw std::length_error{"not enough space"};
            }
            if (infoValue != nullptr)
            {
                std::memcpy(infoValue, version, sizeof version);
            }
        }

        void Backend::get_onnx_ir_version(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            constexpr char version[] = ONNX_VERSION;
            std::size_t requested{*infoValueSize};
            *infoValueSize = sizeof version;
            if (requested < *infoValueSize)
            {
                throw std::length_error{"not enough space"};
            }
            if (infoValue != nullptr)
            {
                std::memcpy(infoValue, version, sizeof version);
            }
        }

        void Backend::get_opset_version(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
            std::size_t requested{*infoValueSize};
            *infoValueSize = 1;
            if (requested < *infoValueSize)
            {
                throw std::length_error{"not enough space"};
            }
            if (infoValue != nullptr)
            {
                *reinterpret_cast<char*>(infoValue) = '8';
            }
        }

        void Backend::get_init_properties(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
        }

        void Backend::get_memory_types(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
        }

        void Backend::get_synchronization_types(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
        }

        void Backend::get_memory_size(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
        }

        void Backend::get_max_graph_size(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
        }

        void Backend::get_max_graph_count(void* infoValue, std::size_t* infoValueSize) const
        {
            if (infoValueSize == nullptr)
            {
                throw std::invalid_argument{"null pointer"};
            }
        }

    } // namespace onnxifi

} // namespace ngraph
