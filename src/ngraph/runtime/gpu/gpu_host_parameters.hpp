//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <cinttypes>
#include <list>

#include "ngraph/except.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPUHostParameters
            {
            public:
                GPUHostParameters() = default;

                void* cache(const char& value)
                {
                    m_char_params.push_back(value);
                    return &m_char_params.back();
                }
                void* cache(const float& value)
                {
                    m_float_params.push_back(value);
                    return &m_float_params.back();
                }
                void* cache(const double& value)
                {
                    m_double_params.push_back(value);
                    return &m_double_params.back();
                }
                void* cache(const int8_t& value)
                {
                    m_int8_t_params.push_back(value);
                    return &m_int8_t_params.back();
                }
                void* cache(const int16_t& value)
                {
                    m_int16_t_params.push_back(value);
                    return &m_int16_t_params.back();
                }
                void* cache(const int32_t& value)
                {
                    m_int32_t_params.push_back(value);
                    return &m_int32_t_params.back();
                }
                void* cache(const int64_t& value)
                {
                    m_int64_t_params.push_back(value);
                    return &m_int64_t_params.back();
                }
                void* cache(const uint8_t& value)
                {
                    m_uint8_t_params.push_back(value);
                    return &m_uint8_t_params.back();
                }
                void* cache(const uint16_t& value)
                {
                    m_uint16_t_params.push_back(value);
                    return &m_uint16_t_params.back();
                }
                void* cache(const uint32_t& value)
                {
                    m_uint32_t_params.push_back(value);
                    return &m_uint32_t_params.back();
                }
                void* cache(const uint64_t& value)
                {
                    m_uint64_t_params.push_back(value);
                    return &m_uint64_t_params.back();
                }

                template <typename T1, typename T2>
                void* getVal(T2 val)
                {
                    return cache(static_cast<T1>(val));
                }

                void* val_by_datatype(const std::string& type, double val)
                {
                    if (type == "char")
                    {
                        return getVal<char>(val);
                    }
                    else if (type == "float")
                    {
                        return getVal<float>(val);
                    }
                    else if (type == "double")
                    {
                        return getVal<double>(val);
                    }
                    else if (type == "int8_t")
                    {
                        return getVal<int8_t>(val);
                    }
                    else if (type == "int16_t")
                    {
                        return getVal<int16_t>(val);
                    }
                    else if (type == "int32_t")
                    {
                        return getVal<int32_t>(val);
                    }
                    else if (type == "int64_t")
                    {
                        return getVal<int64_t>(val);
                    }
                    else if (type == "uint8_t")
                    {
                        return getVal<uint8_t>(val);
                    }
                    else if (type == "uint16_t")
                    {
                        return getVal<uint16_t>(val);
                    }
                    else if (type == "uint32_t")
                    {
                        return getVal<uint32_t>(val);
                    }
                    else if (type == "uint64_t")
                    {
                        return getVal<uint64_t>(val);
                    }
                    throw ngraph_error("Cast requested for invalid dtype");
                }

                void* val_by_datatype(const std::string& type, int64_t val)
                {
                    if (type == "char")
                    {
                        return getVal<char>(val);
                    }
                    else if (type == "float")
                    {
                        return getVal<float>(val);
                    }
                    else if (type == "double")
                    {
                        return getVal<double>(val);
                    }
                    else if (type == "int8_t")
                    {
                        return getVal<int8_t>(val);
                    }
                    else if (type == "int16_t")
                    {
                        return getVal<int16_t>(val);
                    }
                    else if (type == "int32_t")
                    {
                        return getVal<int32_t>(val);
                    }
                    else if (type == "int64_t")
                    {
                        return getVal<int64_t>(val);
                    }
                    else if (type == "uint8_t")
                    {
                        return getVal<uint8_t>(val);
                    }
                    else if (type == "uint16_t")
                    {
                        return getVal<uint16_t>(val);
                    }
                    else if (type == "uint32_t")
                    {
                        return getVal<uint32_t>(val);
                    }
                    else if (type == "uint64_t")
                    {
                        return getVal<uint64_t>(val);
                    }
                    throw ngraph_error("Cast requested for invalid dtype");
                }

            private:
                std::list<char> m_char_params;
                std::list<float> m_float_params;
                std::list<double> m_double_params;
                std::list<int8_t> m_int8_t_params;
                std::list<int16_t> m_int16_t_params;
                std::list<int32_t> m_int32_t_params;
                std::list<int64_t> m_int64_t_params;
                std::list<uint8_t> m_uint8_t_params;
                std::list<uint16_t> m_uint16_t_params;
                std::list<uint32_t> m_uint32_t_params;
                std::list<uint64_t> m_uint64_t_params;
            };
        }
    }
}
