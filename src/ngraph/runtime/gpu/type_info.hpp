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

#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class TypeInfo
            {
            public:
                virtual ~TypeInfo() = default;
                // Helper functions to request information about the underlying c-type
                // that is implicitly associated with the registed element::Type
                virtual std::string lowest() const = 0;
                virtual std::string min() const = 0;
                virtual std::string max() const = 0;
                virtual void* lowest_ptr() = 0;
                virtual void* min_ptr() = 0;
                virtual void* max_ptr() = 0;

                using TypeDispatch = std::unordered_map<std::string, std::shared_ptr<TypeInfo>>;
                static const std::shared_ptr<TypeInfo>& Get(const element::Type& type)
                {
                    return dispatcher.at(type.c_type_string());
                }
                static const std::shared_ptr<TypeInfo>& Get(std::string type)
                {
                    return dispatcher.at(type);
                }

            protected:
                template <typename T>
                std::string to_string(const T& val) const
                {
                    std::stringstream ss;
                    ss.precision(std::numeric_limits<T>::digits10 + 2);
                    ss << val;
                    return ss.str();
                }

            private:
                static const TypeDispatch dispatcher;
            };

            template <typename T>
            class TypeInfo_Impl : public TypeInfo
            {
            public:
                TypeInfo_Impl()
                    : m_min(std::numeric_limits<T>::min())
                    , m_max(std::numeric_limits<T>::has_infinity
                                ? std::numeric_limits<T>::infinity()
                                : std::numeric_limits<T>::max())
                    , m_lowest(std::numeric_limits<T>::has_infinity
                                   ? -std::numeric_limits<T>::infinity()
                                   : std::numeric_limits<T>::lowest())
                {
                }

                std::string lowest() const override
                {
                    return to_string<T>(std::numeric_limits<T>::lowest());
                }
                std::string min() const override
                {
                    return to_string<T>(std::numeric_limits<T>::min());
                }
                std::string max() const override
                {
                    return to_string<T>(std::numeric_limits<T>::max());
                }
                void* lowest_ptr() override { return &m_lowest; }
                void* min_ptr() override { return &m_min; }
                void* max_ptr() override { return &m_max; }
            private:
                T m_min;
                T m_max;
                T m_lowest;
            };
        }
    }
}
