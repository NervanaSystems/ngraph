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

#pragma once

#include <functional>
#include <map>
#include <mutex>
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    NGRAPH_API std::mutex& get_registry_mutex();

    template <typename T>
    class FactoryRegistry
    {
    public:
        using Factory = std::function<T*()>;
        using FactoryMap = std::map<decltype(T::type_info), Factory>;

        template <typename U>
        void register_factory()
        {
            std::lock_guard<std::mutex> guard(get_registry_mutex());
            m_factory_map[U::type_info] = []() { return new U(); };
        }

        bool has_factory(const decltype(T::type_info) & info)
        {
            std::lock_guard<std::mutex> guard(get_registry_mutex());
            return m_factory_map.find(info) != m_factory_map.end();
        }

        T* create(const decltype(T::type_info) & info)
        {
            std::lock_guard<std::mutex> guard(get_registry_mutex());
            auto it = m_factory_map.find(info);
            return it == m_factory_map.end() ? nullptr : it->second();
        }
        static FactoryRegistry<T>& get();

    protected:
        // Need a Compare on type_info
        FactoryMap m_factory_map;
    };
}
