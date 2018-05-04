/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <map>
#include <string>

#include "ngraph/attribute.hpp"

namespace ngraph
{
    class AttributeMap
    {
    public:
        AttributeMap(std::initializer_list<std::pair<const std::string, const Attribute&>> init)
        {
            for (auto kv : init)
            {
                m_map[kv.first] = kv.second.clone();
            }
        }

        AttributeMap(const AttributeMap& other)
        {
            for (auto kv : other.m_map)
            {
                m_map[kv.first] = kv.second->clone();
            }
        }

        ~AttributeMap()
        {
            for (auto kv : m_map)
            {
                delete kv.second;
            }
        }

        std::map<std::string, Attribute*>::iterator begin() { return m_map.begin(); }
        std::map<std::string, Attribute*>::const_iterator cbegin() { return m_map.cbegin(); }
        std::map<std::string, Attribute*>::iterator end() { return m_map.end(); }
        std::map<std::string, Attribute*>::const_iterator cend() { return m_map.cend(); }
        std::reverse_iterator<std::map<std::string, Attribute*>::iterator> rbegin()
        {
            return m_map.rbegin();
        }
        std::reverse_iterator<std::map<std::string, Attribute*>::const_iterator> crbegin()
        {
            return m_map.crbegin();
        }
        std::reverse_iterator<std::map<std::string, Attribute*>::iterator> rend()
        {
            return m_map.rend();
        }
        std::reverse_iterator<std::map<std::string, Attribute*>::const_iterator> crend()
        {
            return m_map.crend();
        }

    private:
        std::map<std::string, Attribute*> m_map;
    };
}
