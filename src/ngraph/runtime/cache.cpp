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

#include "ngraph/runtime/cache.hpp"

using namespace ngraph;
using namespace std;

runtime::LRUCache::LRUCache(int size)
{
    m_size = size;
}

void runtime::LRUCache::add_entry(Shape shape, shared_ptr<Function> funct)
{
    // check if the list is empty
    if(m_list.size() == m_size)
    {
        m_map.erase(m_list.pop_back()); 
    }
    
    m_map.insert({shape, funct});
    m_list.push_front(shape);
}

bool runtime::LRUCache::is_cached(Shape shape)
{
    for (auto itr = m_list.begin(); itr != m_list.end(); itr++)
    {
        if (*itr == shape)
        {
            return true;
        }
    }
    return false;
}

shared_ptr<Function> runtime::LRUCache::get_cached_entry(Shape shape)
{
    // find the entry and return the function
    auto it = m_map.find(shape);
    if (it != m_map.end())
    {
        // update list to push this reference to the front
        for (auto itr = m_list.begin(); itr != m_list.end(); itr++)
        {
            if (*itr == shape)
            {
                m_list.remove(*itr);
                m_list.push_front(*itr);
            }
            return it->second;
        }
    }
}
