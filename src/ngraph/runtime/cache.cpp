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

// Constructor
runtime::LRUCache::LRUCache()
{
    m_size = 1024;
    m_map = {};
    m_list = {};
}

// Destructor
runtime::LRUCache::~LRUCache()
{
    m_list.clear();
    m_map.clear();
}

ostringstream runtime::LRUCache::convert_shape_to_string(const vector<int>& shape)
{
    ostringstream key;

    if (!shape.empty())
    {
        std::copy(shape.begin(), shape.end(), std::ostream_iterator<int>(key, ", "));
    }
    return key;
}

void runtime::LRUCache::add_entry(const vector<int>& shape, shared_ptr<runtime::Executable> exec)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    ostringstream key;
    // check if the list is empty
    if (m_list.size() == m_size)
    {
        ostringstream key = convert_shape_to_string(m_list.back());
        m_list.pop_back();
        m_map.erase(key.str());
    }

    key = convert_shape_to_string(shape);
    m_map.insert({key.str(), exec});
    m_list.push_front(shape);
    std::cout << "Key is " << key.str() << std::endl;
}

bool runtime::LRUCache::is_cached(const vector<int>& shape)
{
    std::cout << "List size " << m_list.size() << std::endl;
    for (auto itr = m_list.begin(); itr != m_list.end(); itr++)
    {
        if (*itr == shape)
        {
            return true;
        }
    }
    return false;
}

shared_ptr<runtime::Executable> runtime::LRUCache::get_cached_entry(const vector<int>& shape)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    // find the entry and return the function
    ostringstream key;
    key = convert_shape_to_string(shape);
    auto it = m_map.find(key.str());

    // update list to push this reference to the front
    for (auto itr = m_list.begin(); itr != m_list.end(); itr++)
    {
        if (*itr == shape)
        {
            m_list.remove(shape);
            m_list.push_front(shape);
            break;
        }
    }
    return it->second;
}
