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

#include <algorithm>
#include <iostream>
#include <iterator>
#include <list>
#include <sstream>
#include <string>
#include <unordered_map>
#include "ngraph/runtime/executable.hpp"
#include "ngraph/shape.hpp"

using namespace std;

namespace ngraph
{
    namespace runtime
    {
        class LRUCache : public std::enable_shared_from_this<LRUCache>
        {
        public:
            using GraphCache = unordered_map<string, shared_ptr<Executable>>;

            LRUCache();

            virtual ~LRUCache();

            void add_entry(const vector<int>&, shared_ptr<Executable>);
            bool is_cached(const vector<int>&);
            shared_ptr<Executable> get_cached_entry(const vector<int>&);
            ostringstream convert_shape_to_string(const vector<int>& shape);

        private:
            int m_size; // cache size
            GraphCache m_map;
            list<vector<int>> m_list;
            mutex m_mutex;
        };
    }
}
