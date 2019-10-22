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

#include "ngraph/function.hpp"
#include "ngraph/shape.hpp"
#include <map>
#include <unordered_map>
#include <list>

using namespace std;

namespace ngraph
{
    namespace runtime
    {
        class LRUCache: public std::enable_shared_from_this<LRUCache>
        {
        public:
            LRUCache(int);

            void add_entry(Shape, shared_ptr<Function>);
            bool is_cached(Shape);
            shared_ptr<Function> get_cached_entry(Shape);
  
        private:
            int m_size; // cache size
            unordered_map<Shape, shared_ptr<Function>> m_map;
            list<Shape> m_list;
        };
    }
}
