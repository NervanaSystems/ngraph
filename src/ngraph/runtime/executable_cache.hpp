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

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "ngraph/function.hpp"
#include "ngraph/runtime/executable.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        class ExecutableCache;
    }
}

class ngraph::runtime::ExecutableCache
{
public:
    using GraphCache = std::unordered_map<std::string, std::shared_ptr<Executable>>;
    using ClonedFunctionMap = std::unordered_map<std::string, std::shared_ptr<Function>>;

    ExecutableCache();

    virtual ~ExecutableCache();

    void add_entry(const std::vector<int>& shape,
                   std::shared_ptr<Executable> exec,
                   std::shared_ptr<Function> func);
    bool is_cached(const std::vector<int>& shape);
    std::shared_ptr<Executable> get_cached_entry(const std::vector<int>& shape);
    void convert_shape_to_string(const std::vector<int>& shape, std::ostringstream& key);
    std::shared_ptr<Function> get_cloned_function(const std::vector<int>& shape);

private:
    size_t m_cache_size;
    GraphCache m_map;
    ClonedFunctionMap m_clone_function_map;
    std::list<std::vector<int>> m_list;
    std::mutex m_mutex;
};
