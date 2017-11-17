// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <vector>

#include "ngraph/function.hpp"

namespace ngraph
{
    namespace pass
    {
        class ManagerState;
    }
}

class ngraph::pass::ManagerState
{
public:
    const std::vector<std::shared_ptr<Function>>& get_functions();

    template <typename T>
    void set_functions(const T& collection)
    {
        m_function_list.clear();
        m_function_list.insert(m_function_list.begin(), collection.begin(), collection.end());
    }

    size_t get_temporary_pool_size();
    void set_temporary_pool_size(size_t);

private:
    size_t m_temporary_pool_size = 0;
    std::vector<std::shared_ptr<Function>> m_function_list;
};
