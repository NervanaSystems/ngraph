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

#include <functional>
#include <initializer_list>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"

using visualize_tree_ops_init_list_t = std::initializer_list<
    std::pair<std::type_index, std::function<void(const ngraph::Node&, std::ostream&)>>>;

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

    void set_visualize_tree_ops_list(const visualize_tree_ops_init_list_t& init_list)
    {
        m_visualize_tree_ops_list = init_list;
    }

    const visualize_tree_ops_init_list_t& get_visualize_tree_ops_list()
    {
        return m_visualize_tree_ops_list;
    }

private:
    std::vector<std::shared_ptr<Function>> m_function_list;
    visualize_tree_ops_init_list_t m_visualize_tree_ops_list;
};
