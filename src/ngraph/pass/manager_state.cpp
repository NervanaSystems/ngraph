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

#include <iostream>
#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/manager_state.hpp"

using namespace std;
using namespace ngraph;

vector<Function*>& ngraph::pass::ManagerState::get_functions()
{
    return m_function_list;
}

size_t ngraph::pass::ManagerState::get_temporary_pool_size()
{
    return m_temporary_pool_size;
}

void ngraph::pass::ManagerState::set_temporary_pool_size(size_t size)
{
    m_temporary_pool_size = size;
}
