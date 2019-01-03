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
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/node.hpp"
#include "ngraph/pass/manager_state.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            const std::unordered_map<
                std::type_index,
                std::function<bool(std::shared_ptr<Node>, std::shared_ptr<Node>)>>&
                get_cse_handlers_map();
        }
    }
}
