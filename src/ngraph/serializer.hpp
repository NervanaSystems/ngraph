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

#include <memory>
#include <unordered_map>

#include "ngraph/function.hpp"
#include "ngraph/json.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    std::string serialize(std::shared_ptr<ngraph::Function>, size_t indent = 0);
    std::shared_ptr<ngraph::Function> deserialize(std::istream&);
    std::shared_ptr<ngraph::Function> deserialize(const std::string&);

    template <typename T>
    T get_or_default(nlohmann::json& j, const std::string& key, const T& default_value)
    {
        T rc;
        try
        {
            rc = j.at(key).get<T>();
        }
        catch (...)
        {
            rc = default_value;
        }
        return rc;
    }
}
