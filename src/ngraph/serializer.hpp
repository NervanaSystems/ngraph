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
#include <unordered_map>

#include "ngraph/function.hpp"
#include "ngraph/json.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace serialize
    {
        std::string serialize(std::shared_ptr<ngraph::Function>);
        std::shared_ptr<ngraph::Function> deserialize(std::istream&);

        std::shared_ptr<ngraph::Function>
            read_function(const nlohmann::json&,
                          std::unordered_map<std::string, std::shared_ptr<Function>>&);

        nlohmann::json write(const ngraph::Function&);
        nlohmann::json write(const ngraph::Node&);
        nlohmann::json write(const ngraph::element::Type&);
    }
}
