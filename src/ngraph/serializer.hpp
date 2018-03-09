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

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    std::string serialize(std::shared_ptr<ngraph::Function>,
                          size_t indent = 0,
                          bool binary_constant_data = false);
    void serialize(const std::string& path, std::shared_ptr<ngraph::Function>, size_t indent = 0);
    void serialize(std::ostream& out, std::shared_ptr<ngraph::Function>, size_t indent = 0);

    std::shared_ptr<ngraph::Function> deserialize(std::istream&);
    std::shared_ptr<ngraph::Function> deserialize(const std::string&);
}
