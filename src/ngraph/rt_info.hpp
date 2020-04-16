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

#include <string>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    NGRAPH_API
    void copy_runtime_info(std::shared_ptr<ngraph::Node> from, std::shared_ptr<ngraph::Node> to);

    NGRAPH_API
    void copy_runtime_info(std::shared_ptr<ngraph::Node> from, ngraph::NodeVector to);

    NGRAPH_API
    void copy_runtime_info(const ngraph::NodeVector& from, std::shared_ptr<ngraph::Node> to);

    NGRAPH_API
    void copy_runtime_info(const ngraph::NodeVector& from, ngraph::NodeVector to);
}
