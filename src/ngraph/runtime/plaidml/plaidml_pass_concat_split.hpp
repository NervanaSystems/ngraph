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

#include "ngraph/pass/graph_rewrite.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            namespace pass
            {
                class ConcatSplit;
            }
        }
    }
}

// A pass to split concats.
//
// PlaidML's concat operator is remarkably inefficient.  To make it
// slightly less awful, we split concats into groups.
class ngraph::runtime::plaidml::pass::ConcatSplit final : public ngraph::pass::GraphRewrite
{
public:
    ConcatSplit();
};
