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
                class ExplicitLogicals;
            }
        }
    }
}

// ExplicitLogicals handles conversion between logical and binary
// values.
//
// In PlaidML, the logical values do not have well-defined binary
// representations -- for example, due to how some frameworks handle
// vectorization, 'true' might be represented as a binary '1' or as a
// binary '-1', even within the same kernel.
//
// nGraph semantics are that 'false' == '0' and 'true' == '1', that
// booleans are exactly equivalent to binary uint8 values, and that
// binary uint8 values can be passed directly into logical operations.
//
// The ExplicitLogicals pass inserts conversions as needed to preserve
// the semantics expected by the other nGraph operations.
class ngraph::runtime::plaidml::pass::ExplicitLogicals final : public ngraph::pass::GraphRewrite
{
public:
    ExplicitLogicals()
        : GraphRewrite()
    {
        construct_logical_to_data();
    }

private:
    void construct_logical_to_data();
};
