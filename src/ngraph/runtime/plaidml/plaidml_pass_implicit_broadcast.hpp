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

#include "ngraph/pass/graph_rewrite.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            namespace pass
            {
                class ImplicitBroadcast;
            }
        }
    }
}

// The PlaidML nGraph runtime's implementation of the Broadcast
// operation requires a contraction, and then the broadcasted output
// needs to be read by the downstream operation.
//
// Most explicit Broadcast operations are passed as inputs to
// elementwise operations.  When a tensor is used as an input to an
// elementwise operation, PlaidML automatically provides NumPy
// broadcast semantics.
//
// So eliding Broadcast operations can significantly reduce the IO
// needed by an elementwise operation, and eliminates an unnecessary
// contraction.
class ngraph::runtime::plaidml::pass::ImplicitBroadcast final : public ngraph::pass::GraphRewrite
{
public:
    ImplicitBroadcast();
};
