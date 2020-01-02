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
                class ReplicateElision;
            }
        }
    }
}

// Elides unnecessary Replicate ops.
//
// Replicate ops are unneeded when the input dimensions being
// replicated are all equal to one, and the downstream operation is an
// elementwise operation -- in this case, the replication can be
// skipped because PlaidML's NumPy-style broadcast semantics will
// automatically perform the replication.
//
// TODO: Technically, all Replicate operations can be elided.  But
//       doing so would require propagating tensor access information
//       as a separate concept through the operation graph, instead of
//       treating each tensor in the operation graph as a realized
//       PlaidML tensor.  That's a bigger refactor; probably worth
//       doing at some point, but not immediately.
class ngraph::runtime::plaidml::pass::ReplicateElision final : public ngraph::pass::GraphRewrite
{
public:
    ReplicateElision();
};
