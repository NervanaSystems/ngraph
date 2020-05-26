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
#include "ngraph/pass/pass_util.hpp"

namespace ngraph
{
    namespace pass
    {
        class ReshapeEliminationV1;
    }
}

class NGRAPH_API ngraph::pass::ReshapeEliminationV1 : public ngraph::pass::GraphRewrite
{
public:
    ReshapeEliminationV1()
        : GraphRewrite()
    {
        construct_identity_reshape_pattern();
    }

private:
    void construct_identity_reshape_pattern();
};
