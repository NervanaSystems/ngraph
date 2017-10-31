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

#include "ngraph/ops/macro.hpp"
#include "ngraph/ops/op.hpp"

namespace ngraph
{
    namespace op
    {
        class CrossEntropy : public MacroNode
        {
        public:
            CrossEntropy(const std::shared_ptr<Node>& predictions,
                         const std::shared_ptr<Node>& answers)
                : MacroNode({predictions, answers})
            {
            }
            virtual std::shared_ptr<Node> lower() override;
            virtual std::string description() const override { return "CrossEntropy"; }
        };
    }
}
