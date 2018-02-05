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

#include "ngraph/ops/op.hpp"

namespace ngraph
{
    namespace op
    {
        class Relu : public UnaryElementwiseArithmetic
        {
        public:
            Relu(const std::shared_ptr<Node>& arg0)
                : UnaryElementwiseArithmetic("Relu", arg0)
            {
            }

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 1)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<Relu>(new_args.at(0));
            }

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override;
        };
    }
}
