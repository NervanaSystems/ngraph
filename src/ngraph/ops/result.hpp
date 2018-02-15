// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
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

#include "ngraph/ops/op.hpp"

namespace ngraph
{
    namespace op
    {
        class Result : public RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs an arcsin operation.
            ///
            /// \param arg Node that produces the input tensor.
            Result(const std::shared_ptr<Node>& arg)
                : RequiresTensorViewArgs("Result", {arg})
            {
                if (arg->get_outputs().size() != 1)
                {
                    throw ngraph_error("Expected a single-output argument");
                }

                set_value_type_checked(arg->get_element_type(), arg->get_shape());
            }

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 1)
                {
                    throw ngraph_error("Incorrect number of new arguments");
                }

                if (new_args.at(0)->get_outputs().size() != 1)
                {
                    throw ngraph_error("Expected a single-output argument");
                }

                return std::make_shared<Result>(new_args.at(0));
            }

            bool is_functionally_identical(const Node& other) const override { return false; }
            virtual bool is_output() const override { return true; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override
            {
                adjoints.add_delta(get_input_op(0), delta);
            }
        };
    }
}
