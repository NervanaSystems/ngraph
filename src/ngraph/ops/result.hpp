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

#include "ngraph/ops/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        class Result : public util::RequiresTensorViewArgs
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

                //always borrow the placement conf even the default one
                set_placement(arg->get_placement());
                set_value_type_checked(arg->get_element_type(), arg->get_shape());
            }

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override
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
