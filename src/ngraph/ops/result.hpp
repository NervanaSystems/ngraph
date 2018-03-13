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
            Result(const std::shared_ptr<Node>& arg);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            virtual bool is_output() const override { return true; }
            void set_needs_copy(bool val) { m_needs_copy = val; }
            bool needs_copy() const { return m_needs_copy; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override
            {
                adjoints.add_delta(get_input_op(0), delta);
            }

        private:
            bool m_needs_copy{true};
        };
    }
}
