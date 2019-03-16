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

#include <memory>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        class Result : public Op
        {
        public:
            /// \brief Allows a value to be used as a function result.
            ///
            /// \param arg Output that produces the input tensor.
            Result(const NodeOutput& arg);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_source_outputs(const OutputVector& new_source_outputs) const override;

            virtual bool is_output() const override { return true; }
            void set_needs_default_layout(bool val) { m_needs_default_layout = val; }
            bool needs_default_layout() const { return m_needs_default_layout; }
        protected:
            virtual void build_backprop(autodiff::Adjoints& adjoints,
                                        const OutputVector& deltas) override;

        private:
            bool m_needs_default_layout{false};
        };
    }
}
