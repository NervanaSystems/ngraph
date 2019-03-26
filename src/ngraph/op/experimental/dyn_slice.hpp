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

#include "ngraph/node_vector.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Takes a slice of an input tensor, i.e., the sub-tensor that resides within a bounding box, optionally with stride.
        class DynSlice : public Op
        {
        public:
            /// \brief Constructs a dynamic tensor slice operation.
            ///
            /// \param arg The tensor to be sliced.
            /// \param lower_bounds The axiswise lower bounds of the slice (inclusive).
            /// \param upper_bounds The axiswise upper bounds of the slice (exclusive).
            /// \param strides The slicing strides; for example, strides of `{n,m}` means to take
            ///                every nth row and every mth column of the input matrix.
            DynSlice(const std::shared_ptr<Node>& arg,
                     const std::shared_ptr<Node>& lower_bounds,
                     const std::shared_ptr<Node>& upper_bounds,
                     const std::shared_ptr<Node>& strides);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
            void validate_and_infer_types() override;
        };
    }
}
