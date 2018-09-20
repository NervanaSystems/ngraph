//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        class BatchDot : public Op
        {
        public:
            BatchDot(std::shared_ptr<Node> a,
                     std::shared_ptr<Node> b,
                     bool transpose_a,
                     bool transpose_b);

            bool get_is_a_transposed() const { return m_transpose_a; }
            bool get_is_b_transposed() const { return m_transpose_b; }
            Shape get_a_shape() const { return get_argument(0)->get_shape(); }
            Shape get_b_shape() const { return get_argument(1)->get_shape(); }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            bool m_transpose_a;
            bool m_transpose_b;
        };
    }
}
