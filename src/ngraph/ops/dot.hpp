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

namespace ngraph
{
    namespace op
    {
        class Dot : public Builtin
        {
        public:
            /// Computes the dot product of two tensors.
            ///
            /// There are three possible cases:
            ///  (1) arg0 or arg1 is 0-dimensional. Then, we treat the 0-dimensional
            ///      argument(s) as scalars and compute a scalar-tensor or
            ///      scalar-scalar product.
            ///         (Example: arg0 has shape {1,2,3} and arg1 has shape {}; then
            ///         the result will have shape {1,2,3}.)
            ///
            ///  (2) arg1 is 1-dimensional. Then, we compute a dot product reducing
            ///      on the innermost (rightmost) dimensions of arg0 and arg1.
            ///         (Example: arg0 has shape {1,2,3} and arg1 has shape {3}; then
            ///         the result will have shape {1,2}.)
            ///
            ///  (3) arg1 is more than 1-dimensional. Then, we compute a dot product
            ///      reducing on the innermost (rightmost) dimension of arg0, and the
            ///      next-to-innermost dimension of arg1.
            ///         (Example: arg0 has shape {3,4} and arg1 has shape {4,3}; then
            ///         the result will have shape {3,3}.)
            Dot(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Dot"; }
            virtual void        propagate_types() override;
        };
    }
}
