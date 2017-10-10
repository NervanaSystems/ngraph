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
        class Slice : public Builtin
        {
        public:
            ///
            /// @param arg The tensor view to be sliced.
            /// @param lower_bounds The axiswise lower bounds of the slice.
            /// @param upper_bounds The axiswise upper bounds of the slice (exclusive).
            /// @param step The slicing step; for example, step of {n,m} means to take
            ///             every nth row and everyth mth column of the input matrix.
            ///
            Slice(const std::shared_ptr<Node>& arg,
                  const Coordinate& lower_bounds,
                  const Coordinate& upper_bounds,
                  const Shape& step)
                : Builtin({arg})
                , m_lower_bounds(lower_bounds)
                , m_upper_bounds(upper_bounds)
                , m_step(step)
            {
            }

            Slice(const std::shared_ptr<Node>& arg,
                  const Coordinate& lower_bounds,
                  const Coordinate& upper_bounds)
                : Builtin({arg})
                , m_lower_bounds(lower_bounds)
                , m_upper_bounds(upper_bounds)
                , m_step(Shape(lower_bounds.size(), 1))
            {
            }

            virtual std::string description() const override { return "Slice"; }
            virtual void propagate_types() override;

            const Coordinate& get_lower_bounds() const { return m_lower_bounds; }
            const Coordinate& get_upper_bounds() const { return m_upper_bounds; }
            const Shape& get_step() const { return m_step; }
        protected:
            const Coordinate m_lower_bounds;
            const Coordinate m_upper_bounds;
            const Shape m_step;
        };
    }
}
