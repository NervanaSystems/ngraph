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
        class Reduce : public Builtin
        {
        public:
            ///
            /// @param arg_reductee The tensor view to be reduced.
            /// @param arg_init The initial value for reduction.
            /// @param reduction_function The reduction function to use.
            /// @param reduction_axes The axis positions (0-based) to be reduced.
            ///
            Reduce(const std::shared_ptr<Node>& arg_reductee,
                   const std::shared_ptr<Node>& arg_init,
                   const std::shared_ptr<Function>& reduction_function,
                   const AxisSet& reduction_axes)
                : Builtin({arg_reductee, arg_init})
                , m_reduction_function(reduction_function)
                , m_reduction_axes(reduction_axes)
            {
            }

            virtual std::string description() const override { return "Reduce"; }
            virtual void propagate_types() override;

            std::shared_ptr<Function> get_reduction_function() const
            {
                return m_reduction_function;
            }
            const AxisSet& get_reduction_axes() const { return m_reduction_axes; }
        protected:
            std::shared_ptr<Function> m_reduction_function;
            AxisSet m_reduction_axes;
        };
    }
}
