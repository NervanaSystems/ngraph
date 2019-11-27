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

#include "ngraph/pass/pass.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace pass
    {
        class NGRAPH_API ReshapeSinking : public ngraph::pass::FunctionPass
        {
        public:
            ReshapeSinking() { set_property(PassProperty::REQUIRE_STATIC_SHAPE, true); }
            bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
        };
    }
}

extern template ngraph::AxisVector
    ngraph::apply_permutation<ngraph::AxisVector>(ngraph::AxisVector input,
                                                  ngraph::AxisVector order);

extern template ngraph::Coordinate
    ngraph::apply_permutation<ngraph::Coordinate>(ngraph::Coordinate input,
                                                  ngraph::AxisVector order);

extern template ngraph::Strides
    ngraph::apply_permutation<ngraph::Strides>(ngraph::Strides input, ngraph::AxisVector order);

extern template ngraph::Shape ngraph::apply_permutation<ngraph::Shape>(ngraph::Shape input,
                                                                       ngraph::AxisVector order);
