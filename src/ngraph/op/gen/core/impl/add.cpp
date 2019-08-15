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

#include "ngraph/op/gen/core/add.hpp"

void ngraph::op::gen::core::Add::validate_and_infer_types()
{
    validate_and_infer_elementwise_arithmetic(get_autobroadcast());
}

void ngraph::op::gen::core::Add::generate_adjoints(autodiff::Adjoints& adjoints,
                                                   const NodeVector& deltas)
{
    if (get_autobroadcast().m_type != op::AutoBroadcastType::NONE)
    {
        throw ngraph_error("Autodiff not supported with auto broadcasting");
    }

    auto delta = deltas.at(0);

    auto x = get_x().get_source_output();
    auto y = get_y().get_source_output();

    adjoints.add_delta(x, delta);
    adjoints.add_delta(y, delta);
}
