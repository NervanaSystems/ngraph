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

#include <memory>
#include <vector>

#include "ngraph/runtime/parameterized_tensor_view.hpp"
#include "ngraph/runtime/tuple.hpp"
#include "ngraph/runtime/value.hpp"
#include "ngraph/types/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        /// @brief Framework constructor of a tensor of a specific element type and shape.
        template <typename ET>
        std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>
            make_tensor(const Shape& shape)
        {
            return std::make_shared<runtime::ParameterizedTensorView<ET>>(shape);
        }

        /// @brief Framework constructor of a tuple from a sequence of values.
        std::shared_ptr<ngraph::runtime::Tuple>
            make_tuple(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& elements);
    }
}
