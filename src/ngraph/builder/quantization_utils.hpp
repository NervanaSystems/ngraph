//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
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

#include <limits>
#include <vector>
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace builder
    {
        namespace quantization_utils
        {
            std::shared_ptr<Node> max_abs(std::shared_ptr<Node> a, std::shared_ptr<Node> b);

            std::shared_ptr<Node> get_scale(std::shared_ptr<Node> input_min_range,
                                            std::shared_ptr<Node> input_max_range,
                                            const ngraph::element::Type& quant_type,
                                            bool bump_by_eps = false);
        }
    }
}
