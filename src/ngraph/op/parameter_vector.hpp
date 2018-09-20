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

#include <memory>
#include <vector>

#include "ngraph/op/parameter.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Zero or more nodes.
        class ParameterVector : public std::vector<std::shared_ptr<op::Parameter>>
        {
        public:
            ParameterVector(const std::initializer_list<std::shared_ptr<op::Parameter>>& parameters)
                : std::vector<std::shared_ptr<op::Parameter>>(parameters)
            {
            }

            ParameterVector(const std::vector<std::shared_ptr<op::Parameter>>& parameters)
                : std::vector<std::shared_ptr<op::Parameter>>(parameters)
            {
            }

            ParameterVector(const ParameterVector& parameters)
                : std::vector<std::shared_ptr<op::Parameter>>(parameters)
            {
            }

            ParameterVector& operator=(const ParameterVector& parameters) = default;

            ParameterVector() {}
        };
    }
}
