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

#include "ngraph/output_vector.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

OutputVector::OutputVector(const std::initializer_list<std::shared_ptr<Node>>& nodes)
    : OutputVector(std::vector<std::shared_ptr<Node>>(nodes.begin(), nodes.end()))
{
}

OutputVector::OutputVector(const std::vector<std::shared_ptr<Node>>& nodes)
    : std::vector<Output<Node>>(nodes.begin(), nodes.end())
{
}
