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

#include "ngraph/op/activate.hpp"
#include <memory>

using namespace std;
using namespace ngraph;

std::shared_ptr<ngraph::Node>
    ngraph::op::Activate::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() == 1)
    {
        return make_shared<op::Activate>(new_args.at(0), m_state);
    }

    return make_shared<op::Activate>(m_state);
}
