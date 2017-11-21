/*
 Copyright 2017 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include "ngraph/builder/sum.hpp"
#include "ngraph/builder/reduce.hpp"
#include "ngraph/ops/add.hpp"

using namespace std;

namespace ngraph
{
    namespace builder
    {
        std::shared_ptr<Node> Sum(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes)
        {
            return create_reduction<op::Add>(node, "0", reduction_axes);
        }
    } // namespace builder
} // namespace ngraph
