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

#include "ngraph/builder/reduce.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/types/type.hpp"
#include "ngraph/function.hpp"

using namespace std;

namespace ngraph
{
    namespace builder
    {
        template <typename T>
        std::shared_ptr<Node> create_reduction(
            const std::shared_ptr<Node>& node, const std::string& init_val,
            const AxisSet& reduction_axes) 
        {
            const auto& et = node->get_element_type();

            auto f_A = make_shared<op::Parameter>(et, Shape{});
            auto f_B = make_shared<op::Parameter>(et, Shape{});
            auto f_rt = make_shared<TensorViewType>(et, Shape{});
            auto f = make_shared<Function>(make_shared<T>(f_A, f_B), f_rt,
                                           op::Parameters{f_A, f_B});

            auto init = make_shared<op::Constant>(et, Shape{}, init_val);

            return make_shared<op::Reduce>(node, init, f, reduction_axes);
        }
    } // namespace builder
} // namespace ngraph
